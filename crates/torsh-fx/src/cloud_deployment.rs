//! Cloud Deployment Tools and Integrations
//!
//! This module provides utilities for deploying FX graphs to various cloud platforms
//! including AWS, Google Cloud Platform (GCP), and Microsoft Azure. It handles:
//! - Model packaging for cloud deployment
//! - Container image generation (Docker)
//! - Cloud-specific configuration files
//! - Serverless deployment configurations
//! - Auto-scaling and load balancing setup
//! - Monitoring and logging integration

use crate::model_zoo::{ModelMetadata, ModelZooEntry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

/// Graph executor emitted into the generated inference server.
///
/// The exported `model.json` carries the serialized FX graph, so the server
/// executes that graph with numpy instead of guessing an output. Operations whose
/// parameters are not part of the export (linear, conv2d, batch_norm, ...) raise
/// `NotImplementedError`, which the prediction endpoint turns into HTTP 501 — a
/// generated service must never substitute an identity for the model.
const GRAPH_EXECUTOR_PY: &str = r#"
def _predecessors(node_id):
    """Return the ids feeding a node, in edge declaration order."""
    return [src for src, dst in GRAPH_EDGES if dst == node_id]


def _topological_order():
    """Return graph node ids in dependency order."""
    indegree = {node_id: 0 for node_id in GRAPH_NODES}
    for _, dst in GRAPH_EDGES:
        if dst in indegree:
            indegree[dst] += 1
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    order = []
    while ready:
        current = ready.pop()
        order.append(current)
        for src, dst in GRAPH_EDGES:
            if src == current and dst in indegree:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
    if len(order) != len(GRAPH_NODES):
        raise ValueError('exported graph contains a cycle and cannot be executed')
    return order


def _apply_op(op_name, operands):
    """Evaluate a single graph operation with numpy."""
    if op_name in ('relu', 'relu_inplace'):
        return np.maximum(operands[0], 0.0)
    if op_name in ('sigmoid', 'sigmoid_inplace'):
        return 1.0 / (1.0 + np.exp(-operands[0]))
    if op_name in ('tanh', 'tanh_inplace'):
        return np.tanh(operands[0])
    if op_name == 'gelu':
        x = operands[0]
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
        return 0.5 * x * (1.0 + np.tanh(inner))
    if op_name == 'softmax':
        x = operands[0]
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)
    if op_name == 'exp':
        return np.exp(operands[0])
    if op_name == 'log':
        return np.log(operands[0])
    if op_name == 'sqrt':
        return np.sqrt(operands[0])
    if op_name == 'neg':
        return -operands[0]
    if op_name == 'abs':
        return np.abs(operands[0])
    if op_name in ('add', 'add_inplace'):
        return operands[0] + operands[1]
    if op_name == 'sub':
        return operands[0] - operands[1]
    if op_name in ('mul', 'mul_inplace'):
        return operands[0] * operands[1]
    if op_name == 'div':
        return operands[0] / operands[1]
    if op_name == 'matmul':
        return np.matmul(operands[0], operands[1])
    if op_name == 'identity':
        return operands[0]
    if op_name == 'constant_zero':
        return np.zeros(1, dtype=np.float32)
    if op_name == 'constant_one':
        return np.ones(1, dtype=np.float32)
    raise NotImplementedError(
        "operation '" + op_name + "' cannot be executed from the exported graph: it "
        "requires model parameters that are not part of model.json. Supply a runtime "
        "that provides the weights for this operation, or export a graph that only "
        "uses parameter-free operations."
    )


def run_graph(input_arrays):
    """Execute the exported graph and return its outputs as nested lists."""
    values = {}
    graph_inputs = GRAPH['inputs']
    if len(input_arrays) != len(graph_inputs):
        raise ValueError(
            'expected %d input(s), received %d' % (len(graph_inputs), len(input_arrays))
        )
    for node_id, array in zip(graph_inputs, input_arrays):
        values[node_id] = array

    for node_id in _topological_order():
        if node_id in values:
            continue
        node = GRAPH_NODES.get(node_id)
        if node is None:
            raise ValueError('graph references unknown node %r' % (node_id,))
        node_type = node['node_type']
        if node_type.startswith('input:'):
            raise ValueError('no value supplied for graph input %r' % (node_id,))
        preds = _predecessors(node_id)
        if node_type == 'output':
            if preds:
                values[node_id] = values[preds[0]]
            continue
        if node_type.startswith('call:'):
            op_name = node_type.split(':', 1)[1]
            if op_name == 'constant':
                params = node.get('params', {})
                values[node_id] = np.asarray(
                    [float(params.get('value', 0.0))], dtype=np.float32
                )
                continue
            values[node_id] = _apply_op(op_name, [values[p] for p in preds])
            continue
        raise NotImplementedError(
            "graph node type '" + node_type + "' is not supported by the generated server"
        )

    return [
        np.asarray(values[node_id]).tolist()
        for node_id in GRAPH['outputs']
        if node_id in values
    ]


"#;

/// Prediction endpoint emitted into the generated inference server.
///
/// Delegates to [`GRAPH_EXECUTOR_PY`]'s `run_graph` and maps a refusal to HTTP 501
/// so an unrunnable model is visible instead of silently echoing the request.
const PREDICT_ENDPOINT_PY: &str = r#"@app.route('/predict', methods=['POST'])
def predict():
    global request_count, total_inference_time, error_count
    try:
        start_time = time.time()

        data = request.get_json()
        if not data or 'inputs' not in data:
            return jsonify({'error': 'Missing inputs'}), 400

        inputs = data.get('inputs')

        # Convert the request payload into the graph's input arrays
        if len(GRAPH['inputs']) == 1:
            input_arrays = [np.array(inputs, dtype=np.float32)]
        else:
            input_arrays = [np.array(item, dtype=np.float32) for item in inputs]

        # Run the exported graph
        outputs = run_graph(input_arrays)

        inference_time = time.time() - start_time
        request_count += 1
        total_inference_time += inference_time

        return jsonify({
            'outputs': outputs,
            'inference_time_ms': inference_time * 1000,
            'request_id': request_count
        })
    except NotImplementedError as exc:
        error_count += 1
        logger.error(f'Model cannot be executed: {exc}')
        return jsonify({'error': str(exc)}), 501
    except Exception as e:
        error_count += 1
        logger.error(f'Prediction error: {e}')
        return jsonify({'error': str(e)}), 500

"#;

/// Cloud deployment target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudPlatform {
    /// Amazon Web Services
    AWS { region: String, service: AWSService },
    /// Google Cloud Platform
    GCP {
        project_id: String,
        region: String,
        service: GCPService,
    },
    /// Microsoft Azure
    Azure {
        subscription_id: String,
        resource_group: String,
        region: String,
        service: AzureService,
    },
    /// Custom cloud platform
    Custom { name: String, endpoint: String },
}

/// AWS service options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AWSService {
    /// Amazon SageMaker
    SageMaker {
        instance_type: String,
        endpoint_name: String,
    },
    /// AWS Lambda (serverless)
    Lambda {
        runtime: String,
        memory_mb: usize,
        timeout_seconds: usize,
    },
    /// Amazon ECS (container service)
    ECS {
        cluster_name: String,
        task_definition: String,
    },
    /// Amazon EKS (Kubernetes)
    EKS {
        cluster_name: String,
        namespace: String,
    },
}

/// GCP service options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GCPService {
    /// Vertex AI
    VertexAI {
        model_name: String,
        machine_type: String,
    },
    /// Cloud Run (serverless)
    CloudRun {
        service_name: String,
        memory_mb: usize,
        max_instances: usize,
    },
    /// Google Kubernetes Engine
    GKE {
        cluster_name: String,
        namespace: String,
    },
    /// Cloud Functions
    CloudFunctions {
        function_name: String,
        runtime: String,
        memory_mb: usize,
    },
}

/// Azure service options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AzureService {
    /// Azure Machine Learning
    AzureML {
        workspace_name: String,
        endpoint_name: String,
    },
    /// Azure Functions (serverless)
    AzureFunctions {
        function_app_name: String,
        runtime: String,
    },
    /// Azure Kubernetes Service
    AKS {
        cluster_name: String,
        namespace: String,
    },
    /// Azure Container Instances
    ACI {
        container_group_name: String,
        cpu_cores: f32,
        memory_gb: f32,
    },
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Deployment name
    pub name: String,
    /// Cloud platform
    pub platform: CloudPlatform,
    /// Container configuration
    pub container: ContainerConfig,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Auto-scaling configuration
    pub autoscaling: Option<AutoScalingConfig>,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Monitoring and logging
    pub monitoring: MonitoringConfig,
}

/// Container configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Base Docker image
    pub base_image: String,
    /// Python version
    pub python_version: String,
    /// Additional system packages
    pub system_packages: Vec<String>,
    /// Python packages
    pub python_packages: Vec<String>,
    /// Entry point command
    pub entrypoint: Vec<String>,
    /// Port to expose
    pub port: u16,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores
    pub cpu_cores: f32,
    /// Memory in GB
    pub memory_gb: f32,
    /// GPU count
    pub gpu_count: u32,
    /// GPU type
    pub gpu_type: Option<String>,
    /// Storage in GB
    pub storage_gb: u32,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Minimum instances
    pub min_instances: u32,
    /// Maximum instances
    pub max_instances: u32,
    /// Target CPU utilization (percentage)
    pub target_cpu_utilization: f32,
    /// Target memory utilization (percentage)
    pub target_memory_utilization: f32,
    /// Scale-up cooldown (seconds)
    pub scale_up_cooldown: u32,
    /// Scale-down cooldown (seconds)
    pub scale_down_cooldown: u32,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check path
    pub path: String,
    /// Check interval (seconds)
    pub interval_seconds: u32,
    /// Timeout (seconds)
    pub timeout_seconds: u32,
    /// Healthy threshold
    pub healthy_threshold: u32,
    /// Unhealthy threshold
    pub unhealthy_threshold: u32,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable logging
    pub enable_logging: bool,
    /// Enable distributed tracing
    pub enable_tracing: bool,
    /// Metrics endpoint
    pub metrics_endpoint: Option<String>,
    /// Log level
    pub log_level: String,
    /// Custom metrics
    pub custom_metrics: Vec<String>,
}

/// Cloud deployment packager
pub struct CloudDeploymentPackager {
    /// Output directory
    output_dir: PathBuf,
    /// Deployment configuration
    config: DeploymentConfig,
}

impl CloudDeploymentPackager {
    /// Create a new deployment packager
    pub fn new<P: AsRef<Path>>(output_dir: P, config: DeploymentConfig) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();

        // Create output directory
        fs::create_dir_all(&output_dir).map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(Self { output_dir, config })
    }

    /// Package model for deployment
    pub fn package_model(&self, entry: &ModelZooEntry) -> Result<DeploymentPackage> {
        // Create deployment structure
        let deployment_dir = self.output_dir.join(&self.config.name);
        fs::create_dir_all(&deployment_dir).map_err(|e| TorshError::IoError(e.to_string()))?;

        // Generate Dockerfile
        let dockerfile = self.generate_dockerfile(entry)?;
        fs::write(deployment_dir.join("Dockerfile"), dockerfile)
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        // Generate inference server
        let server_code = self.generate_inference_server(entry)?;
        fs::write(deployment_dir.join("server.py"), server_code)
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        // Generate requirements.txt
        let requirements = self.generate_requirements()?;
        fs::write(deployment_dir.join("requirements.txt"), requirements)
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        // Generate platform-specific configuration
        let platform_config = self.generate_platform_config(entry)?;
        fs::write(
            deployment_dir.join("platform_config.json"),
            serde_json::to_string_pretty(&platform_config)
                .map_err(|e| TorshError::SerializationError(e.to_string()))?,
        )
        .map_err(|e| TorshError::IoError(e.to_string()))?;

        // Generate deployment scripts
        let deploy_script = self.generate_deployment_script()?;
        fs::write(deployment_dir.join("deploy.sh"), deploy_script)
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        // Copy model files
        entry.save_to_file(deployment_dir.join("model.json"))?;

        Ok(DeploymentPackage {
            path: deployment_dir,
            config: self.config.clone(),
            metadata: entry.metadata.clone(),
        })
    }

    /// Generate Dockerfile
    fn generate_dockerfile(&self, _entry: &ModelZooEntry) -> Result<String> {
        let mut dockerfile = String::new();

        dockerfile.push_str(&format!("FROM {}\n\n", self.config.container.base_image));

        dockerfile.push_str("WORKDIR /app\n\n");

        // Install system packages
        if !self.config.container.system_packages.is_empty() {
            dockerfile.push_str("RUN apt-get update && apt-get install -y \\\n");
            for pkg in &self.config.container.system_packages {
                dockerfile.push_str(&format!("    {} \\\n", pkg));
            }
            dockerfile.push_str("    && rm -rf /var/lib/apt/lists/*\n\n");
        }

        // Copy files
        dockerfile.push_str("COPY requirements.txt .\n");
        dockerfile.push_str("RUN pip install --no-cache-dir -r requirements.txt\n\n");
        dockerfile.push_str("COPY . .\n\n");

        // Expose port
        dockerfile.push_str(&format!("EXPOSE {}\n\n", self.config.container.port));

        // Health check
        dockerfile.push_str(&format!(
            "HEALTHCHECK --interval={}s --timeout={}s --start-period=30s --retries={} \\\n",
            self.config.health_check.interval_seconds,
            self.config.health_check.timeout_seconds,
            self.config.health_check.healthy_threshold
        ));
        dockerfile.push_str(&format!(
            "    CMD curl -f http://localhost:{}{} || exit 1\n\n",
            self.config.container.port, self.config.health_check.path
        ));

        // Entry point
        dockerfile.push_str("CMD ");
        dockerfile.push_str(
            &serde_json::to_string(&self.config.container.entrypoint)
                .map_err(|e| TorshError::SerializationError(e.to_string()))?,
        );
        dockerfile.push('\n');

        Ok(dockerfile)
    }

    /// Generate inference server code
    fn generate_inference_server(&self, entry: &ModelZooEntry) -> Result<String> {
        let mut server = String::new();

        server.push_str("#!/usr/bin/env python3\n");
        server.push_str("\"\"\"Inference server for ToRSh FX model deployment.\"\"\"\n\n");

        server.push_str("import json\n");
        server.push_str("import logging\n");
        server.push_str("import os\n");
        server.push_str("import time\n");
        server.push_str("from typing import Any, Dict, List\n\n");
        server.push_str("from flask import Flask, request, jsonify\n");
        server.push_str("import numpy as np\n\n");

        // Configure logging
        server.push_str(&format!(
            "logging.basicConfig(level=logging.{})\n",
            self.config.monitoring.log_level.to_uppercase()
        ));
        server.push_str("logger = logging.getLogger(__name__)\n\n");

        // Create Flask app
        server.push_str("app = Flask(__name__)\n\n");

        // Global metrics tracking
        server.push_str("# Global metrics\n");
        server.push_str("request_count = 0\n");
        server.push_str("total_inference_time = 0.0\n");
        server.push_str("error_count = 0\n");
        server.push_str("start_timestamp = time.time()\n\n");

        // Load model
        server.push_str("# Load model\n");
        server.push_str("logger.info('Loading model...')\n");
        server.push_str("with open('model.json', 'r') as f:\n");
        server.push_str("    model_data = json.load(f)\n");
        server.push_str("GRAPH = model_data['graph']\n");
        server.push_str("GRAPH_NODES = {node['id']: node for node in GRAPH['nodes']}\n");
        server.push_str("GRAPH_EDGES = [tuple(edge) for edge in GRAPH['edges']]\n");
        server.push_str(&format!(
            "logger.info('Loaded model: {}')\n\n",
            entry.metadata.name
        ));

        // Graph executor. The exported graph is executed for real; operations whose
        // parameters are not part of the export refuse loudly instead of returning
        // something that merely looks like a prediction.
        server.push_str(GRAPH_EXECUTOR_PY);

        // Health check endpoint
        server.push_str("@app.route('/health', methods=['GET'])\n");
        server.push_str("def health_check():\n");
        server.push_str("    return jsonify({'status': 'healthy'})\n\n");

        // Prediction endpoint
        server.push_str(PREDICT_ENDPOINT_PY);

        // Metrics endpoint
        if self.config.monitoring.enable_metrics {
            server.push_str("@app.route('/metrics', methods=['GET'])\n");
            server.push_str("def metrics():\n");
            server.push_str("    \"\"\"Return comprehensive server and model metrics.\"\"\"\n");
            server.push_str(
                "    global request_count, total_inference_time, error_count, start_timestamp\n",
            );
            server.push_str("    \n");
            server.push_str("    uptime = time.time() - start_timestamp\n");
            server.push_str("    avg_inference_time = (total_inference_time / request_count) if request_count > 0 else 0\n");
            server.push_str("    \n");
            server.push_str("    metrics_data = {\n");
            server.push_str("        'server': {\n");
            server.push_str("            'uptime_seconds': uptime,\n");
            server.push_str("            'uptime_hours': uptime / 3600,\n");
            server.push_str("            'start_time': start_timestamp\n");
            server.push_str("        },\n");
            server.push_str("        'requests': {\n");
            server.push_str("            'total': request_count,\n");
            server.push_str("            'errors': error_count,\n");
            server.push_str("            'success_rate': ((request_count - error_count) / request_count * 100) if request_count > 0 else 100,\n");
            server.push_str(
                "            'requests_per_second': request_count / uptime if uptime > 0 else 0\n",
            );
            server.push_str("        },\n");
            server.push_str("        'inference': {\n");
            server.push_str("            'total_time_seconds': total_inference_time,\n");
            server.push_str("            'average_time_ms': avg_inference_time * 1000,\n");
            server.push_str("            'throughput': request_count / total_inference_time if total_inference_time > 0 else 0\n");
            server.push_str("        },\n");
            server.push_str("        'system': {\n");
            server.push_str("            'memory_usage_mb': __import__('psutil').Process().memory_info().rss / 1024 / 1024 if __import__('importlib').util.find_spec('psutil') else 0,\n");
            server.push_str("            'cpu_percent': __import__('psutil').Process().cpu_percent() if __import__('importlib').util.find_spec('psutil') else 0\n");
            server.push_str("        }\n");
            server.push_str("    }\n");
            server.push_str("    \n");
            server.push_str("    return jsonify(metrics_data)\n\n");
        }

        // Main
        server.push_str("if __name__ == '__main__':\n");
        server.push_str(&format!(
            "    app.run(host='0.0.0.0', port={}, debug=False)\n",
            self.config.container.port
        ));

        Ok(server)
    }

    /// Generate requirements.txt
    fn generate_requirements(&self) -> Result<String> {
        let mut requirements = String::new();

        requirements.push_str("# Core dependencies\n");
        requirements.push_str("flask>=2.0.0\n");
        requirements.push_str("numpy>=1.20.0\n");
        requirements.push_str("torch>=2.0.0\n\n");

        requirements.push_str("# Additional packages\n");
        for pkg in &self.config.container.python_packages {
            requirements.push_str(&format!("{}\n", pkg));
        }

        if self.config.monitoring.enable_metrics {
            requirements.push_str("\n# Monitoring\n");
            requirements.push_str("prometheus-client>=0.14.0\n");
        }

        if self.config.monitoring.enable_tracing {
            requirements.push_str("opentelemetry-api>=1.0.0\n");
            requirements.push_str("opentelemetry-sdk>=1.0.0\n");
        }

        Ok(requirements)
    }

    /// Generate platform-specific configuration
    fn generate_platform_config(&self, _entry: &ModelZooEntry) -> Result<serde_json::Value> {
        match &self.config.platform {
            CloudPlatform::AWS { service, .. } => self.generate_aws_config(service),
            CloudPlatform::GCP { service, .. } => self.generate_gcp_config(service),
            CloudPlatform::Azure { service, .. } => self.generate_azure_config(service),
            CloudPlatform::Custom { .. } => Ok(serde_json::json!({"type": "custom"})),
        }
    }

    /// Generate AWS-specific configuration
    fn generate_aws_config(&self, service: &AWSService) -> Result<serde_json::Value> {
        match service {
            AWSService::SageMaker {
                instance_type,
                endpoint_name,
            } => Ok(serde_json::json!({
                "service": "sagemaker",
                "instance_type": instance_type,
                "endpoint_name": endpoint_name,
                "resources": {
                    "initial_instance_count": 1,
                }
            })),
            AWSService::Lambda {
                runtime,
                memory_mb,
                timeout_seconds,
            } => Ok(serde_json::json!({
                "service": "lambda",
                "runtime": runtime,
                "memory_mb": memory_mb,
                "timeout_seconds": timeout_seconds,
            })),
            AWSService::ECS {
                cluster_name,
                task_definition,
            } => Ok(serde_json::json!({
                "service": "ecs",
                "cluster_name": cluster_name,
                "task_definition": task_definition,
            })),
            AWSService::EKS {
                cluster_name,
                namespace,
            } => Ok(serde_json::json!({
                "service": "eks",
                "cluster_name": cluster_name,
                "namespace": namespace,
            })),
        }
    }

    /// Generate GCP-specific configuration
    fn generate_gcp_config(&self, service: &GCPService) -> Result<serde_json::Value> {
        match service {
            GCPService::VertexAI {
                model_name,
                machine_type,
            } => Ok(serde_json::json!({
                "service": "vertex_ai",
                "model_name": model_name,
                "machine_type": machine_type,
            })),
            GCPService::CloudRun {
                service_name,
                memory_mb,
                max_instances,
            } => Ok(serde_json::json!({
                "service": "cloud_run",
                "service_name": service_name,
                "memory_mb": memory_mb,
                "max_instances": max_instances,
            })),
            GCPService::GKE {
                cluster_name,
                namespace,
            } => Ok(serde_json::json!({
                "service": "gke",
                "cluster_name": cluster_name,
                "namespace": namespace,
            })),
            GCPService::CloudFunctions {
                function_name,
                runtime,
                memory_mb,
            } => Ok(serde_json::json!({
                "service": "cloud_functions",
                "function_name": function_name,
                "runtime": runtime,
                "memory_mb": memory_mb,
            })),
        }
    }

    /// Generate Azure-specific configuration
    fn generate_azure_config(&self, service: &AzureService) -> Result<serde_json::Value> {
        match service {
            AzureService::AzureML {
                workspace_name,
                endpoint_name,
            } => Ok(serde_json::json!({
                "service": "azure_ml",
                "workspace_name": workspace_name,
                "endpoint_name": endpoint_name,
            })),
            AzureService::AzureFunctions {
                function_app_name,
                runtime,
            } => Ok(serde_json::json!({
                "service": "azure_functions",
                "function_app_name": function_app_name,
                "runtime": runtime,
            })),
            AzureService::AKS {
                cluster_name,
                namespace,
            } => Ok(serde_json::json!({
                "service": "aks",
                "cluster_name": cluster_name,
                "namespace": namespace,
            })),
            AzureService::ACI {
                container_group_name,
                cpu_cores,
                memory_gb,
            } => Ok(serde_json::json!({
                "service": "aci",
                "container_group_name": container_group_name,
                "cpu_cores": cpu_cores,
                "memory_gb": memory_gb,
            })),
        }
    }

    /// Generate deployment script
    fn generate_deployment_script(&self) -> Result<String> {
        let mut script = String::new();

        script.push_str("#!/bin/bash\n");
        script.push_str("# Deployment script for ToRSh FX model\n\n");
        script.push_str("set -e\n\n");

        script.push_str("echo 'Building Docker image...'\n");
        script.push_str(&format!("docker build -t {} .\n\n", self.config.name));

        match &self.config.platform {
            CloudPlatform::AWS { region, .. } => {
                script.push_str("echo 'Deploying to AWS...'\n");
                script.push_str(&format!("export AWS_REGION={}\n", region));
                script.push_str("# Add AWS-specific deployment commands here\n\n");
            }
            CloudPlatform::GCP {
                project_id, region, ..
            } => {
                script.push_str("echo 'Deploying to GCP...'\n");
                script.push_str(&format!("export GCP_PROJECT={}\n", project_id));
                script.push_str(&format!("export GCP_REGION={}\n", region));
                script.push_str("# Add GCP-specific deployment commands here\n\n");
            }
            CloudPlatform::Azure {
                subscription_id, ..
            } => {
                script.push_str("echo 'Deploying to Azure...'\n");
                script.push_str(&format!(
                    "export AZURE_SUBSCRIPTION_ID={}\n",
                    subscription_id
                ));
                script.push_str("# Add Azure-specific deployment commands here\n\n");
            }
            CloudPlatform::Custom { endpoint, .. } => {
                script.push_str("echo 'Deploying to custom platform...'\n");
                script.push_str(&format!("export ENDPOINT={}\n", endpoint));
                script.push_str("# Add custom deployment commands here\n\n");
            }
        }

        script.push_str("echo 'Deployment complete!'\n");

        Ok(script)
    }
}

/// Deployment package
#[derive(Debug, Clone)]
pub struct DeploymentPackage {
    /// Package path
    pub path: PathBuf,
    /// Deployment configuration
    pub config: DeploymentConfig,
    /// Model metadata
    pub metadata: ModelMetadata,
}

impl DeploymentPackage {
    /// Get package path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get configuration
    pub fn config(&self) -> &DeploymentConfig {
        &self.config
    }

    /// Get metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// Default configurations for common deployment scenarios
impl DeploymentConfig {
    /// Create a default AWS SageMaker configuration
    pub fn aws_sagemaker(name: String, region: String) -> Self {
        Self {
            name,
            platform: CloudPlatform::AWS {
                region,
                service: AWSService::SageMaker {
                    instance_type: "ml.m5.xlarge".to_string(),
                    endpoint_name: "torsh-fx-endpoint".to_string(),
                },
            },
            container: ContainerConfig::default(),
            resources: ResourceRequirements::default(),
            autoscaling: Some(AutoScalingConfig::default()),
            environment_variables: HashMap::new(),
            health_check: HealthCheckConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }

    /// Create a default GCP Vertex AI configuration
    pub fn gcp_vertex_ai(name: String, project_id: String, region: String) -> Self {
        Self {
            name,
            platform: CloudPlatform::GCP {
                project_id,
                region,
                service: GCPService::VertexAI {
                    model_name: "torsh-fx-model".to_string(),
                    machine_type: "n1-standard-4".to_string(),
                },
            },
            container: ContainerConfig::default(),
            resources: ResourceRequirements::default(),
            autoscaling: Some(AutoScalingConfig::default()),
            environment_variables: HashMap::new(),
            health_check: HealthCheckConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            base_image: "python:3.11-slim".to_string(),
            python_version: "3.11".to_string(),
            system_packages: vec!["curl".to_string()],
            python_packages: Vec::new(),
            entrypoint: vec!["python".to_string(), "server.py".to_string()],
            port: 8080,
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 2.0,
            memory_gb: 4.0,
            gpu_count: 0,
            gpu_type: None,
            storage_gb: 10,
        }
    }
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            min_instances: 1,
            max_instances: 10,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            scale_up_cooldown: 60,
            scale_down_cooldown: 300,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            path: "/health".to_string(),
            interval_seconds: 30,
            timeout_seconds: 10,
            healthy_threshold: 2,
            unhealthy_threshold: 3,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_logging: true,
            enable_tracing: false,
            metrics_endpoint: None,
            log_level: "INFO".to_string(),
            custom_metrics: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_config_generation() {
        let config =
            DeploymentConfig::aws_sagemaker("test-deployment".to_string(), "us-east-1".to_string());

        assert_eq!(config.name, "test-deployment");
        matches!(config.platform, CloudPlatform::AWS { .. });
    }

    #[test]
    fn test_gcp_config_generation() {
        let config = DeploymentConfig::gcp_vertex_ai(
            "test-deployment".to_string(),
            "my-project".to_string(),
            "us-central1".to_string(),
        );

        assert_eq!(config.name, "test-deployment");
        matches!(config.platform, CloudPlatform::GCP { .. });
    }

    #[test]
    fn test_deployment_packager_creation() {
        let temp_dir = std::env::temp_dir().join("torsh_fx_cloud_deploy_test");
        let config = DeploymentConfig::aws_sagemaker("test".to_string(), "us-east-1".to_string());

        let result = CloudDeploymentPackager::new(&temp_dir, config);
        assert!(result.is_ok());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
