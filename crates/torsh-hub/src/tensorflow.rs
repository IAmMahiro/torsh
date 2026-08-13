//! TensorFlow model support for ToRSh Hub
//!
//! This module provides functionality to load and convert TensorFlow models
//! for use within the ToRSh ecosystem.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// TensorFlow model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfModelMetadata {
    pub model_name: String,
    pub version: String,
    pub description: Option<String>,
    pub framework_version: String,
    pub input_shapes: Vec<TfTensorInfo>,
    pub output_shapes: Vec<TfTensorInfo>,
    pub model_type: TfModelType,
}

/// TensorFlow tensor information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfTensorInfo {
    pub name: String,
    pub shape: Vec<Option<i64>>, // None for dynamic dimensions
    pub dtype: String,
}

/// TensorFlow model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TfModelType {
    SavedModel,
    FrozenGraph,
    Checkpoint,
    TensorFlowLite,
    TensorFlowJS,
}

/// TensorFlow model wrapper for ToRSh integration
pub struct TfModel {
    #[cfg(feature = "tensorflow")]
    session: tensorflow::Session,
    #[cfg(not(feature = "tensorflow"))]
    _placeholder: (),
    input_names: Vec<String>,
    output_names: Vec<String>,
    metadata: TfModelMetadata,
}

/// TensorFlow Runtime configuration
#[derive(Debug, Clone)]
pub struct TfConfig {
    pub allow_growth: bool,
    pub memory_limit: Option<usize>,
    pub device_placement: bool,
    pub inter_op_parallelism_threads: Option<i32>,
    pub intra_op_parallelism_threads: Option<i32>,
    pub use_gpu: bool,
    pub gpu_memory_fraction: Option<f64>,
}

impl Default for TfConfig {
    fn default() -> Self {
        Self {
            allow_growth: true,
            memory_limit: None,
            device_placement: true,
            inter_op_parallelism_threads: None,
            intra_op_parallelism_threads: None,
            use_gpu: false,
            gpu_memory_fraction: None,
        }
    }
}

impl TfModel {
    /// Load a TensorFlow SavedModel from directory
    pub fn from_saved_model<P: AsRef<Path>>(
        path: P,
        tags: &[&str],
        _config: Option<TfConfig>, // Reserved for future TensorFlow configuration
    ) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(TorshError::IoError(format!(
                "TensorFlow model directory not found: {:?}",
                path
            )));
        }

        #[cfg(feature = "tensorflow")]
        {
            use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs};

            // Create session options from config
            let mut session_opts = SessionOptions::new();

            // Load the saved model
            let mut graph = Graph::new();
            let path_str = path.to_str().ok_or_else(|| {
                TorshError::Other("Model path contains invalid UTF-8".to_string())
            })?;
            let session = Session::from_saved_model(&session_opts, tags, &mut graph, path_str)
                .map_err(|e| {
                    TorshError::Other(format!("Failed to load TensorFlow model: {}", e))
                })?;

            // Extract input and output information
            let input_names = Self::extract_input_names(&graph)?;
            let output_names = Self::extract_output_names(&graph)?;

            // Extract metadata
            let metadata = Self::extract_metadata(&graph, path)?;

            Ok(Self {
                session,
                input_names,
                output_names,
                metadata,
            })
        }

        #[cfg(not(feature = "tensorflow"))]
        {
            // The `tensorflow` feature binds the upstream `tensorflow` crate,
            // which links the TensorFlow C++ runtime. ToRSh ships pure Rust by
            // default, so without that feature there is no session to build and
            // the honest answer is an explicit error rather than a fake model.
            let _ = tags;
            Err(TorshError::Other(
                "TensorFlow support not enabled. Enable the 'tensorflow' feature to use TensorFlow models".to_string(),
            ))
        }
    }

    /// Load a TensorFlow frozen graph from file
    pub fn from_frozen_graph<P: AsRef<Path>>(path: P, _config: Option<TfConfig>) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(TorshError::IoError(format!(
                "TensorFlow frozen graph file not found: {:?}",
                path
            )));
        }

        #[cfg(feature = "tensorflow")]
        {
            use std::fs;
            use tensorflow::{Graph, Session, SessionOptions};

            // Read the frozen graph
            let graph_def = fs::read(path)?;

            // Create graph and import the graph def
            let mut graph = Graph::new();
            graph
                .import_graph_def(&graph_def, &tensorflow::ImportGraphDefOptions::new())
                .map_err(|e| TorshError::Other(format!("Failed to import graph: {}", e)))?;

            // Create session
            let session_opts = SessionOptions::new();
            let session = Session::new(&session_opts, &graph)
                .map_err(|e| TorshError::Other(format!("Failed to create session: {}", e)))?;

            // Extract input and output information
            let input_names = Self::extract_input_names(&graph)?;
            let output_names = Self::extract_output_names(&graph)?;

            // Extract metadata
            let metadata = Self::extract_metadata(&graph, path)?;

            Ok(Self {
                session,
                input_names,
                output_names,
                metadata,
            })
        }

        #[cfg(not(feature = "tensorflow"))]
        {
            Err(TorshError::Other(
                "TensorFlow support not enabled. Enable the 'tensorflow' feature to use TensorFlow models".to_string(),
            ))
        }
    }

    /// Run inference on the model
    pub fn forward(
        &self,
        inputs: &HashMap<String, Tensor<f32>>,
    ) -> Result<HashMap<String, Tensor<f32>>> {
        #[cfg(feature = "tensorflow")]
        {
            use tensorflow::{SessionRunArgs, Tensor as TfTensor};

            // Convert ToRSh tensors to TensorFlow tensors
            let mut args = SessionRunArgs::new();

            for (name, tensor) in inputs {
                let tf_tensor = self.tensor_to_tf_tensor(tensor)?;
                args.add_feed(
                    &self.session.graph().operation_by_name_required(name)?,
                    0,
                    &tf_tensor,
                );
            }

            // Add outputs to fetch
            for output_name in &self.output_names {
                let output_op = self
                    .session
                    .graph()
                    .operation_by_name_required(output_name)?;
                args.add_fetch(&output_op, 0);
            }

            // Run the session
            let outputs = self
                .session
                .run(&mut args)
                .map_err(|e| TorshError::Other(format!("TensorFlow inference failed: {}", e)))?;

            // Convert TensorFlow tensors back to ToRSh tensors
            let mut result = HashMap::new();
            for (i, output_name) in self.output_names.iter().enumerate() {
                let tf_tensor = outputs.fetch(i)?;
                let tensor = self.tf_tensor_to_tensor(tf_tensor)?;
                result.insert(output_name.clone(), tensor);
            }

            Ok(result)
        }

        #[cfg(not(feature = "tensorflow"))]
        {
            let _ = inputs;
            Err(TorshError::Other(
                "TensorFlow support not enabled".to_string(),
            ))
        }
    }

    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Get model metadata
    pub fn metadata(&self) -> &TfModelMetadata {
        &self.metadata
    }

    #[cfg(feature = "tensorflow")]
    fn extract_input_names(graph: &tensorflow::Graph) -> Result<Vec<String>> {
        // This is a simplified implementation
        // In practice, you'd parse the graph to find actual input operations
        Ok(vec!["input".to_string()])
    }

    #[cfg(feature = "tensorflow")]
    fn extract_output_names(graph: &tensorflow::Graph) -> Result<Vec<String>> {
        // This is a simplified implementation
        // In practice, you'd parse the graph to find actual output operations
        Ok(vec!["output".to_string()])
    }

    #[cfg(feature = "tensorflow")]
    fn extract_metadata(graph: &tensorflow::Graph, model_path: &Path) -> Result<TfModelMetadata> {
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("tf_model")
            .to_string();

        Ok(TfModelMetadata {
            model_name,
            version: "1.0".to_string(),
            description: Some(format!("TensorFlow model loaded from {:?}", model_path)),
            framework_version: "2.x".to_string(),
            input_shapes: vec![TfTensorInfo {
                name: "input".to_string(),
                shape: vec![None, Some(224), Some(224), Some(3)],
                dtype: "float32".to_string(),
            }],
            output_shapes: vec![TfTensorInfo {
                name: "output".to_string(),
                shape: vec![None, Some(1000)],
                dtype: "float32".to_string(),
            }],
            model_type: TfModelType::SavedModel,
        })
    }

    #[cfg(feature = "tensorflow")]
    fn tensor_to_tf_tensor(&self, tensor: &Tensor<f32>) -> Result<tensorflow::Tensor<f32>> {
        use tensorflow::Tensor as TfTensor;

        // Get tensor data and shape
        let shape = tensor.shape().dims();
        let data: Vec<f32> = tensor.to_vec();

        // Create TensorFlow tensor
        TfTensor::new(&shape.iter().map(|&x| x as u64).collect::<Vec<_>>())
            .with_values(&data)
            .map_err(|e| TorshError::Other(format!("Failed to create TensorFlow tensor: {}", e)))
    }

    #[cfg(feature = "tensorflow")]
    fn tf_tensor_to_tensor(&self, tf_tensor: &tensorflow::Tensor<f32>) -> Result<Tensor<f32>> {
        use torsh_tensor::creation::from_vec;

        // Get shape and data from TensorFlow tensor
        let shape: Vec<usize> = tf_tensor.dims().iter().map(|&x| x as usize).collect();
        let data: Vec<f32> = tf_tensor.iter().cloned().collect();

        // Create ToRSh tensor
        from_vec(data, &shape)
    }
}

/// TensorFlow model loader utility functions
pub struct TfLoader;

impl TfLoader {
    /// Download and load TensorFlow model from URL
    pub async fn from_url(url: &str, config: Option<TfConfig>) -> Result<TfModel> {
        use crate::download::download_file;

        // Create temporary directory for the model
        let temp_dir = tempfile::tempdir()
            .map_err(|e| TorshError::IoError(format!("Failed to create temp directory: {}", e)))?;

        let model_path = temp_dir.path().join("model");
        std::fs::create_dir_all(&model_path)?;

        // Download the model (assuming it's a compressed archive). No
        // registry checksum is available for an arbitrary caller-supplied
        // URL, so integrity verification is not requested here.
        let archive_path = temp_dir.path().join("model.tar.gz");
        download_file(url, &archive_path, true, None)?;

        // Extract the archive
        Self::extract_archive(&archive_path, &model_path)?;

        // Load the model
        TfModel::from_saved_model(&model_path, &["serve"], config)
    }

    /// Load TensorFlow model from ToRSh Hub
    pub fn from_hub(repo: &str, model_name: &str, config: Option<TfConfig>) -> Result<TfModel> {
        use crate::{download_repo, parse_repo_info, HubConfig};

        let hub_config = HubConfig::default();

        // Parse repository information
        let (owner, repo_name, branch) = parse_repo_info(repo)?;

        // Download repository
        let repo_dir = download_repo(&owner, &repo_name, &branch, &hub_config)?;

        // Look for TensorFlow model directory
        let model_path = repo_dir.join(model_name);
        if !model_path.exists() {
            return Err(TorshError::IoError(format!(
                "TensorFlow model directory not found: {:?}",
                model_path
            )));
        }

        // Load the model
        TfModel::from_saved_model(&model_path, &["serve"], config)
    }

    /// List available TensorFlow models in a repository
    pub fn list_models_in_repo(repo_dir: &Path) -> Result<Vec<String>> {
        let mut models = Vec::new();

        if !repo_dir.exists() {
            return Ok(models);
        }

        // Look for directories that contain saved_model.pb
        for entry in std::fs::read_dir(repo_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let saved_model_file = path.join("saved_model.pb");
                if saved_model_file.exists() {
                    if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                        models.push(dir_name.to_string());
                    }
                }
            }
        }

        Ok(models)
    }

    /// Validate TensorFlow model directory
    pub fn validate_model(path: &Path) -> Result<bool> {
        if !path.exists() || !path.is_dir() {
            return Ok(false);
        }

        // Check for saved_model.pb
        let saved_model_file = path.join("saved_model.pb");
        if !saved_model_file.exists() {
            return Ok(false);
        }

        // Check for variables directory
        let variables_dir = path.join("variables");
        if !variables_dir.exists() || !variables_dir.is_dir() {
            return Ok(false);
        }

        Ok(true)
    }

    /// Extract archive to destination
    fn extract_archive(archive_path: &Path, dest_path: &Path) -> Result<()> {
        use oxiarc_archive::TarStreamReader;
        use oxiarc_core::EntryType;
        use oxiarc_deflate::GzipStreamDecoder;
        use std::fs::File;
        use std::io;

        let file = File::open(archive_path)?;
        let decoder = GzipStreamDecoder::new(file);
        let mut stream = TarStreamReader::new(decoder);
        while let Some(mut entry) = stream
            .next_entry()
            .map_err(|e| TorshError::IoError(format!("Failed to read tar entry: {}", e)))?
        {
            // Reject entries that would escape `dest_path` ("tar-slip"),
            // matching the same guard used in `download::parallel::extract_tarball`.
            let dest = crate::utils::sanitize_archive_entry_path(dest_path, &entry.header.name)?;
            match entry.header.entry_type() {
                EntryType::Directory => {
                    std::fs::create_dir_all(&dest)
                        .map_err(|e| TorshError::IoError(format!("Failed to create dir: {}", e)))?;
                }
                EntryType::File => {
                    if let Some(parent) = dest.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            TorshError::IoError(format!("Failed to create parent dir: {}", e))
                        })?;
                    }
                    let mut out = File::create(&dest).map_err(|e| {
                        TorshError::IoError(format!("Failed to create file: {}", e))
                    })?;
                    io::copy(&mut entry, &mut out)
                        .map_err(|e| TorshError::IoError(format!("Failed to write file: {}", e)))?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Convert TensorFlow model to ToRSh Module (wrapper)
pub struct TfToTorshWrapper {
    tf_model: TfModel,
}

impl TfToTorshWrapper {
    pub fn new(tf_model: TfModel) -> Self {
        Self { tf_model }
    }
}

impl torsh_nn::Module for TfToTorshWrapper {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For simplicity, assume single input/output
        if self.tf_model.input_names.len() != 1 || self.tf_model.output_names.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "TfToTorshWrapper only supports models with single input and output".to_string(),
            ));
        }

        let input_name = &self.tf_model.input_names[0];
        let output_name = &self.tf_model.output_names[0];

        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input.clone());

        let outputs = self.tf_model.forward(&inputs)?;

        Ok(outputs
            .get(output_name)
            .ok_or_else(|| TorshError::Other("Missing output tensor".to_string()))?
            .clone())
    }

    fn parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        // A loaded TensorFlow graph owns its own variables; none of them are
        // exposed as ToRSh parameters, so this wrapper is inference-only.
        HashMap::new()
    }

    fn named_parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        HashMap::new()
    }

    fn train(&mut self) {
        // TensorFlow models loaded this way are inference-only
    }

    fn eval(&mut self) {
        // TensorFlow models are always in eval mode when loaded
    }

    fn training(&self) -> bool {
        false
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        Err(TorshError::Other(
            "TensorFlow models don't support state dict loading in this wrapper".to_string(),
        ))
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        // TensorFlow models don't support state dict saving in this wrapper
        HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tf_config_default() {
        let config = TfConfig::default();
        assert!(config.allow_growth);
        assert!(config.device_placement);
        assert!(!config.use_gpu);
        assert!(config.memory_limit.is_none());
    }

    #[test]
    fn test_tf_loader_validate_model_nonexistent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("nonexistent");

        let result = TfLoader::validate_model(&path);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_tf_tensor_info() {
        let tensor_info = TfTensorInfo {
            name: "input".to_string(),
            shape: vec![None, Some(224), Some(224), Some(3)],
            dtype: "float32".to_string(),
        };

        assert_eq!(tensor_info.name, "input");
        assert_eq!(tensor_info.shape.len(), 4);
        assert_eq!(tensor_info.dtype, "float32");
    }

    #[test]
    fn test_tf_metadata() {
        let metadata = TfModelMetadata {
            model_name: "test_model".to_string(),
            version: "1.0".to_string(),
            description: Some("Test TensorFlow model".to_string()),
            framework_version: "2.8.0".to_string(),
            input_shapes: vec![],
            output_shapes: vec![],
            model_type: TfModelType::SavedModel,
        };

        assert_eq!(metadata.model_name, "test_model");
        assert_eq!(metadata.version, "1.0");
        assert_eq!(metadata.framework_version, "2.8.0");
        assert!(matches!(metadata.model_type, TfModelType::SavedModel));
    }

    /// Build a `.tar.gz` byte stream containing a single entry named
    /// `entry_name` with the given `data`, mirroring
    /// `download::parallel`'s test helper of the same shape (kept local to
    /// this file since it is test-only and the two extraction sites are
    /// otherwise independent).
    fn build_tar_gz(entry_name: &str, data: &[u8]) -> Vec<u8> {
        use oxiarc_archive::TarWriter;

        let mut tar_bytes = Vec::new();
        {
            let mut writer = TarWriter::new(&mut tar_bytes);
            writer.add_file(entry_name, data).expect(
                "crafting a tar entry must itself succeed (writer does not sanitise names)",
            );
            writer.finish().expect("tar finish should succeed");
        }
        oxiarc_deflate::gzip_compress(&tar_bytes, 6).expect("gzip compression should succeed")
    }

    /// F022 (tar-slip site 2): `TfLoader::extract_archive` must reject a
    /// tar entry whose name contains a `..` component rather than writing
    /// outside `dest_path`.
    #[test]
    fn f022_extract_archive_rejects_parent_dir_traversal() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let archive_path = temp_dir.path().join("evil.tar.gz");
        let dest_path = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_path).expect("create dest dir");

        let archive_bytes = build_tar_gz("../../evil_outside.txt", b"pwned-by-tar-slip");
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        let result = TfLoader::extract_archive(&archive_path, &dest_path);
        assert!(
            result.is_err(),
            "extract_archive must reject a '..'-escaping entry name"
        );
        assert!(!temp_dir.path().join("evil_outside.txt").exists());
    }

    /// F022 (tar-slip site 2): `TfLoader::extract_archive` must reject an
    /// absolute tar entry path rather than extracting it verbatim.
    #[test]
    fn f022_extract_archive_rejects_absolute_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let archive_path = temp_dir.path().join("evil_abs.tar.gz");
        let dest_path = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_path).expect("create dest dir");

        let absolute_name = "/torsh_hardening_test_tf_absolute_evil.txt";
        let archive_bytes = build_tar_gz(absolute_name, b"pwned-by-absolute-path");
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        let result = TfLoader::extract_archive(&archive_path, &dest_path);
        assert!(
            result.is_err(),
            "extract_archive must reject an absolute entry path"
        );
        assert!(!Path::new(absolute_name).exists());
    }

    /// Control case: a well-behaved relative entry must still extract
    /// normally under `dest_path` after the sanitisation fix.
    #[test]
    fn f022_extract_archive_accepts_benign_relative_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let archive_path = temp_dir.path().join("benign.tar.gz");
        let dest_path = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_path).expect("create dest dir");

        let archive_bytes = build_tar_gz("variables/model.bin", b"legit-content");
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        TfLoader::extract_archive(&archive_path, &dest_path)
            .expect("benign entry should extract cleanly");

        let extracted = dest_path.join("variables").join("model.bin");
        assert!(extracted.exists());
        assert_eq!(
            std::fs::read(&extracted).expect("read extracted file"),
            b"legit-content"
        );
    }

    /// F022 (tar-slip site 2, symlink case): a `Symlink` tar entry must
    /// never be materialised on disk, matching the guard proven for
    /// `download::parallel::extract_tarball`. A lexically-clean entry
    /// *name* combined with a malicious symlink *target* is exactly the
    /// case a path-only guard would miss.
    #[test]
    fn f022_extract_archive_never_materialises_symlink_entries() {
        use oxiarc_archive::TarWriter;

        let temp_dir = tempfile::tempdir().expect("temp dir");
        let archive_path = temp_dir.path().join("symlink_evil.tar.gz");
        let dest_path = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_path).expect("create dest dir");

        let mut tar_bytes = Vec::new();
        {
            let mut writer = TarWriter::new(&mut tar_bytes);
            writer
                .add_symlink("escape_link", "../../../../../../../../tmp")
                .expect("writing a symlink entry must itself succeed");
            writer
                .add_file("escape_link/pwned.txt", b"pwned-via-symlink")
                .expect("writing the follow-up file entry must itself succeed");
            writer.finish().expect("tar finish should succeed");
        }
        let archive_bytes =
            oxiarc_deflate::gzip_compress(&tar_bytes, 6).expect("gzip compression should succeed");
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        TfLoader::extract_archive(&archive_path, &dest_path).expect(
            "a Symlink entry must be silently skipped (not rejected), so extraction of the \
             remaining benign entries still succeeds",
        );

        let escape_link_path = dest_path.join("escape_link");
        if let Ok(meta) = std::fs::symlink_metadata(&escape_link_path) {
            assert!(
                !meta.file_type().is_symlink(),
                "a tar Symlink entry must never be materialised as a real filesystem symlink"
            );
        }

        let nested_file = escape_link_path.join("pwned.txt");
        assert!(nested_file.exists());
        assert_eq!(
            std::fs::read(&nested_file).expect("read nested file"),
            b"pwned-via-symlink"
        );
        assert!(nested_file.starts_with(&dest_path));
    }
}
