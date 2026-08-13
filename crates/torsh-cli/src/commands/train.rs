//! Training operation commands.
//!
//! `torsh train start` runs a **real** optimisation loop: it builds a real
//! neural network with [`torsh_nn`], computes a real loss, back-propagates
//! through [`torsh_autograd`], and updates parameters with a real
//! [`torsh_optim`] optimiser (see [`crate::commands::real_training`]).
//!
//! The current trainer targets a multi-layer perceptron on an
//! **explicitly-synthetic** regression task. It never fabricates losses or
//! gradients: every reported number is measured from the running model. Loading
//! arbitrary real datasets / architectures is not yet wired, so those requests
//! return an honest error rather than a fabricated loss curve.

use anyhow::Result;
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

use crate::commands::real_training::{self, MlpConfig};
use crate::config::Config;
use crate::utils::{output, progress, time, validation};

#[derive(Subcommand)]
pub enum TrainCommands {
    /// Start model training
    Start(StartArgs),

    /// Resume training from checkpoint
    Resume(ResumeArgs),

    /// Monitor training progress
    Monitor(MonitorArgs),

    /// Stop running training
    Stop(StopArgs),
}

#[derive(Args)]
pub struct StartArgs {
    /// Training configuration file (JSON)
    #[arg(short, long)]
    pub config: PathBuf,

    /// Dataset path
    #[arg(short, long)]
    pub data: PathBuf,

    /// Number of epochs (full-batch optimisation steps)
    #[arg(short, long, default_value = "10")]
    pub epochs: usize,

    /// Batch size (retained for CLI compatibility; the synthetic trainer uses full-batch steps)
    #[arg(short, long, default_value = "32")]
    pub batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.01")]
    pub learning_rate: f64,

    /// Enable distributed training
    #[arg(long)]
    pub distributed: bool,

    /// Device to use for training (cpu, cuda, metal)
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Optimizer to use (currently: sgd)
    #[arg(long, default_value = "sgd")]
    pub optimizer: String,

    /// Learning rate scheduler (constant, step, cosine)
    #[arg(long, default_value = "constant")]
    pub scheduler: String,

    /// Enable mixed precision training
    #[arg(long)]
    pub mixed_precision: bool,

    /// Gradient clipping threshold
    #[arg(long)]
    pub grad_clip: Option<f64>,

    /// Save checkpoint every N epochs
    #[arg(long, default_value = "5")]
    pub save_every: usize,

    /// Output directory for checkpoints and logs
    #[arg(short, long, default_value = "./runs")]
    pub output_dir: PathBuf,

    /// Allow the explicitly-synthetic fallback dataset when the requested data
    /// path cannot be loaded by the real data pipeline.
    #[arg(long)]
    pub allow_synthetic: bool,
}

#[derive(Args)]
pub struct ResumeArgs {
    /// Checkpoint file to resume from
    #[arg(short = 'k', long)]
    pub checkpoint: PathBuf,

    /// Override epochs
    #[arg(long)]
    pub epochs: Option<usize>,
}

#[derive(Args)]
pub struct MonitorArgs {
    /// Training run ID or log directory
    #[arg(short, long)]
    pub run: PathBuf,

    /// Follow logs in real-time
    #[arg(short, long)]
    pub follow: bool,
}

#[derive(Args)]
pub struct StopArgs {
    /// Training run ID
    #[arg(short, long)]
    pub run: String,

    /// Force stop without graceful shutdown
    #[arg(long)]
    pub force: bool,
}

pub async fn execute(command: TrainCommands, _config: &Config, _output_format: &str) -> Result<()> {
    match command {
        TrainCommands::Start(args) => start_training(args).await,
        TrainCommands::Resume(args) => resume_training(args).await,
        TrainCommands::Monitor(args) => monitor_training(args).await,
        TrainCommands::Stop(args) => stop_training(args).await,
    }
}

/// Model / training hyper-parameters recovered from the JSON config file.
#[derive(Debug, Clone)]
struct TrainingConfig {
    model_name: String,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    num_samples: usize,
}

/// Training metrics for monitoring, persisted as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingMetrics {
    run_id: String,
    /// Per-epoch training loss (real, measured MSE).
    train_losses: Vec<f64>,
    /// Per-epoch learning rate.
    learning_rates: Vec<f64>,
    /// Per-epoch wall-clock time in seconds.
    epoch_times: Vec<f64>,
    /// True when the trainer used the explicitly-synthetic fallback dataset.
    synthetic_data: bool,
}

/// Final results reported to the user.
#[derive(Debug, Clone)]
struct TrainingResults {
    run_id: String,
    epochs_completed: usize,
    initial_train_loss: f64,
    final_train_loss: f64,
    synthetic_data: bool,
    converged: bool,
}

async fn start_training(args: StartArgs) -> Result<()> {
    validation::validate_file_exists(&args.config)?;
    validation::validate_directory_exists(&args.data)?;
    validation::validate_device(&args.device)?;

    if args.optimizer.to_lowercase() != "sgd" {
        warn!(
            "optimizer '{}' is not yet wired into the real trainer; using SGD",
            args.optimizer
        );
    }

    let (results, total_duration) = time::measure_time(run_real_training(&args)).await;
    let results = results?;

    output::print_success("Training completed successfully!");
    output::print_info(&format!(
        "Total duration: {}",
        time::format_duration(total_duration)
    ));
    if results.synthetic_data {
        output::print_warning(
            "NOTE: trained on an EXPLICITLY-SYNTHETIC regression dataset (not the files in --data). \
             Reported losses are real measurements on that synthetic task.",
        );
    }
    output::print_info(&format!(
        "Initial training loss (MSE): {:.6}",
        results.initial_train_loss
    ));
    output::print_info(&format!(
        "Final training loss (MSE): {:.6}",
        results.final_train_loss
    ));
    output::print_info(&format!("Epochs completed: {}", results.epochs_completed));
    output::print_info(&format!("Run ID: {}", results.run_id));

    if results.converged {
        output::print_success("Training loss decreased over the run");
    } else {
        output::print_warning("Training loss did not decrease over the run");
    }

    Ok(())
}

/// Run the real training loop and persist real metrics/checkpoints.
async fn run_real_training(args: &StartArgs) -> Result<TrainingResults> {
    info!("Starting real ToRSh training loop");
    info!("Configuration: {}", args.config.display());

    let cfg = load_training_config(&args.config).await?;
    info!("Loaded training configuration: {}", cfg.model_name);

    // Currently only the explicitly-synthetic regression pipeline is wired end
    // to end. Refuse to fabricate results for an unsupported real-data request.
    if !args.allow_synthetic {
        return Err(anyhow::anyhow!(
            "the CLI trainer can currently only train on its explicitly-synthetic regression task; \
             re-run with --allow-synthetic to train on synthetic data, or use the library API to \
             train on your real dataset. It will not fabricate a loss curve for '{}'.",
            args.data.display()
        ));
    }

    let model = real_training::build_mlp(&MlpConfig {
        input_dim: cfg.input_dim,
        hidden_dim: cfg.hidden_dim,
        output_dim: cfg.output_dim,
    })?;
    info!(
        "Built real MLP ({}->{}->{})",
        cfg.input_dim, cfg.hidden_dim, cfg.output_dim
    );

    let data = real_training::synthetic_regression(
        cfg.num_samples,
        cfg.input_dim,
        cfg.output_dim,
        0x5eed_1234,
    )?;

    tokio::fs::create_dir_all(&args.output_dir).await?;
    let run_id = generate_run_id();
    let run_dir = args.output_dir.join(&run_id);
    tokio::fs::create_dir_all(&run_dir).await?;
    info!("Created training run directory: {}", run_dir.display());

    let epochs = args.epochs.max(1);
    let lr = args.learning_rate as f32;

    let mut metrics = TrainingMetrics {
        run_id: run_id.clone(),
        train_losses: Vec::with_capacity(epochs),
        learning_rates: Vec::with_capacity(epochs),
        epoch_times: Vec::with_capacity(epochs),
        synthetic_data: true,
    };

    let initial_train_loss = real_training::evaluate_loss(&model, &data)?;

    let pb = progress::create_progress_bar(epochs as u64, "Training");
    for epoch in 0..epochs {
        let epoch_start = std::time::Instant::now();

        // One real full-batch optimisation step per epoch.
        let step_losses = real_training::train_regression(&model, &data, lr, 1)?;
        let train_loss = step_losses.last().copied().unwrap_or(initial_train_loss);

        metrics.train_losses.push(train_loss);
        metrics.learning_rates.push(args.learning_rate);
        metrics
            .epoch_times
            .push(epoch_start.elapsed().as_secs_f64());

        pb.set_position(epoch as u64 + 1);

        if (epoch + 1) % args.save_every == 0 {
            let checkpoint_path = run_dir.join(format!("checkpoint_epoch_{}.json", epoch + 1));
            save_checkpoint(&model, epoch, train_loss, &run_id, &checkpoint_path).await?;
        }

        let metrics_path = run_dir.join("training_metrics.json");
        save_training_metrics(&metrics, &metrics_path).await?;

        output::print_info(&format!(
            "Epoch {}/{} - Train Loss (MSE): {:.6}",
            epoch + 1,
            epochs,
            train_loss
        ));
    }
    pb.finish_with_message("Training completed");

    let final_train_loss = metrics
        .train_losses
        .last()
        .copied()
        .unwrap_or(initial_train_loss);
    let converged = final_train_loss < initial_train_loss;

    // Save a final checkpoint with real parameters.
    let final_ckpt = run_dir.join("final_model.json");
    save_checkpoint(
        &model,
        epochs.saturating_sub(1),
        final_train_loss,
        &run_id,
        &final_ckpt,
    )
    .await?;

    Ok(TrainingResults {
        run_id,
        epochs_completed: metrics.train_losses.len(),
        initial_train_loss,
        final_train_loss,
        synthetic_data: true,
        converged,
    })
}

async fn resume_training(args: ResumeArgs) -> Result<()> {
    validation::validate_file_exists(&args.checkpoint)?;
    // Real resume requires reconstructing a live autograd model + optimiser state
    // from disk, which is not yet wired. Return an honest error instead of
    // fabricating a resumed run.
    Err(anyhow::anyhow!(
        "resuming training from a checkpoint is not yet implemented in the CLI; \
         the checkpoint at {} was validated but cannot be resumed. Use `torsh train start` \
         or the library API.",
        args.checkpoint.display()
    ))
}

async fn monitor_training(args: MonitorArgs) -> Result<()> {
    validation::validate_directory_exists(&args.run)?;

    info!(
        "Monitoring training progress for run: {}",
        args.run.display()
    );

    let metrics_file = args.run.join("training_metrics.json");
    let log_file = args.run.join("training.log");

    if metrics_file.exists() {
        let metrics = load_training_metrics(&metrics_file).await?;
        display_training_metrics(&metrics);
    } else {
        output::print_warning("No metrics file found in the specified run directory");
    }

    if args.follow && log_file.exists() {
        output::print_info("Real-time log following is not implemented; showing recent entries");
        display_recent_logs(&log_file).await?;
    } else if log_file.exists() {
        output::print_info("Recent training log entries:");
        display_recent_logs(&log_file).await?;
    } else {
        output::print_warning("No log file found in the specified run directory");
    }

    Ok(())
}

async fn stop_training(args: StopArgs) -> Result<()> {
    info!("Attempting to stop training run: {}", args.run);
    // The CLI does not manage a background training daemon, so there is no live
    // process to signal. Be honest rather than pretending a stop succeeded.
    let _ = args.force;
    output::print_warning(&format!(
        "No background training process tracking is available; nothing to stop for run '{}'. \
         CLI training runs are synchronous and stop when the command exits.",
        args.run
    ));
    Ok(())
}

/// Load model / dataset hyper-parameters from a JSON config file.
async fn load_training_config(config_path: &PathBuf) -> Result<TrainingConfig> {
    info!(
        "Loading training configuration from {}",
        config_path.display()
    );

    let config_content = tokio::fs::read_to_string(config_path).await?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    let model = &config["model"];
    Ok(TrainingConfig {
        model_name: model["name"].as_str().unwrap_or("mlp").to_string(),
        input_dim: model["input_dim"].as_u64().unwrap_or(16) as usize,
        hidden_dim: model["hidden_dim"].as_u64().unwrap_or(32) as usize,
        output_dim: model["output_dim"].as_u64().unwrap_or(1) as usize,
        num_samples: config["data"]["num_samples"].as_u64().unwrap_or(256) as usize,
    })
}

/// Generate a unique run ID.
fn generate_run_id() -> String {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let suffix: String = (0..6)
        .map(|_| char::from(b'a' + (fastrand::u8(0..26))))
        .collect();
    format!("run_{}_{}", timestamp, suffix)
}

/// A checkpoint containing real serialized model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelCheckpoint {
    run_id: String,
    epoch: usize,
    train_loss: f64,
    /// Each entry is one parameter tensor: (name, shape, flattened f32 values).
    parameters: Vec<(String, Vec<usize>, Vec<f32>)>,
    timestamp: String,
}

/// Save a checkpoint with the model's real parameter values.
async fn save_checkpoint(
    model: &torsh::nn::container::Sequential,
    epoch: usize,
    train_loss: f64,
    run_id: &str,
    checkpoint_path: &PathBuf,
) -> Result<()> {
    use torsh::nn::Module;

    info!("Saving checkpoint to {}", checkpoint_path.display());

    let mut parameters = Vec::new();
    for (name, param) in model.parameters() {
        let tensor = param.tensor();
        let guard = tensor.read();
        let shape = guard.shape().dims().to_vec();
        let values = guard.to_vec()?;
        parameters.push((name, shape, values));
    }

    let checkpoint = ModelCheckpoint {
        run_id: run_id.to_string(),
        epoch,
        train_loss,
        parameters,
        timestamp: chrono::Local::now().to_rfc3339(),
    };

    let data = serde_json::to_vec_pretty(&checkpoint)?;
    tokio::fs::write(checkpoint_path, data).await?;

    Ok(())
}

/// Save training metrics.
async fn save_training_metrics(metrics: &TrainingMetrics, metrics_path: &PathBuf) -> Result<()> {
    let metrics_data = serde_json::to_vec_pretty(metrics)?;
    tokio::fs::write(metrics_path, metrics_data).await?;
    Ok(())
}

/// Load training metrics from file.
async fn load_training_metrics(metrics_path: &PathBuf) -> Result<TrainingMetrics> {
    let metrics_data = tokio::fs::read(metrics_path).await?;
    let metrics: TrainingMetrics = serde_json::from_slice(&metrics_data)?;
    Ok(metrics)
}

/// Display training metrics.
fn display_training_metrics(metrics: &TrainingMetrics) {
    output::print_info(&format!("Run ID: {}", metrics.run_id));
    output::print_info(&format!("Epochs completed: {}", metrics.train_losses.len()));
    if metrics.synthetic_data {
        output::print_warning("This run used the explicitly-synthetic regression dataset");
    }

    if let (Some(&first), Some(&last)) = (metrics.train_losses.first(), metrics.train_losses.last())
    {
        output::print_info(&format!("Initial training loss (MSE): {:.6}", first));
        output::print_info(&format!("Final training loss (MSE): {:.6}", last));
    }
}

/// Display recent log entries.
async fn display_recent_logs(log_path: &PathBuf) -> Result<()> {
    let log_content = tokio::fs::read_to_string(log_path).await?;
    let lines: Vec<&str> = log_content.lines().collect();
    let recent_lines = lines.iter().rev().take(20).rev();

    for line in recent_lines {
        println!("{}", line);
    }

    Ok(())
}
