//! Advanced parallel download functionality for ToRSh Hub
//!
//! This module provides sophisticated parallel download capabilities including:
//! - Multiple file downloads with coordination and load balancing
//! - Streaming downloads for memory-efficient processing of very large files
//! - CDN management with automatic failover and health checking
//! - Mirror management with intelligent selection algorithms
//! - Specialized downloads like GitHub repositories with archive extraction
//!
//! These features build upon the core download functionality to provide
//! enterprise-grade download capabilities with high performance and reliability.

use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs::File as AsyncFile;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use torsh_core::error::{Result, TorshError};

// Import from our other modules
use super::config::{CdnConfig, CdnEndpoint, FailoverStrategy, ParallelDownloadConfig};
use super::core::{download_file_parallel, print_progress};
use super::validation::validate_url;

/// Download multiple files in parallel with coordination
///
/// This function provides sophisticated multi-file downloading with load balancing,
/// progress tracking, and error handling. It's designed for scenarios where you need
/// to download many files efficiently while managing system resources.
///
/// # Arguments
/// * `downloads` - Vector of (url, dest_path) pairs to download
/// * `config` - Configuration for parallel download behavior
/// * `progress` - Whether to display progress information
///
/// # Returns
/// * `Ok(Vec<Result<()>>)` - Results for each download (preserves order)
/// * `Err(TorshError)` - If there's a configuration or setup error
///
/// # Examples
/// ```rust,no_run
/// use torsh_hub::download::parallel::download_files_parallel;
/// use torsh_hub::download::config::ParallelDownloadConfig;
/// use std::path::PathBuf;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let downloads = vec![
///     ("https://example.com/file1.txt".to_string(), std::env::temp_dir().join("file1.txt")),
///     ("https://example.com/file2.txt".to_string(), std::env::temp_dir().join("file2.txt")),
/// ];
/// let config = ParallelDownloadConfig::default();
/// let results = download_files_parallel(downloads, config, true).await?;
/// # Ok(())
/// # }
/// ```
pub async fn download_files_parallel(
    downloads: Vec<(String, PathBuf)>, // (url, dest_path) pairs
    config: ParallelDownloadConfig,
    progress: bool,
) -> Result<Vec<Result<()>>> {
    if downloads.is_empty() {
        return Ok(vec![]);
    }

    if progress {
        println!("Starting parallel download of {} files", downloads.len());
    }

    // Create semaphore to limit concurrent downloads
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_downloads));

    let download_tasks = downloads
        .into_iter()
        .enumerate()
        .map(|(i, (url, dest_path))| {
            let config = config.clone();
            let semaphore = semaphore.clone();

            async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .expect("semaphore acquire should succeed");

                if progress {
                    println!(
                        "[{}/{}] Starting download: {}",
                        i + 1,
                        semaphore.available_permits() + 1,
                        url
                    );
                }

                // No per-file checksum is available in this batch API's
                // (url, dest_path) tuple shape, so integrity verification is
                // not requested here. See `crate::model_info::ModelInfo::download_files`
                // for a checksum-verified download path.
                let result = download_file_parallel(&url, &dest_path, config, false, None).await;

                if progress {
                    match &result {
                        Ok(()) => println!(
                            "[{}/{}] Completed: {}",
                            i + 1,
                            semaphore.available_permits() + 1,
                            url
                        ),
                        Err(e) => println!(
                            "[{}/{}] Failed: {} - {:?}",
                            i + 1,
                            semaphore.available_permits() + 1,
                            url,
                            e
                        ),
                    }
                }

                result
            }
        });

    // Execute all downloads with limited concurrency
    let results: Vec<Result<()>> = stream::iter(download_tasks)
        .buffer_unordered(config.max_concurrent_downloads)
        .collect()
        .await;

    if progress {
        let successful = results.iter().filter(|r| r.is_ok()).count();
        let failed = results.len() - successful;
        println!(
            "Parallel download completed: {} successful, {} failed",
            successful, failed
        );
    }

    Ok(results)
}

/// Streaming download for very large files that don't fit in memory
///
/// This function provides memory-efficient downloading for extremely large files
/// by processing data in chunks with optional transformation. It's ideal for
/// scenarios where you need to download and process files larger than available RAM.
///
/// # Arguments
/// * `url` - The URL to download from
/// * `dest_path` - The destination file path
/// * `config` - Download configuration
/// * `progress` - Whether to display progress
/// * `chunk_processor` - Optional function to process each chunk (e.g., compression)
///
/// # Returns
/// * `Ok(())` if the streaming download succeeded
/// * `Err(TorshError)` if the download failed
///
/// # Examples
/// ```rust,no_run
/// use torsh_hub::download::parallel::download_file_streaming;
/// use torsh_hub::download::config::ParallelDownloadConfig;
/// use std::path::Path;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ParallelDownloadConfig::default();
/// let result = download_file_streaming(
///     "https://example.com/huge_file.bin",
///     &std::env::temp_dir().join("huge_file.bin"),
///     config,
///     true,
///     None // No chunk processing
/// ).await?;
/// # Ok(())
/// # }
/// ```
///
/// Type alias for chunk processing function
type ChunkProcessor = Box<dyn Fn(&[u8]) -> Result<Vec<u8>> + Send + Sync>;

pub async fn download_file_streaming(
    url: &str,
    dest_path: &Path,
    config: ParallelDownloadConfig,
    progress: bool,
    chunk_processor: Option<ChunkProcessor>,
) -> Result<()> {
    // Validate URL before attempting download
    validate_url(url)?;

    // Create parent directory if needed
    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Create HTTP client
    let client = crate::tls::client_builder()?
        .user_agent("torsh-hub/0.1.0-alpha.2")
        .timeout(Duration::from_secs(config.timeout_seconds))
        .build()
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    // Start download
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    if !response.status().is_success() {
        return Err(TorshError::IoError(format!(
            "Failed to download {}: HTTP {}",
            url,
            response.status()
        )));
    }

    let total_size = response.content_length();

    if progress {
        if let Some(total) = total_size {
            println!(
                "Streaming download {} ({:.1} MB)",
                url,
                total as f64 / 1_048_576.0
            );
        } else {
            println!("Streaming download {}", url);
        }
    }

    // Create file for streaming write
    let temp_path = dest_path.with_extension("tmp");
    let mut file = AsyncFile::create(&temp_path).await?;

    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| TorshError::IoError(e.to_string()))?;

        // Apply chunk processor if provided (e.g., for compression/decompression)
        let processed_chunk = if let Some(ref processor) = chunk_processor {
            processor(&chunk)?
        } else {
            chunk.to_vec()
        };

        file.write_all(&processed_chunk).await?;
        downloaded += chunk.len() as u64;

        if progress {
            if let Some(total) = total_size {
                print_progress(downloaded, total);
            }
        }

        // Periodically sync to disk for very large files
        if downloaded % (10 * 1024 * 1024) == 0 {
            // Every 10MB
            file.sync_data().await?;
        }
    }

    file.sync_all().await?;
    drop(file);

    if progress {
        println!(); // New line after progress
    }

    // Move temporary file to final destination
    tokio::fs::rename(&temp_path, dest_path).await?;

    Ok(())
}

/// CDN manager for handling multiple endpoints with automatic failover
///
/// This structure provides sophisticated CDN management with health checking,
/// performance monitoring, and intelligent failover strategies. It's designed
/// for production environments where download reliability is critical.
pub struct CdnManager {
    config: CdnConfig,
    client: Client,
}

impl CdnManager {
    /// Create a new CDN manager
    ///
    /// # Arguments
    /// * `config` - CDN configuration with endpoints and policies
    ///
    /// # Returns
    /// * `Ok(CdnManager)` - Successfully created manager
    /// * `Err(TorshError)` - If HTTP client creation fails
    pub fn new(config: CdnConfig) -> Result<Self> {
        let client = crate::tls::client_builder()?
            .user_agent("torsh-hub/0.1.0-alpha.2")
            .timeout(config.endpoint_timeout)
            .build()
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(Self { config, client })
    }

    /// Get available CDN endpoints sorted by strategy
    ///
    /// This method filters healthy endpoints and sorts them according to the
    /// configured failover strategy for optimal download performance.
    pub fn get_available_endpoints(&self) -> Vec<&CdnEndpoint> {
        let mut endpoints: Vec<&CdnEndpoint> =
            self.config.endpoints.iter().filter(|e| e.healthy).collect();

        match self.config.failover_strategy {
            FailoverStrategy::Priority => {
                endpoints.sort_by_key(|e| e.priority);
            }
            FailoverStrategy::Fastest => {
                endpoints.sort_by_key(|e| e.avg_response_time.unwrap_or(u64::MAX));
            }
            FailoverStrategy::RoundRobin => {
                // For simplicity, use priority order. Real implementation would maintain state
                endpoints.sort_by_key(|e| e.priority);
            }
            FailoverStrategy::Random => {
                // For simplicity, use priority order. Real implementation would shuffle
                endpoints.sort_by_key(|e| e.priority);
            }
        }

        endpoints
    }

    /// Download a file using CDN with automatic failover
    ///
    /// This method attempts to download from multiple CDN endpoints with
    /// intelligent failover and performance tracking.
    ///
    /// # Arguments
    /// * `file_path` - Relative path to the file on CDN servers
    /// * `dest_path` - Local destination path
    /// * `progress` - Whether to show progress information
    pub async fn download_with_cdn(
        &self,
        file_path: &str,
        dest_path: &Path,
        progress: bool,
    ) -> Result<()> {
        let endpoints = self.get_available_endpoints();

        if endpoints.is_empty() {
            return Err(TorshError::IoError(
                "No healthy CDN endpoints available".to_string(),
            ));
        }

        let mut last_error = None;
        let mut attempts = 0;

        for endpoint in endpoints.iter().take(self.config.max_retries) {
            attempts += 1;
            let url = format!(
                "{}/{}",
                endpoint.base_url.trim_end_matches('/'),
                file_path.trim_start_matches('/')
            );

            if progress {
                println!(
                    "Attempting download from CDN: {} (attempt {}/{})",
                    endpoint.name, attempts, self.config.max_retries
                );
            }

            let start_time = Instant::now();

            match self
                .download_from_endpoint(&url, dest_path, endpoint, progress)
                .await
            {
                Ok(()) => {
                    let elapsed = start_time.elapsed();
                    if progress {
                        println!(
                            "Successfully downloaded from CDN: {} ({:.2}s)",
                            endpoint.name,
                            elapsed.as_secs_f64()
                        );
                    }
                    return Ok(());
                }
                Err(e) => {
                    let elapsed = start_time.elapsed();
                    if progress {
                        println!(
                            "Failed to download from CDN: {} ({:.2}s) - {:?}",
                            endpoint.name,
                            elapsed.as_secs_f64(),
                            e
                        );
                    }
                    last_error = Some(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| TorshError::IoError("All CDN endpoints failed".to_string())))
    }

    /// Download from a specific endpoint
    async fn download_from_endpoint(
        &self,
        url: &str,
        dest_path: &Path,
        endpoint: &CdnEndpoint,
        progress: bool,
    ) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = dest_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Build request with custom headers
        let mut request = self.client.get(url);
        for (key, value) in &endpoint.headers {
            request = request.header(key, value);
        }

        // Start download
        let response = request
            .send()
            .await
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(TorshError::IoError(format!(
                "Failed to download from {}: HTTP {}",
                endpoint.name,
                response.status()
            )));
        }

        let total_size = response.content_length();

        if progress {
            if let Some(total) = total_size {
                println!(
                    "Downloading from {} ({:.1} MB)",
                    endpoint.name,
                    total as f64 / 1_048_576.0
                );
            } else {
                println!("Downloading from {}", endpoint.name);
            }
        }

        // Create temporary file
        let temp_path = dest_path.with_extension("tmp");
        let mut file = AsyncFile::create(&temp_path).await?;

        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| TorshError::IoError(e.to_string()))?;

            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if progress {
                if let Some(total) = total_size {
                    print_progress(downloaded, total);
                }
            }
        }

        file.sync_all().await?;
        drop(file);

        if progress {
            println!(); // New line after progress
        }

        // Move temporary file to final destination
        tokio::fs::rename(&temp_path, dest_path).await?;

        Ok(())
    }

    /// Get CDN statistics
    ///
    /// Returns statistical information about CDN endpoint health and performance.
    pub fn get_statistics(&self) -> CdnStatistics {
        let total_endpoints = self.config.endpoints.len();
        let healthy_endpoints = self.config.endpoints.iter().filter(|e| e.healthy).count();
        let avg_response_time = {
            let times: Vec<u64> = self
                .config
                .endpoints
                .iter()
                .filter_map(|e| e.avg_response_time)
                .collect();
            if times.is_empty() {
                None
            } else {
                Some(times.iter().sum::<u64>() / times.len() as u64)
            }
        };

        CdnStatistics {
            total_endpoints,
            healthy_endpoints,
            avg_response_time,
            failover_strategy: self.config.failover_strategy.clone(),
        }
    }
}

/// CDN usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnStatistics {
    pub total_endpoints: usize,
    pub healthy_endpoints: usize,
    pub avg_response_time: Option<u64>,
    pub failover_strategy: FailoverStrategy,
}

/// Download a GitHub repository as an archive
///
/// This function downloads a GitHub repository branch as a tar.gz archive,
/// extracts it, and organizes the files in the specified destination directory.
/// It's useful for downloading model repositories, examples, or documentation.
///
/// # Arguments
/// * `owner` - GitHub repository owner/organization
/// * `repo` - Repository name
/// * `branch` - Branch to download (e.g., "main", "master", "v1.0")
/// * `dest_dir` - Destination directory for extracted files
/// * `verbose` - Whether to show detailed progress information
///
/// # Returns
/// * `Ok(())` if the repository was successfully downloaded and extracted
/// * `Err(TorshError)` if the download or extraction failed
///
/// # Examples
/// ```rust,no_run
/// use torsh_hub::download::parallel::download_github_repo;
/// use std::path::Path;
///
/// let result = download_github_repo(
///     "torsh-ai",
///     "examples",
///     "main",
///     &std::env::temp_dir().join("torsh-examples"),
///     true
/// );
/// ```
pub fn download_github_repo(
    owner: &str,
    repo: &str,
    branch: &str,
    dest_dir: &Path,
    verbose: bool,
) -> Result<()> {
    // Use GitHub archive API
    let url = format!(
        "https://github.com/{}/{}/archive/refs/heads/{}.tar.gz",
        owner, repo, branch
    );

    let archive_path = dest_dir.with_extension("tar.gz");

    if verbose {
        println!("Downloading repository {}/{}@{}", owner, repo, branch);
    }

    // Download archive. GitHub does not publish a stable content hash for
    // branch/tag tarballs (they are regenerated on demand), so there is no
    // checksum to verify against here.
    super::core::download_file(&url, &archive_path, verbose, None)?;

    // Extract archive
    extract_tarball(&archive_path, dest_dir)?;

    // Clean up archive
    fs::remove_file(&archive_path)?;

    // The extracted directory will be named repo-branch
    // Rename it to the expected dest_dir
    let extracted_name = format!("{}-{}", repo, branch);
    let extracted_path = dest_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&extracted_name);

    if extracted_path.exists() && extracted_path != dest_dir {
        if dest_dir.exists() {
            fs::remove_dir_all(dest_dir)?;
        }
        fs::rename(&extracted_path, dest_dir)?;
    }

    if verbose {
        println!("Repository downloaded to {:?}", dest_dir);
    }

    Ok(())
}

/// Extract a tar.gz archive
///
/// This internal function handles the extraction of tar.gz archives using
/// the oxiarc-deflate and oxiarc-archive crates for decompression and archive extraction.
fn extract_tarball(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    use oxiarc_archive::TarStreamReader;
    use oxiarc_core::EntryType;
    use oxiarc_deflate::GzipStreamDecoder;
    use std::io;

    let file = std::fs::File::open(archive_path)?;
    let decoder = GzipStreamDecoder::new(file);
    let mut stream = TarStreamReader::new(decoder);

    let base_dir = dest_dir.parent().unwrap_or_else(|| Path::new("."));
    if let Some(parent) = dest_dir.parent() {
        fs::create_dir_all(parent)?;
    }

    while let Some(mut entry) = stream
        .next_entry()
        .map_err(|e| TorshError::IoError(format!("Failed to read tar entry: {}", e)))?
    {
        // Reject entries that would escape `base_dir` ("tar-slip"): an
        // absolute name or one containing a `..` component. See
        // `crate::utils::sanitize_archive_entry_path` for the exact rules.
        let dest = crate::utils::sanitize_archive_entry_path(base_dir, &entry.header.name)?;
        match entry.header.entry_type() {
            EntryType::Directory => {
                fs::create_dir_all(&dest)
                    .map_err(|e| TorshError::IoError(format!("Failed to create dir: {}", e)))?;
            }
            EntryType::File => {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent).map_err(|e| {
                        TorshError::IoError(format!("Failed to create parent dir: {}", e))
                    })?;
                }
                let mut out = std::fs::File::create(&dest)
                    .map_err(|e| TorshError::IoError(format!("Failed to create file: {}", e)))?;
                io::copy(&mut entry, &mut out)
                    .map_err(|e| TorshError::IoError(format!("Failed to write file: {}", e)))?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Convenience function to download with default CDN configuration
///
/// This function provides a simple interface for CDN downloads using
/// sensible default settings. It's ideal for quick integration or testing.
///
/// # Arguments
/// * `file_path` - Relative path to the file on CDN servers
/// * `dest_path` - Local destination path
/// * `progress` - Whether to show progress information
pub async fn download_with_default_cdn(
    file_path: &str,
    dest_path: &Path,
    progress: bool,
) -> Result<()> {
    let config = CdnConfig::default();
    let manager = CdnManager::new(config)?;
    manager
        .download_with_cdn(file_path, dest_path, progress)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_parallel_downloads_empty() {
        let downloads = vec![];
        let config = ParallelDownloadConfig::default();
        let results = download_files_parallel(downloads, config, false)
            .await
            .expect("operation should succeed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_download_url_validation() {
        let temp_dir = TempDir::new().expect("Temp Dir should succeed");
        let dest_path = temp_dir.path().join("test.txt");
        let config = ParallelDownloadConfig::default();

        // Test invalid URL
        let result = download_file_streaming("invalid-url", &dest_path, config, false, None).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_cdn_manager_creation() {
        let config = CdnConfig::default();
        let manager = CdnManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_cdn_endpoint_sorting() {
        let config = CdnConfig::default();
        let manager = CdnManager::new(config).expect("Cdn Manager should succeed");

        let endpoints = manager.get_available_endpoints();
        assert_eq!(endpoints.len(), 2);

        // Should be sorted by priority (lower numbers first)
        assert!(endpoints[0].priority <= endpoints[1].priority);
    }

    #[test]
    fn test_cdn_statistics() {
        let config = CdnConfig::default();
        let manager = CdnManager::new(config).expect("Cdn Manager should succeed");

        let stats = manager.get_statistics();
        assert_eq!(stats.total_endpoints, 2);
        assert_eq!(stats.healthy_endpoints, 2);
        assert_eq!(stats.failover_strategy, FailoverStrategy::Priority);
    }

    #[test]
    fn test_github_repo_url_format() {
        let owner = "torsh-ai";
        let repo = "examples";
        let branch = "main";

        let expected_url = format!(
            "https://github.com/{}/{}/archive/refs/heads/{}.tar.gz",
            owner, repo, branch
        );

        assert_eq!(
            expected_url,
            "https://github.com/torsh-ai/examples/archive/refs/heads/main.tar.gz"
        );
    }

    #[test]
    fn test_extracted_directory_name() {
        let repo = "examples";
        let branch = "main";
        let extracted_name = format!("{}-{}", repo, branch);

        assert_eq!(extracted_name, "examples-main");
    }

    #[tokio::test]
    async fn test_default_cdn_manager() {
        let test_dest = std::env::temp_dir().join("test.txt");
        let result = download_with_default_cdn("test/file.txt", &test_dest, false).await;

        // This should fail since we're using dummy URLs, but it tests the setup
        assert!(result.is_err());
    }

    #[test]
    fn test_temp_file_path_generation() {
        let dest_path = std::env::temp_dir().join("test.txt");
        let temp_path = dest_path.with_extension("tmp");
        assert_eq!(temp_path, std::env::temp_dir().join("test.tmp"));
    }

    /// Build a `.tar.gz` byte stream containing a single entry named
    /// `entry_name` with the given `data`, using the same TarWriter +
    /// gzip_compress machinery `download_github_repo` downloads in
    /// production (just constructed locally instead of over HTTP).
    fn build_malicious_tar_gz(entry_name: &str, data: &[u8]) -> Vec<u8> {
        use oxiarc_archive::TarWriter;

        let mut tar_bytes = Vec::new();
        {
            let mut writer = TarWriter::new(&mut tar_bytes);
            writer
                .add_file(entry_name, data)
                .expect("crafting a malicious tar entry must itself succeed (writer does not sanitise names)");
            writer.finish().expect("tar finish should succeed");
        }
        oxiarc_deflate::gzip_compress(&tar_bytes, 6).expect("gzip compression should succeed")
    }

    /// F022 (tar-slip site 1): a tar entry named with a `..`-escaping
    /// relative path must not be able to write outside the extraction
    /// directory via `extract_tarball`.
    #[test]
    fn f022_extract_tarball_rejects_parent_dir_traversal() {
        let temp_dir = TempDir::new().expect("temp dir");
        let archive_path = temp_dir.path().join("evil.tar.gz");
        let dest_dir = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_dir).expect("create dest dir");

        // Escapes dest_dir *and* its parent (temp_dir) to land right next to
        // the destination, so the outcome is trivial to check for.
        let payload = b"pwned-by-tar-slip";
        let archive_bytes = build_malicious_tar_gz("../../evil_outside.txt", payload);
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        let result = extract_tarball(&archive_path, &dest_dir);
        assert!(
            result.is_err(),
            "extract_tarball must reject a '..'-escaping entry name"
        );

        // The escape target must never have been created anywhere near the
        // destination tree.
        assert!(!temp_dir.path().join("evil_outside.txt").exists());
        assert!(!dest_dir.join("../evil_outside.txt").exists());
    }

    /// F022 (tar-slip site 1): an absolute tar entry path must not be
    /// extracted verbatim (which would discard `dest_dir` entirely via
    /// `Path::join`'s absolute-path semantics). Uses a short, fixed
    /// absolute name (rather than one derived from the temp dir) so the
    /// entry name always fits in a plain UStar header regardless of how
    /// long the sandbox's temp path happens to be.
    #[test]
    fn f022_extract_tarball_rejects_absolute_path() {
        let temp_dir = TempDir::new().expect("temp dir");
        let archive_path = temp_dir.path().join("evil_abs.tar.gz");
        let dest_dir = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_dir).expect("create dest dir");

        let absolute_name = "/torsh_hardening_test_absolute_evil.txt";
        let archive_bytes = build_malicious_tar_gz(absolute_name, b"pwned-by-absolute-path");
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        let result = extract_tarball(&archive_path, &dest_dir);
        assert!(
            result.is_err(),
            "extract_tarball must reject an absolute entry path"
        );
        // The rejection must happen before any write is attempted, so the
        // absolute path must never have been created.
        assert!(!Path::new(absolute_name).exists());
    }

    /// Control case: a well-behaved relative entry must still extract
    /// normally after the sanitisation fix. `extract_tarball` extracts
    /// relative to `dest_dir.parent()` (matching GitHub's tarball layout,
    /// which embeds a top-level `repo-branch/` folder that
    /// `download_github_repo` renames to `dest_dir` afterward), so the
    /// benign entry lands under `temp_dir.path()`, not `dest_dir` itself.
    #[test]
    fn f022_extract_tarball_accepts_benign_relative_path() {
        let temp_dir = TempDir::new().expect("temp dir");
        let archive_path = temp_dir.path().join("benign.tar.gz");
        let dest_dir = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_dir).expect("create dest dir");

        let archive_bytes = build_malicious_tar_gz("sub/model.bin", b"legit-content");
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        extract_tarball(&archive_path, &dest_dir).expect("benign entry should extract cleanly");

        let extracted = temp_dir.path().join("sub").join("model.bin");
        assert!(extracted.exists());
        assert_eq!(
            std::fs::read(&extracted).expect("read extracted file"),
            b"legit-content"
        );
    }

    /// F022 (tar-slip, symlink case): a `Symlink` tar entry must never be
    /// materialised on disk, so a later entry naming a path "through" it
    /// (as if the symlink existed and pointed outside the destination
    /// tree) cannot use it to escape `base_dir`. `extract_tarball` only
    /// creates filesystem entries for `EntryType::Directory` and
    /// `EntryType::File` (every other entry type, including `Symlink` and
    /// `Hardlink`, falls into its `_ => {}` no-op arm), so this test
    /// proves that property end-to-end rather than merely asserting it —
    /// a lexically-clean entry *name* (which `sanitize_archive_entry_path`
    /// alone would accept) combined with a malicious symlink *target* is
    /// exactly the case a path-only guard would miss.
    #[test]
    fn f022_extract_tarball_never_materialises_symlink_entries() {
        use oxiarc_archive::TarWriter;

        let temp_dir = TempDir::new().expect("temp dir");
        let archive_path = temp_dir.path().join("symlink_evil.tar.gz");
        let dest_dir = temp_dir.path().join("extract_here");
        std::fs::create_dir_all(&dest_dir).expect("create dest dir");

        let mut tar_bytes = Vec::new();
        {
            let mut writer = TarWriter::new(&mut tar_bytes);
            // A symlink entry whose *name* is perfectly clean but whose
            // *target* escapes the extraction root entirely.
            writer
                .add_symlink("escape_link", "../../../../../../../../tmp")
                .expect("writing a symlink entry must itself succeed");
            // A follow-up entry that would write "through" the symlink
            // above, were it ever materialised on disk.
            writer
                .add_file("escape_link/pwned.txt", b"pwned-via-symlink")
                .expect("writing the follow-up file entry must itself succeed");
            writer.finish().expect("tar finish should succeed");
        }
        let archive_bytes =
            oxiarc_deflate::gzip_compress(&tar_bytes, 6).expect("gzip compression should succeed");
        std::fs::write(&archive_path, &archive_bytes).expect("write archive");

        extract_tarball(&archive_path, &dest_dir).expect(
            "a Symlink entry must be silently skipped (not rejected), so extraction of the \
             remaining benign entries still succeeds",
        );

        let escape_link_path = temp_dir.path().join("escape_link");
        // `symlink_metadata` does not follow links, so this would still
        // observe a real filesystem symlink had one been created.
        if let Ok(meta) = std::fs::symlink_metadata(&escape_link_path) {
            assert!(
                !meta.file_type().is_symlink(),
                "a tar Symlink entry must never be materialised as a real filesystem symlink"
            );
        }

        // The follow-up "escape_link/pwned.txt" entry must land as an
        // ordinary nested file under `base_dir`, never through wherever
        // the (never created) symlink would have pointed.
        let nested_file = escape_link_path.join("pwned.txt");
        assert!(nested_file.exists());
        assert_eq!(
            std::fs::read(&nested_file).expect("read nested file"),
            b"pwned-via-symlink"
        );
        assert!(
            nested_file.starts_with(temp_dir.path()),
            "the nested file must stay under the extraction root, not escape through a symlink"
        );
    }
}
