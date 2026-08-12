//! Security-hardening regression tests for `torsh-hub`.
//!
//! Each test is tied to a specific finding from the production-hardening
//! security audit:
//!
//! - **F022** (tar-slip, 2 sites): `sanitize_archive_entry_path` here, plus
//!   `download::parallel::extract_tarball` and `TfLoader::extract_archive`
//!   (both tested inline in `src/download/parallel.rs` and
//!   `src/tensorflow.rs` respectively, since both are private functions),
//!   including a symlink-entry case at both sites.
//! - **F023** (downloads never integrity-checked): `download_file`,
//!   `download_with_retry`, `download_file_parallel`, and the new
//!   `ModelInfo::download_files` registry-checksum-verified download path.
//! - **F120** (orphaned `src/security/` module tree): proven by this crate
//!   compiling at all post-deletion; see the FINAL REPORT for details, no
//!   dedicated runtime test is meaningful for a deletion.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::thread;
use std::time::Duration;

use tempfile::TempDir;
use torsh_core::error::TorshError;
use torsh_hub::download::config::ParallelDownloadConfig;
use torsh_hub::download::core::{download_file, download_file_parallel, download_with_retry};
use torsh_hub::download::validation::{calculate_file_hash, HashAlgorithm};
use torsh_hub::model_info::{FileInfo, ModelInfo, Version};
use torsh_hub::sanitize_archive_entry_path;

/// Spawn a minimal single-purpose HTTP/1.1 server on loopback that answers
/// up to `max_requests` connections with the same canned `body`
/// (`Content-Length`-framed, `Connection: close`, no `Accept-Ranges` so
/// callers always take the "simple" download path), then stops accepting.
///
/// This lets the SHA-256 integrity-verification wiring (F023) be exercised
/// end-to-end against the crate's real, public download functions without
/// depending on outside network access or a real file host.
fn spawn_canned_http_server(body: &'static [u8], max_requests: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral loopback port");
    let addr = listener.local_addr().expect("read local_addr");

    thread::spawn(move || {
        for stream in listener.incoming().take(max_requests) {
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);

            let response_head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(response_head.as_bytes());
            let _ = stream.write_all(body);
            let _ = stream.flush();
        }
    });

    format!("http://{}", addr)
}

/// Compute the hex SHA-256 of `bytes` via the crate's own hashing utility
/// (round-tripped through a temp file, since `calculate_file_hash` hashes a
/// file on disk rather than an in-memory buffer).
fn sha256_hex_of(bytes: &[u8]) -> String {
    let dir = TempDir::new().expect("tempdir for hashing");
    let path = dir.path().join("hash_input.bin");
    std::fs::write(&path, bytes).expect("write hash input");
    calculate_file_hash(&path, HashAlgorithm::Sha256).expect("calculate_file_hash should succeed")
}

// ---------------------------------------------------------------------
// F022: tar-slip / archive path sanitisation
// ---------------------------------------------------------------------

#[test]
fn f022_sanitize_archive_entry_path_rejects_parent_dir_traversal() {
    let root = Path::new("/safe/extract/root");
    let err = sanitize_archive_entry_path(root, "../../../../etc/cron.d/pwn")
        .expect_err("must reject a '..'-escaping entry name");
    assert!(matches!(err, TorshError::InvalidArgument(_)));
}

#[test]
fn f022_sanitize_archive_entry_path_rejects_absolute_path() {
    let root = Path::new("/safe/extract/root");
    let err = sanitize_archive_entry_path(root, "/Users/x/.ssh/authorized_keys")
        .expect_err("must reject an absolute entry name");
    assert!(matches!(err, TorshError::InvalidArgument(_)));
}

#[test]
fn f022_sanitize_archive_entry_path_accepts_and_joins_benign_paths() {
    let root = Path::new("/safe/extract/root");
    let dest =
        sanitize_archive_entry_path(root, "sub/dir/net.bin").expect("benign path is accepted");
    assert_eq!(dest, root.join("sub").join("dir").join("net.bin"));
    assert!(dest.starts_with(root));
}

// ---------------------------------------------------------------------
// F023: hub downloads are now integrity-checked
// ---------------------------------------------------------------------

#[test]
fn f023_download_file_rejects_mismatched_hash_and_cleans_up_temp_file() {
    let body: &'static [u8] = b"totally-legit-model-weights";
    let base_url = spawn_canned_http_server(body, 1);

    let temp_dir = TempDir::new().expect("tempdir");
    let dest = temp_dir.path().join("model.bin");
    let wrong_hash = "0".repeat(64);

    let result = download_file(
        &format!("{base_url}/model.bin"),
        &dest,
        false,
        Some(&wrong_hash),
    );

    assert!(result.is_err(), "mismatched hash must fail closed");
    assert!(
        !dest.exists(),
        "destination file must not be created on hash mismatch"
    );
    assert!(
        !dest.with_extension("tmp").exists(),
        "temp file must be cleaned up after a failed integrity check"
    );
}

#[test]
fn f023_download_file_accepts_matching_hash() {
    let body: &'static [u8] = b"totally-legit-model-weights-v2";
    let base_url = spawn_canned_http_server(body, 1);
    let correct_hash = sha256_hex_of(body);

    let temp_dir = TempDir::new().expect("tempdir");
    let dest = temp_dir.path().join("model.bin");

    download_file(
        &format!("{base_url}/model.bin"),
        &dest,
        false,
        Some(&correct_hash),
    )
    .expect("download with a correct hash should succeed");
    assert_eq!(std::fs::read(&dest).expect("read downloaded file"), body);
}

#[test]
fn f023_download_file_accepts_matching_hash_case_insensitively() {
    // `verify_file_integrity` lowercases both sides before comparing, so a
    // registry/caller-supplied checksum in uppercase (or mixed case) hex
    // must be accepted, not fail closed spuriously.
    let body: &'static [u8] = b"totally-legit-model-weights-v3-uppercase-hash";
    let base_url = spawn_canned_http_server(body, 1);
    let uppercase_hash = sha256_hex_of(body).to_uppercase();

    let temp_dir = TempDir::new().expect("tempdir");
    let dest = temp_dir.path().join("model.bin");

    download_file(
        &format!("{base_url}/model.bin"),
        &dest,
        false,
        Some(&uppercase_hash),
    )
    .expect("an uppercase-hex expected hash must still match the (lowercase) computed hash");
    assert_eq!(std::fs::read(&dest).expect("read downloaded file"), body);
}

#[test]
fn f023_download_file_without_hash_still_succeeds_documented_no_verification() {
    // Documents the intentional behavior: `expected_hash: None` means "no
    // checksum available", not "verification failed".
    let body: &'static [u8] = b"unverified-content-no-checksum-available";
    let base_url = spawn_canned_http_server(body, 1);
    let temp_dir = TempDir::new().expect("tempdir");
    let dest = temp_dir.path().join("model.bin");

    download_file(&format!("{base_url}/model.bin"), &dest, false, None)
        .expect("download without a hash should still succeed");
    assert!(dest.exists());
}

#[test]
fn f023_download_with_retry_fails_closed_on_mismatched_hash() {
    let body: &'static [u8] = b"weights-that-do-not-match";
    let base_url = spawn_canned_http_server(body, 1);
    let temp_dir = TempDir::new().expect("tempdir");
    let dest = temp_dir.path().join("model.bin");
    let wrong_hash = "f".repeat(64);

    // max_retries = 0 => exactly one attempt, no exponential-backoff sleep.
    let result = download_with_retry(
        &format!("{base_url}/model.bin"),
        &dest,
        0,
        false,
        Some(&wrong_hash),
    );
    assert!(result.is_err(), "mismatched hash must fail closed");
    assert!(!dest.exists());
}

#[tokio::test]
async fn f023_download_file_parallel_rejects_mismatched_hash() {
    let body: &'static [u8] = b"parallel-path-weights";
    // download_file_parallel issues a HEAD followed by a GET.
    let base_url = spawn_canned_http_server(body, 2);
    let temp_dir = TempDir::new().expect("tempdir");
    let dest = temp_dir.path().join("model.bin");
    let wrong_hash = "1".repeat(64);
    let config = ParallelDownloadConfig::default();

    let result = download_file_parallel(
        &format!("{base_url}/model.bin"),
        &dest,
        config,
        false,
        Some(&wrong_hash),
    )
    .await;
    assert!(result.is_err(), "mismatched hash must fail closed");
    assert!(!dest.exists());
}

#[tokio::test]
async fn f023_download_file_parallel_accepts_matching_hash() {
    let body: &'static [u8] = b"parallel-path-weights-ok";
    let base_url = spawn_canned_http_server(body, 2);
    let correct_hash = sha256_hex_of(body);
    let temp_dir = TempDir::new().expect("tempdir");
    let dest = temp_dir.path().join("model.bin");
    let config = ParallelDownloadConfig::default();

    download_file_parallel(
        &format!("{base_url}/model.bin"),
        &dest,
        config,
        false,
        Some(&correct_hash),
    )
    .await
    .expect("matching hash should succeed");
    assert_eq!(std::fs::read(&dest).expect("read downloaded file"), body);
}

// ---------------------------------------------------------------------
// F023: the real registry-driven consumer, `ModelInfo::download_files`
// ---------------------------------------------------------------------

#[test]
fn f023_model_info_download_files_verifies_registry_checksum() {
    let body: &'static [u8] = b"real-model-weights-bytes";
    let correct_hash = sha256_hex_of(body);
    let base_url = spawn_canned_http_server(body, 1);

    let mut info = ModelInfo::new(
        "demo-model".to_string(),
        "demo-author".to_string(),
        Version::new(1, 0, 0),
    );
    info.files.push(FileInfo {
        path: "weights.bin".to_string(),
        size_bytes: body.len() as u64,
        sha256: correct_hash,
        description: None,
    });

    let temp_dir = TempDir::new().expect("tempdir");
    let downloaded = info
        .download_files(&base_url, temp_dir.path(), false)
        .expect("download_files with a correct registry checksum should succeed");

    assert_eq!(downloaded.len(), 1);
    assert_eq!(std::fs::read(&downloaded[0]).expect("read"), body);
}

#[test]
fn f023_model_info_download_files_fails_closed_on_wrong_checksum() {
    let body: &'static [u8] = b"tampered-or-mitm-substituted-weights";
    let base_url = spawn_canned_http_server(body, 1);

    let mut info = ModelInfo::new(
        "demo-model".to_string(),
        "demo-author".to_string(),
        Version::new(1, 0, 0),
    );
    info.files.push(FileInfo {
        path: "weights.bin".to_string(),
        size_bytes: body.len() as u64,
        sha256: "a".repeat(64), // does not match `body`'s real SHA-256
        description: None,
    });

    let temp_dir = TempDir::new().expect("tempdir");
    let result = info.download_files(&base_url, temp_dir.path(), false);

    assert!(
        result.is_err(),
        "a mismatching registry checksum must fail closed, e.g. a MITM'd/compromised mirror"
    );
    assert!(!temp_dir.path().join("weights.bin").exists());
}

#[test]
fn f023_model_info_download_files_rejects_path_traversal_in_file_info_path() {
    // A compromised/malicious registry entry must not be able to use
    // FileInfo::path to escape dest_dir.
    let mut info = ModelInfo::new(
        "demo-model".to_string(),
        "demo-author".to_string(),
        Version::new(1, 0, 0),
    );
    info.files.push(FileInfo {
        path: "../../etc/passwd".to_string(),
        size_bytes: 0,
        sha256: "b".repeat(64),
        description: None,
    });

    let temp_dir = TempDir::new().expect("tempdir");
    // Deliberately never-listening address: if sanitisation were bypassed,
    // this would surface as a network error instead, which the specific
    // InvalidArgument match below would catch as a test failure.
    let result = info.download_files("http://127.0.0.1:1", temp_dir.path(), false);

    match result {
        Err(TorshError::InvalidArgument(_)) => {}
        other => panic!(
            "expected TorshError::InvalidArgument (path-traversal rejection), got: {:?}",
            other
        ),
    }
}

#[test]
fn f023_model_info_download_files_treats_non_hash_sha256_as_unverified() {
    // A placeholder/malformed `sha256` (not 64 hex chars) must be treated
    // as "no checksum available", not as an expected hash of the empty
    // string (which would make every such file fail closed spuriously).
    let body: &'static [u8] = b"file-with-a-placeholder-checksum";
    let base_url = spawn_canned_http_server(body, 1);

    let mut info = ModelInfo::new(
        "demo-model".to_string(),
        "demo-author".to_string(),
        Version::new(1, 0, 0),
    );
    info.files.push(FileInfo {
        path: "weights.bin".to_string(),
        size_bytes: body.len() as u64,
        sha256: String::new(),
        description: None,
    });

    let temp_dir = TempDir::new().expect("tempdir");
    let downloaded = info
        .download_files(&base_url, temp_dir.path(), false)
        .expect("an empty/placeholder sha256 must not be treated as an expected empty-hash");
    assert_eq!(std::fs::read(&downloaded[0]).expect("read"), body);
}
