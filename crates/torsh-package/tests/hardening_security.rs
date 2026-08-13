//! Security-hardening regression tests for `torsh-package`.
//!
//! Each test is tied to a specific finding from the production-hardening
//! security audit:
//!
//! - **F041** (zip-slip): `sanitize_archive_entry_path` and
//!   `PackageImporter::extract_package`.
//! - **F240** (export/import roundtrip loses resource paths):
//!   `PackageImporter::read_resources` (exercised via
//!   `PackageExporter`/`PackageImporter` round trips).
//! - **F306** (dependency solver panic): tested inline in
//!   `src/dependency_solver.rs` since `choose_decision_variable` is a
//!   private method.

use std::fs::File;
use std::path::Path;

use oxiarc_archive::zip::{ZipReader, ZipWriter};
use tempfile::TempDir;
use torsh_core::error::TorshError;
use torsh_package::{
    sanitize_archive_entry_path, ExportConfig, ImportConfig, Package, PackageExporter,
    PackageImporter, Resource, ResourceType,
};

// ---------------------------------------------------------------------
// F041: zip-slip in PackageImporter::extract_package
// ---------------------------------------------------------------------

#[test]
fn f041_sanitize_archive_entry_path_rejects_parent_dir_traversal() {
    let root = Path::new("/safe/extract/root");
    let err = sanitize_archive_entry_path(root, "../../.bashrc")
        .expect_err("must reject a '..'-escaping entry name");
    assert!(matches!(err, TorshError::InvalidArgument(_)));
}

#[test]
fn f041_sanitize_archive_entry_path_rejects_absolute_path() {
    let root = Path::new("/safe/extract/root");
    let err = sanitize_archive_entry_path(root, "/etc/x")
        .expect_err("must reject an absolute entry name");
    assert!(matches!(err, TorshError::InvalidArgument(_)));
}

/// Craft a `.torshpkg`-shaped ZIP file (bypassing `PackageExporter`
/// entirely, the same way an attacker or a hand-built malicious package
/// would) containing a single entry named `entry_name`, and confirm the
/// entry name survives the writer verbatim before it is used to test
/// `extract_package`'s rejection — otherwise a name-normalising writer
/// could make the test pass for the wrong reason.
fn write_malicious_package(path: &Path, entry_name: &str, data: &[u8]) {
    {
        let file = File::create(path).expect("create archive file");
        let mut zip = ZipWriter::new(file);
        zip.add_file(entry_name, data)
            .expect("writing a malicious entry name must itself succeed");
        zip.finish().expect("zip finish should succeed");
    }

    // Confirm the writer did not normalise/reject the malicious name.
    let file = File::open(path).expect("reopen archive for verification");
    let reader = ZipReader::new(file).expect("open zip for verification");
    let names: Vec<&str> = reader.entries().iter().map(|e| e.name.as_str()).collect();
    assert!(
        names.contains(&entry_name),
        "test setup invariant violated: the crafted entry name {:?} did not \
         survive the ZIP writer verbatim (got {:?}); the rejection asserted \
         below would not actually exercise the '..'/absolute-path guard",
        entry_name,
        names
    );
}

#[test]
fn f041_extract_package_rejects_parent_dir_traversal() {
    let temp_dir = TempDir::new().expect("tempdir");
    let package_path = temp_dir.path().join("evil.torshpkg");
    let output_dir = temp_dir.path().join("extract_here");

    write_malicious_package(&package_path, "../../.bashrc", b"pwned-by-zip-slip");

    let importer = PackageImporter::new(ImportConfig::default());
    let result = importer.extract_package(&package_path, &output_dir);

    assert!(
        result.is_err(),
        "extract_package must reject a '..'-escaping entry name"
    );
    assert!(!temp_dir.path().join(".bashrc").exists());
    assert!(!temp_dir.path().parent().unwrap().join(".bashrc").exists());
}

#[test]
fn f041_extract_package_rejects_absolute_path() {
    let temp_dir = TempDir::new().expect("tempdir");
    let package_path = temp_dir.path().join("evil_abs.torshpkg");
    let output_dir = temp_dir.path().join("extract_here");

    let absolute_name = "/torsh_hardening_test_absolute_evil.txt";
    write_malicious_package(&package_path, absolute_name, b"pwned-by-absolute-path");

    let importer = PackageImporter::new(ImportConfig::default());
    let result = importer.extract_package(&package_path, &output_dir);

    assert!(
        result.is_err(),
        "extract_package must reject an absolute entry path"
    );
    assert!(!Path::new(absolute_name).exists());
}

#[test]
fn f041_extract_package_accepts_benign_relative_path() {
    let temp_dir = TempDir::new().expect("tempdir");
    let package_path = temp_dir.path().join("benign.torshpkg");
    let output_dir = temp_dir.path().join("extract_here");

    write_malicious_package(&package_path, "sub/model.bin", b"legit-content");

    let importer = PackageImporter::new(ImportConfig::default());
    importer
        .extract_package(&package_path, &output_dir)
        .expect("benign entry should extract cleanly");

    let extracted = output_dir.join("sub").join("model.bin");
    assert!(extracted.exists());
    assert_eq!(
        std::fs::read(&extracted).expect("read extracted file"),
        b"legit-content"
    );
}

// ---------------------------------------------------------------------
// F240: package export/import roundtrip must preserve distinct resources
// ---------------------------------------------------------------------

#[test]
fn f240_roundtrip_preserves_two_resources_sharing_a_basename() {
    let temp_dir = TempDir::new().expect("tempdir");
    let package_path = temp_dir.path().join("roundtrip.torshpkg");

    let mut package = Package::new("roundtrip_pkg".to_string(), "1.0.0".to_string());

    // Two resources whose *map keys* (and therefore, per
    // `PackageExporter::write_resource`, their archive paths) differ only
    // in an inner subdirectory, but which share a basename
    // ("config.json"). Before the fix, importing collapsed both to the
    // same basename-only key and one silently overwrote the other.
    let mut model_metadata = std::collections::HashMap::new();
    model_metadata.insert("owner".to_string(), "model-team".to_string());
    let mut model_config = Resource::new(
        "config.json".to_string(),
        ResourceType::Model,
        b"MODEL-CONFIG-CONTENT".to_vec(),
    );
    model_config.metadata = model_metadata.clone();

    let other_config = Resource::new(
        "config.json".to_string(),
        ResourceType::Config,
        b"OTHER-CONFIG-CONTENT".to_vec(),
    );

    package
        .resources_mut()
        .insert("config.json".to_string(), model_config);
    package
        .resources_mut()
        .insert("other/config.json".to_string(), other_config);

    let exporter = PackageExporter::new(ExportConfig::default());
    exporter
        .export_package(&package, &package_path)
        .expect("export should succeed");

    let importer = PackageImporter::new(ImportConfig {
        verify_integrity: false,
        ..ImportConfig::default()
    });
    let imported = importer
        .import_package(&package_path)
        .expect("import should succeed");

    assert_eq!(
        imported.resources().len(),
        2,
        "both resources must survive the roundtrip distinctly, not collapse into one \
         (got keys: {:?})",
        imported.resources().keys().collect::<Vec<_>>()
    );

    let model_res = imported
        .resources()
        .get("config.json")
        .expect("the top-level 'config.json' resource must be preserved under its original key");
    assert_eq!(model_res.data, b"MODEL-CONFIG-CONTENT");
    assert_eq!(model_res.resource_type, ResourceType::Model);
    assert_eq!(
        model_res.metadata.get("owner").map(String::as_str),
        Some("model-team"),
        "resource metadata must also survive the roundtrip"
    );

    let other_res = imported
        .resources()
        .get("other/config.json")
        .expect("the nested 'other/config.json' resource must be preserved under its original key");
    assert_eq!(other_res.data, b"OTHER-CONFIG-CONTENT");
    assert_eq!(other_res.resource_type, ResourceType::Config);
}

#[test]
fn f240_roundtrip_preserves_single_nested_resource_path() {
    let temp_dir = TempDir::new().expect("tempdir");
    let package_path = temp_dir.path().join("nested.torshpkg");

    let mut package = Package::new("nested_pkg".to_string(), "1.0.0".to_string());
    package.resources_mut().insert(
        "weights/layer1/kernel.bin".to_string(),
        Resource::new(
            "weights/layer1/kernel.bin".to_string(),
            ResourceType::Model,
            b"KERNEL-BYTES".to_vec(),
        ),
    );

    let exporter = PackageExporter::new(ExportConfig::default());
    exporter
        .export_package(&package, &package_path)
        .expect("export should succeed");

    let importer = PackageImporter::new(ImportConfig {
        verify_integrity: false,
        ..ImportConfig::default()
    });
    let imported = importer
        .import_package(&package_path)
        .expect("import should succeed");

    assert_eq!(imported.resources().len(), 1);
    let res = imported
        .resources()
        .get("weights/layer1/kernel.bin")
        .expect("nested subdirectory structure in the resource name must survive the roundtrip");
    assert_eq!(res.data, b"KERNEL-BYTES");
    assert_eq!(res.resource_type, ResourceType::Model);
}

/// F240 (type half): every [`ResourceType`] whose archive path falls into
/// `PackageExporter::write_resource`'s `_ => "resources/{name}"` fallback
/// (License, Binary, Text, Metadata all share that one prefix) must still
/// come back as its original type after a roundtrip, not whatever
/// `determine_resource_type` would re-derive from `name`'s file extension.
///
/// Each name below is deliberately chosen so that extension-based guessing
/// (`ResourceType::from_extension`) would produce the *wrong* answer,
/// proving the type is actually carried by the `.metadata` side-file
/// (F240) rather than by a lucky coincidence between the extension and the
/// original type.
#[test]
fn f240_roundtrip_preserves_resource_type_for_fallback_prefix_types() {
    let temp_dir = TempDir::new().expect("tempdir");
    let package_path = temp_dir.path().join("types.torshpkg");

    let mut package = Package::new("types_pkg".to_string(), "1.0.0".to_string());
    // "txt" extension alone would extension-classify as License, not Text.
    package.resources_mut().insert(
        "notes.txt".to_string(),
        Resource::new(
            "notes.txt".to_string(),
            ResourceType::Text,
            b"just a text note".to_vec(),
        ),
    );
    // No extension at all falls through `from_extension`'s `_ => Data` arm.
    package.resources_mut().insert(
        "NOTICE".to_string(),
        Resource::new(
            "NOTICE".to_string(),
            ResourceType::License,
            b"Copyright ...".to_vec(),
        ),
    );
    // "bin" extension alone would extension-classify as Binary, not Metadata.
    package.resources_mut().insert(
        "state.bin".to_string(),
        Resource::new(
            "state.bin".to_string(),
            ResourceType::Metadata,
            b"{}".to_vec(),
        ),
    );
    // "raw" is not in `from_extension`'s match at all, falls to `_ => Data`.
    package.resources_mut().insert(
        "blob.raw".to_string(),
        Resource::new(
            "blob.raw".to_string(),
            ResourceType::Binary,
            b"\x00\x01\x02".to_vec(),
        ),
    );

    let exporter = PackageExporter::new(ExportConfig::default());
    exporter
        .export_package(&package, &package_path)
        .expect("export should succeed");

    let importer = PackageImporter::new(ImportConfig {
        verify_integrity: false,
        ..ImportConfig::default()
    });
    let imported = importer
        .import_package(&package_path)
        .expect("import should succeed");

    assert_eq!(imported.resources().len(), 4);
    for (key, expected_type) in [
        ("notes.txt", ResourceType::Text),
        ("NOTICE", ResourceType::License),
        ("state.bin", ResourceType::Metadata),
        ("blob.raw", ResourceType::Binary),
    ] {
        let res = imported
            .resources()
            .get(key)
            .unwrap_or_else(|| panic!("resource {:?} must survive the roundtrip", key));
        assert_eq!(
            res.resource_type, expected_type,
            "resource {:?} changed type across the roundtrip: expected {:?}, got {:?} \
             (extension-based classification must not override the recorded type)",
            key, expected_type, res.resource_type
        );
    }
}
