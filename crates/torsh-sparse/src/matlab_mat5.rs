//! MATLAB Level-5 MAT-file reading and writing for sparse matrices.
//!
//! MATLAB stores sparse matrices exactly the way [`crate::CscTensor`] does — an
//! `ir` array of row indices, a `jc` array of column pointers and a `pr` array
//! of values — so the mapping is direct. This module implements the documented
//! Level-5 container around those arrays:
//!
//! * a 128 byte header (description, version, endian indicator),
//! * a sequence of tagged data elements, each padded to an 8 byte boundary,
//! * `miMATRIX` elements holding array flags, dimensions, the variable name and
//!   the class specific payload.
//!
//! Files written by MATLAB usually wrap every variable in a `miCOMPRESSED`
//! element; those are inflated with `oxiarc-deflate` (Pure Rust zlib).
//! Files written here are uncompressed, which MATLAB reads without complaint.

use std::path::Path;
use torsh_core::{Result as TorshResult, TorshError};

// MAT-file data types.
const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;

// MAT-file array classes.
const MX_SPARSE: u32 = 5;
const MX_DOUBLE: u32 = 6;
const MX_SINGLE: u32 = 7;

/// A sparse matrix as stored in a MAT-file: CSC layout with 0-based indices.
#[derive(Debug, Clone)]
pub struct MatSparseMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Column pointers, `cols + 1` entries.
    pub col_ptr: Vec<usize>,
    /// Row indices, one per stored value.
    pub row_indices: Vec<usize>,
    /// Stored values.
    pub values: Vec<f64>,
}

fn invalid(message: impl Into<String>) -> TorshError {
    TorshError::InvalidArgument(message.into())
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

fn push_tag(out: &mut Vec<u8>, data_type: u32, byte_count: usize) {
    out.extend_from_slice(&data_type.to_le_bytes());
    out.extend_from_slice(&(byte_count as u32).to_le_bytes());
}

fn pad_to_eight(out: &mut Vec<u8>) {
    while out.len() % 8 != 0 {
        out.push(0);
    }
}

/// Serialise a sparse matrix into a complete Level-5 MAT-file image.
pub fn encode_sparse_mat(name: &str, matrix: &MatSparseMatrix) -> TorshResult<Vec<u8>> {
    if name.is_empty() {
        return Err(invalid("MATLAB variable name must not be empty"));
    }
    if matrix.col_ptr.len() != matrix.cols + 1 {
        return Err(invalid(format!(
            "column pointer must have cols + 1 = {} entries, got {}",
            matrix.cols + 1,
            matrix.col_ptr.len()
        )));
    }
    if matrix.row_indices.len() != matrix.values.len() {
        return Err(invalid(
            "row index and value arrays must have the same length",
        ));
    }

    let nnz = matrix.values.len();

    // ---- array payload -------------------------------------------------
    let mut payload: Vec<u8> = Vec::new();

    // Array flags: class in the low byte, no extra flags (real, non-global).
    push_tag(&mut payload, MI_UINT32, 8);
    payload.extend_from_slice(&MX_SPARSE.to_le_bytes());
    payload.extend_from_slice(&(nnz.max(1) as u32).to_le_bytes()); // nzmax

    // Dimensions.
    push_tag(&mut payload, MI_INT32, 8);
    payload.extend_from_slice(&(matrix.rows as i32).to_le_bytes());
    payload.extend_from_slice(&(matrix.cols as i32).to_le_bytes());

    // Variable name.
    push_tag(&mut payload, MI_INT8, name.len());
    payload.extend_from_slice(name.as_bytes());
    pad_to_eight(&mut payload);

    // Row indices (ir).
    push_tag(&mut payload, MI_INT32, nnz * 4);
    for &row in &matrix.row_indices {
        payload.extend_from_slice(&(row as i32).to_le_bytes());
    }
    pad_to_eight(&mut payload);

    // Column pointers (jc).
    push_tag(&mut payload, MI_INT32, matrix.col_ptr.len() * 4);
    for &ptr in &matrix.col_ptr {
        payload.extend_from_slice(&(ptr as i32).to_le_bytes());
    }
    pad_to_eight(&mut payload);

    // Values (pr).
    push_tag(&mut payload, MI_DOUBLE, nnz * 8);
    for &value in &matrix.values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    pad_to_eight(&mut payload);

    // ---- file image ----------------------------------------------------
    let mut out: Vec<u8> = Vec::with_capacity(128 + 8 + payload.len());

    let description = "MATLAB 5.0 MAT-file, written by ToRSh torsh-sparse";
    let mut header = description.as_bytes().to_vec();
    header.resize(116, b' ');
    out.extend_from_slice(&header);
    out.extend_from_slice(&[0u8; 8]); // subsystem data offset
    out.extend_from_slice(&0x0100u16.to_le_bytes()); // version
    out.extend_from_slice(b"IM"); // little-endian indicator

    push_tag(&mut out, MI_MATRIX, payload.len());
    out.extend_from_slice(&payload);

    Ok(out)
}

/// Write a sparse matrix to `path` as a Level-5 MAT-file.
pub fn write_sparse_mat(path: &Path, name: &str, matrix: &MatSparseMatrix) -> TorshResult<()> {
    let bytes = encode_sparse_mat(name, matrix)?;
    std::fs::write(path, bytes)
        .map_err(|e| TorshError::IoError(format!("failed to write .mat file: {e}")))
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

struct Element<'a> {
    data_type: u32,
    data: &'a [u8],
    /// Total size of the element including its tag and padding.
    total_len: usize,
}

fn read_u32(bytes: &[u8], offset: usize) -> TorshResult<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|slice| slice.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| invalid("truncated .mat file: expected a 4 byte field"))
}

/// Parse the tagged element starting at `offset`.
fn read_element(bytes: &[u8], offset: usize) -> TorshResult<Element<'_>> {
    let raw = read_u32(bytes, offset)?;

    // Small data element format: the upper 16 bits hold the byte count.
    if raw >> 16 != 0 {
        let data_type = raw & 0xFFFF;
        let byte_count = (raw >> 16) as usize;
        if byte_count > 4 {
            return Err(invalid("malformed small data element in .mat file"));
        }
        let data = bytes
            .get(offset + 4..offset + 4 + byte_count)
            .ok_or_else(|| invalid("truncated small data element in .mat file"))?;
        return Ok(Element {
            data_type,
            data,
            total_len: 8,
        });
    }

    let byte_count = read_u32(bytes, offset + 4)? as usize;
    let data = bytes
        .get(offset + 8..offset + 8 + byte_count)
        .ok_or_else(|| invalid("truncated data element in .mat file"))?;
    let padded = byte_count.div_ceil(8) * 8;
    Ok(Element {
        data_type: raw,
        data,
        total_len: 8 + padded,
    })
}

/// Decode a numeric element into `f64` values.
fn numeric_to_f64(element: &Element<'_>) -> TorshResult<Vec<f64>> {
    let data = element.data;
    let values = match element.data_type {
        MI_INT8 => data.iter().map(|&b| b as i8 as f64).collect(),
        MI_UINT8 | MI_UTF8 => data.iter().map(|&b| b as f64).collect(),
        MI_INT16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        MI_UINT16 => data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        MI_INT32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        MI_UINT32 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        MI_SINGLE => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        MI_DOUBLE => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        MI_INT64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64)
            .collect(),
        MI_UINT64 => data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64)
            .collect(),
        other => {
            return Err(invalid(format!(
                "unsupported numeric data type {other} in .mat file"
            )))
        }
    };
    Ok(values)
}

fn numeric_to_usize(element: &Element<'_>) -> TorshResult<Vec<usize>> {
    let values = numeric_to_f64(element)?;
    values
        .into_iter()
        .map(|v| {
            if v < 0.0 {
                Err(invalid("negative index in .mat sparse matrix"))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

/// Decode a `miMATRIX` payload, returning the variable name and, when the
/// variable is a matrix this crate can represent, the sparse data.
fn decode_matrix(payload: &[u8]) -> TorshResult<(String, Option<MatSparseMatrix>)> {
    let mut offset = 0usize;

    let flags_element = read_element(payload, offset)?;
    offset += flags_element.total_len;
    if flags_element.data.len() < 8 {
        return Err(invalid("malformed array flags in .mat file"));
    }
    let flags = read_u32(flags_element.data, 0)?;
    let class = flags & 0xFF;
    let is_complex = flags & 0x0800 != 0;

    let dims_element = read_element(payload, offset)?;
    offset += dims_element.total_len;
    let dims = numeric_to_usize(&dims_element)?;
    if dims.len() != 2 {
        return Err(invalid(format!(
            "only 2-D MATLAB arrays are supported, got {} dimensions",
            dims.len()
        )));
    }

    let name_element = read_element(payload, offset)?;
    offset += name_element.total_len;
    let name = String::from_utf8_lossy(name_element.data).to_string();

    if is_complex {
        return Err(invalid(format!(
            "complex MATLAB array '{name}' is not supported by torsh-sparse"
        )));
    }

    let (rows, cols) = (dims[0], dims[1]);

    match class {
        MX_SPARSE => {
            let ir_element = read_element(payload, offset)?;
            offset += ir_element.total_len;
            let row_indices = numeric_to_usize(&ir_element)?;

            let jc_element = read_element(payload, offset)?;
            offset += jc_element.total_len;
            let col_ptr = numeric_to_usize(&jc_element)?;

            let pr_element = read_element(payload, offset)?;
            let values = numeric_to_f64(&pr_element)?;

            if col_ptr.len() != cols + 1 {
                return Err(invalid(format!(
                    "sparse variable '{name}': jc has {} entries, expected {}",
                    col_ptr.len(),
                    cols + 1
                )));
            }
            let nnz = col_ptr[cols];
            if row_indices.len() < nnz || values.len() < nnz {
                return Err(invalid(format!(
                    "sparse variable '{name}': ir/pr shorter than jc[end] = {nnz}"
                )));
            }

            Ok((
                name,
                Some(MatSparseMatrix {
                    rows,
                    cols,
                    col_ptr,
                    row_indices: row_indices[..nnz].to_vec(),
                    values: values[..nnz].to_vec(),
                }),
            ))
        }
        MX_DOUBLE | MX_SINGLE => {
            // Full numeric array stored column-major: convert to sparse by
            // dropping the exact zeros.
            let pr_element = read_element(payload, offset)?;
            let dense = numeric_to_f64(&pr_element)?;
            if dense.len() < rows * cols {
                return Err(invalid(format!(
                    "variable '{name}': expected {} elements, got {}",
                    rows * cols,
                    dense.len()
                )));
            }

            let mut col_ptr = Vec::with_capacity(cols + 1);
            let mut row_indices = Vec::new();
            let mut values = Vec::new();
            col_ptr.push(0);
            for col in 0..cols {
                for row in 0..rows {
                    let value = dense[col * rows + row];
                    if value != 0.0 {
                        row_indices.push(row);
                        values.push(value);
                    }
                }
                col_ptr.push(values.len());
            }

            Ok((
                name,
                Some(MatSparseMatrix {
                    rows,
                    cols,
                    col_ptr,
                    row_indices,
                    values,
                }),
            ))
        }
        // Cells, structs, chars and friends: report the name so the caller can
        // list what the file contains, but no data.
        _ => Ok((name, None)),
    }
}

/// Read every top-level variable of a MAT-file image, calling `visit` with the
/// decoded name and payload.
fn for_each_variable(
    bytes: &[u8],
    visit: &mut impl FnMut(String, Option<MatSparseMatrix>) -> TorshResult<bool>,
) -> TorshResult<()> {
    if bytes.len() < 128 {
        return Err(invalid("file is too short to be a MAT-file"));
    }
    match &bytes[126..128] {
        b"IM" => {}
        b"MI" => {
            return Err(invalid(
                "big-endian MAT-files are not supported; re-save the file on a little-endian host",
            ))
        }
        _ => return Err(invalid("not a Level-5 MAT-file (bad endian indicator)")),
    }

    let mut offset = 128usize;
    while offset + 8 <= bytes.len() {
        let element = read_element(bytes, offset)?;
        offset += element.total_len;

        match element.data_type {
            MI_COMPRESSED => {
                let inflated = oxiarc_deflate::zlib_decompress(element.data).map_err(|e| {
                    invalid(format!("failed to inflate compressed .mat element: {e}"))
                })?;
                let inner = read_element(&inflated, 0)?;
                if inner.data_type == MI_MATRIX {
                    let (name, matrix) = decode_matrix(inner.data)?;
                    if !visit(name, matrix)? {
                        return Ok(());
                    }
                }
            }
            MI_MATRIX => {
                let (name, matrix) = decode_matrix(element.data)?;
                if !visit(name, matrix)? {
                    return Ok(());
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// List the names of the top-level variables in a MAT-file.
pub fn list_variables(path: &Path) -> TorshResult<Vec<String>> {
    let bytes = std::fs::read(path)
        .map_err(|e| TorshError::IoError(format!("failed to read .mat file: {e}")))?;
    let mut names = Vec::new();
    for_each_variable(&bytes, &mut |name, _| {
        names.push(name);
        Ok(true)
    })?;
    Ok(names)
}

/// Read the named variable from a MAT-file as a sparse matrix.
///
/// Both genuine MATLAB sparse matrices and full double/single matrices are
/// accepted; a full matrix is converted by dropping its exact zeros.
pub fn read_sparse_mat(path: &Path, name: &str) -> TorshResult<MatSparseMatrix> {
    let bytes = std::fs::read(path)
        .map_err(|e| TorshError::IoError(format!("failed to read .mat file: {e}")))?;

    let mut found: Option<MatSparseMatrix> = None;
    let mut seen: Vec<String> = Vec::new();
    for_each_variable(&bytes, &mut |variable, matrix| {
        if variable == name {
            match matrix {
                Some(data) => {
                    found = Some(data);
                    Ok(false)
                }
                None => Err(invalid(format!(
                    "variable '{name}' is not a numeric or sparse matrix"
                ))),
            }
        } else {
            seen.push(variable);
            Ok(true)
        }
    })?;

    found.ok_or_else(|| {
        invalid(format!(
            "variable '{name}' not found in .mat file. Available variables: {seen:?}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MatSparseMatrix {
        // 3x4 matrix with entries (0,0)=1.5, (2,1)=-2.5, (1,3)=7.0
        MatSparseMatrix {
            rows: 3,
            cols: 4,
            col_ptr: vec![0, 1, 2, 2, 3],
            row_indices: vec![0, 2, 1],
            values: vec![1.5, -2.5, 7.0],
        }
    }

    #[test]
    fn round_trip_through_a_file() -> TorshResult<()> {
        let matrix = sample();
        let mut path = std::env::temp_dir();
        path.push(format!("torsh_mat5_round_trip_{}.mat", std::process::id()));

        write_sparse_mat(&path, "spmat", &matrix)?;
        let decoded = read_sparse_mat(&path, "spmat")?;
        let names = list_variables(&path)?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(names, vec!["spmat".to_string()]);
        assert_eq!(decoded.rows, matrix.rows);
        assert_eq!(decoded.cols, matrix.cols);
        assert_eq!(decoded.col_ptr, matrix.col_ptr);
        assert_eq!(decoded.row_indices, matrix.row_indices);
        assert_eq!(decoded.values, matrix.values);
        Ok(())
    }

    #[test]
    fn header_layout_matches_the_specification() -> TorshResult<()> {
        let bytes = encode_sparse_mat("spmat", &sample())?;
        assert!(bytes.len() > 128);
        assert_eq!(&bytes[126..128], b"IM");
        assert_eq!(u16::from_le_bytes([bytes[124], bytes[125]]), 0x0100);
        assert_eq!(
            u32::from_le_bytes([bytes[128], bytes[129], bytes[130], bytes[131]]),
            MI_MATRIX
        );
        // The element must cover the rest of the file exactly.
        let payload_len =
            u32::from_le_bytes([bytes[132], bytes[133], bytes[134], bytes[135]]) as usize;
        assert_eq!(128 + 8 + payload_len, bytes.len());
        Ok(())
    }

    #[test]
    fn missing_variable_is_reported() -> TorshResult<()> {
        let bytes = encode_sparse_mat("spmat", &sample())?;
        let mut path = std::env::temp_dir();
        path.push(format!("torsh_mat5_missing_{}.mat", std::process::id()));
        std::fs::write(&path, bytes)
            .map_err(|e| TorshError::IoError(format!("failed to write: {e}")))?;
        let result = read_sparse_mat(&path, "other");
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn parses_small_data_element_format() -> TorshResult<()> {
        // MATLAB packs elements of at most 4 bytes into the tag itself: the low
        // 16 bits hold the data type, the high 16 bits the byte count. Our
        // writer always emits the long form, so build such a file by hand.
        let mut payload: Vec<u8> = Vec::new();

        push_tag(&mut payload, MI_UINT32, 8);
        payload.extend_from_slice(&MX_SPARSE.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());

        push_tag(&mut payload, MI_INT32, 8);
        payload.extend_from_slice(&2i32.to_le_bytes());
        payload.extend_from_slice(&2i32.to_le_bytes());

        // Name "x" in small format.
        payload.extend_from_slice(&(MI_INT8 | (1u32 << 16)).to_le_bytes());
        payload.extend_from_slice(&[b'x', 0, 0, 0]);

        // ir = [1] in small format (exactly 4 bytes).
        payload.extend_from_slice(&(MI_INT32 | (4u32 << 16)).to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());

        // jc = [0, 0, 1] in long format.
        push_tag(&mut payload, MI_INT32, 12);
        payload.extend_from_slice(&0i32.to_le_bytes());
        payload.extend_from_slice(&0i32.to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());
        pad_to_eight(&mut payload);

        // pr = [2.5].
        push_tag(&mut payload, MI_DOUBLE, 8);
        payload.extend_from_slice(&2.5f64.to_le_bytes());

        let mut bytes: Vec<u8> = Vec::new();
        let mut header = b"MATLAB 5.0 MAT-file, hand built".to_vec();
        header.resize(116, b' ');
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&0x0100u16.to_le_bytes());
        bytes.extend_from_slice(b"IM");
        push_tag(&mut bytes, MI_MATRIX, payload.len());
        bytes.extend_from_slice(&payload);

        let mut path = std::env::temp_dir();
        path.push(format!("torsh_mat5_small_{}.mat", std::process::id()));
        std::fs::write(&path, bytes)
            .map_err(|e| TorshError::IoError(format!("failed to write: {e}")))?;
        let decoded = read_sparse_mat(&path, "x")?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded.rows, 2);
        assert_eq!(decoded.cols, 2);
        assert_eq!(decoded.col_ptr, vec![0, 0, 1]);
        assert_eq!(decoded.row_indices, vec![1]);
        assert_eq!(decoded.values, vec![2.5]);
        Ok(())
    }

    #[test]
    fn full_double_matrix_is_converted_to_sparse() -> TorshResult<()> {
        // 2x2 full double matrix [[0, 3], [0, 0]] stored column-major.
        let mut payload: Vec<u8> = Vec::new();
        push_tag(&mut payload, MI_UINT32, 8);
        payload.extend_from_slice(&MX_DOUBLE.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        push_tag(&mut payload, MI_INT32, 8);
        payload.extend_from_slice(&2i32.to_le_bytes());
        payload.extend_from_slice(&2i32.to_le_bytes());
        push_tag(&mut payload, MI_INT8, 1);
        payload.push(b'd');
        pad_to_eight(&mut payload);
        push_tag(&mut payload, MI_DOUBLE, 32);
        for value in [0.0f64, 0.0, 3.0, 0.0] {
            payload.extend_from_slice(&value.to_le_bytes());
        }

        let mut bytes: Vec<u8> = Vec::new();
        bytes.resize(116, b' ');
        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&0x0100u16.to_le_bytes());
        bytes.extend_from_slice(b"IM");
        push_tag(&mut bytes, MI_MATRIX, payload.len());
        bytes.extend_from_slice(&payload);

        let mut path = std::env::temp_dir();
        path.push(format!("torsh_mat5_full_{}.mat", std::process::id()));
        std::fs::write(&path, bytes)
            .map_err(|e| TorshError::IoError(format!("failed to write: {e}")))?;
        let decoded = read_sparse_mat(&path, "d")?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded.col_ptr, vec![0, 0, 1]);
        assert_eq!(decoded.row_indices, vec![0]);
        assert_eq!(decoded.values, vec![3.0]);
        Ok(())
    }

    #[test]
    fn rejects_non_mat_files() {
        let mut path = std::env::temp_dir();
        path.push(format!("torsh_mat5_garbage_{}.mat", std::process::id()));
        let _ = std::fs::write(&path, vec![0u8; 256]);
        let result = read_sparse_mat(&path, "x");
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }
}
