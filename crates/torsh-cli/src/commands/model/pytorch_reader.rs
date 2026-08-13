//! Real reader for modern (zip-based) PyTorch checkpoints (`torch.save`, >= 1.6).
//!
//! This is a genuine deserializer: it reads the STORED zip entries of a `.pt`
//! archive, runs a minimal but real pickle virtual machine over `data.pkl` to
//! recover the `state_dict` structure (`torch._utils._rebuild_tensor_v2` over
//! typed storages), then reads the raw storage bytes and reconstructs real
//! tensors honouring `size`/`stride`. Nothing here fabricates tensor values.
//!
//! Supported storage dtypes: `FloatStorage` (f32), `DoubleStorage` (f64) and
//! `LongStorage` (i64). Anything else — including DEFLATE-compressed archives —
//! returns an honest error rather than guessing.

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

/// Element dtype of a reconstructed tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    /// 32-bit float (`torch.FloatStorage`).
    F32,
    /// 64-bit float (`torch.DoubleStorage`).
    F64,
    /// 64-bit signed integer (`torch.LongStorage`).
    I64,
}

impl TensorDType {
    /// Size in bytes of one element.
    pub fn elem_size(self) -> usize {
        match self {
            TensorDType::F32 => 4,
            TensorDType::F64 => 8,
            TensorDType::I64 => 8,
        }
    }
}

/// Reconstructed dense tensor data in row-major (C-contiguous) order.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 32-bit float values.
    F32(Vec<f32>),
    /// 64-bit float values.
    F64(Vec<f64>),
    /// 64-bit signed integer values.
    I64(Vec<i64>),
}

/// A real tensor recovered from a PyTorch checkpoint.
#[derive(Debug, Clone)]
pub struct PytorchTensor {
    /// Fully-qualified state-dict key (e.g. `layer1.0.weight`).
    pub name: String,
    /// Element dtype.
    pub dtype: TensorDType,
    /// Logical shape.
    pub shape: Vec<usize>,
    /// Whether the checkpoint marked this tensor as requiring gradients.
    pub requires_grad: bool,
    /// Dense row-major data.
    pub data: TensorData,
}

impl PytorchTensor {
    /// Number of real elements reconstructed from the storage.
    pub fn element_count(&self) -> usize {
        match &self.data {
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }
}

/// Read and reconstruct every tensor in a zip-based PyTorch checkpoint.
///
/// Returns tensors in the order they appear in the pickled `state_dict`.
pub fn read_state_dict(file_bytes: &[u8]) -> Result<Vec<PytorchTensor>> {
    if !(file_bytes.len() >= 4 && &file_bytes[0..4] == b"PK\x03\x04") {
        bail!("not a zip-based PyTorch checkpoint (missing PK\\x03\\x04 signature); legacy pure-pickle .pt files are not supported");
    }

    let entries = read_stored_zip_entries(file_bytes)?;

    // Locate the pickle stream (`.../data.pkl`).
    let pickle = entries
        .iter()
        .find(|(name, _)| name.ends_with("data.pkl"))
        .map(|(_, body)| body.clone())
        .ok_or_else(|| anyhow!("checkpoint zip has no data.pkl entry"))?;

    // Index storages by their numeric key (`.../data/<key>`).
    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, body) in &entries {
        if let Some(pos) = name.rfind("/data/") {
            let key = &name[pos + "/data/".len()..];
            if !key.is_empty() && key.bytes().all(|b| b.is_ascii_digit()) {
                storages.insert(key.to_string(), body.clone());
            }
        }
    }

    let specs = run_pickle(&pickle)?;

    let mut tensors = Vec::with_capacity(specs.len());
    for spec in specs {
        let storage = storages.get(&spec.storage_key).ok_or_else(|| {
            anyhow!(
                "state_dict references storage '{}' that is not present in the archive",
                spec.storage_key
            )
        })?;
        tensors.push(reconstruct_tensor(spec, storage)?);
    }

    Ok(tensors)
}

// ---------------------------------------------------------------------------
// STORED zip reading (no compression, as produced by torch.save)
// ---------------------------------------------------------------------------

/// Read all STORED (uncompressed) entries from a zip via its local file headers.
fn read_stored_zip_entries(data: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    const LOCAL_SIG: &[u8] = b"PK\x03\x04";
    let mut entries = Vec::new();
    let mut cursor = 0usize;

    while cursor + 30 <= data.len() && &data[cursor..cursor + 4] == LOCAL_SIG {
        let method = u16::from_le_bytes([data[cursor + 8], data[cursor + 9]]);
        let comp_size = u32::from_le_bytes([
            data[cursor + 18],
            data[cursor + 19],
            data[cursor + 20],
            data[cursor + 21],
        ]) as usize;
        let uncomp_size = u32::from_le_bytes([
            data[cursor + 22],
            data[cursor + 23],
            data[cursor + 24],
            data[cursor + 25],
        ]) as usize;
        let name_len = u16::from_le_bytes([data[cursor + 26], data[cursor + 27]]) as usize;
        let extra_len = u16::from_le_bytes([data[cursor + 28], data[cursor + 29]]) as usize;

        let name_start = cursor + 30;
        let name_end = name_start + name_len;
        if name_end > data.len() {
            bail!("corrupt zip: entry name runs past end of file");
        }
        let name = String::from_utf8_lossy(&data[name_start..name_end]).to_string();

        let body_start = name_end + extra_len;
        let body_end = body_start + comp_size;
        if body_end > data.len() {
            bail!("corrupt zip: entry '{}' body runs past end of file", name);
        }

        // Directory entries (trailing '/') have empty bodies; skip them.
        if !name.ends_with('/') {
            if method != 0 {
                bail!(
                    "zip entry '{}' uses compression method {} but only STORED (0) is supported; \
                     torch.save archives are stored uncompressed",
                    name,
                    method
                );
            }
            if comp_size != uncomp_size {
                bail!("zip entry '{}' has inconsistent STORED sizes", name);
            }
            entries.push((name, data[body_start..body_end].to_vec()));
        }

        cursor = body_end;
    }

    if entries.is_empty() {
        bail!("no readable STORED zip entries found");
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// Minimal pickle virtual machine (enough for torch state_dicts)
// ---------------------------------------------------------------------------

/// A tensor description recovered from the pickle stream.
#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    storage_key: String,
    dtype: TensorDType,
    storage_offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
    requires_grad: bool,
}

/// Objects manipulated on the pickle stack.
#[derive(Debug, Clone)]
enum Obj {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<Obj>),
    List(Vec<Obj>),
    Dict(Vec<(Obj, Obj)>),
    Global(String),
    Mark,
    /// A resolved persistent id describing a typed storage.
    Storage {
        dtype: TensorDType,
        key: String,
    },
    /// A reconstructed tensor (result of `_rebuild_tensor_v2`).
    Tensor(TensorSpecPartial),
    /// An opaque reduce result we do not need to interpret.
    Opaque,
}

/// Tensor spec before it is bound to a state-dict name.
#[derive(Debug, Clone)]
struct TensorSpecPartial {
    storage_key: String,
    dtype: TensorDType,
    storage_offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
    requires_grad: bool,
}

/// Run the pickle VM and extract the ordered list of tensor specs.
fn run_pickle(buf: &[u8]) -> Result<Vec<TensorSpec>> {
    let mut stack: Vec<Obj> = Vec::new();
    let mut memo: HashMap<u32, Obj> = HashMap::new();
    let mut pos = 0usize;
    let mut memo_counter: u32 = 0;

    macro_rules! need {
        ($n:expr) => {{
            if pos + $n > buf.len() {
                bail!("pickle stream truncated");
            }
        }};
    }

    let top: Obj = loop {
        if pos >= buf.len() {
            bail!("pickle stream ended without STOP");
        }
        let op = buf[pos];
        pos += 1;

        match op {
            0x80 => {
                // PROTO
                need!(1);
                pos += 1;
            }
            0x95 => {
                // FRAME
                need!(8);
                pos += 8;
            }
            b'(' => stack.push(Obj::Mark), // MARK
            b'.' => {
                // STOP
                break stack
                    .pop()
                    .ok_or_else(|| anyhow!("STOP with empty stack"))?;
            }
            b'0' => {
                // POP
                stack.pop();
            }
            b'}' => stack.push(Obj::Dict(Vec::new())), // EMPTY_DICT
            b']' => stack.push(Obj::List(Vec::new())), // EMPTY_LIST
            b')' => stack.push(Obj::Tuple(Vec::new())), // EMPTY_TUPLE
            b'N' => stack.push(Obj::None),             // NONE
            0x88 => stack.push(Obj::Bool(true)),       // NEWTRUE
            0x89 => stack.push(Obj::Bool(false)),      // NEWFALSE
            b'K' => {
                // BININT1
                need!(1);
                stack.push(Obj::Int(buf[pos] as i64));
                pos += 1;
            }
            b'M' => {
                // BININT2
                need!(2);
                stack.push(Obj::Int(u16::from_le_bytes([buf[pos], buf[pos + 1]]) as i64));
                pos += 2;
            }
            b'J' => {
                // BININT (signed 4-byte)
                need!(4);
                stack.push(Obj::Int(i32::from_le_bytes([
                    buf[pos],
                    buf[pos + 1],
                    buf[pos + 2],
                    buf[pos + 3],
                ]) as i64));
                pos += 4;
            }
            0x8a => {
                // LONG1
                need!(1);
                let n = buf[pos] as usize;
                pos += 1;
                need!(n);
                stack.push(Obj::Int(read_signed_le(&buf[pos..pos + n])));
                pos += n;
            }
            b'G' => {
                // BINFLOAT (big-endian double): consumed but not needed for the
                // tensor structure, so kept as an opaque value.
                need!(8);
                pos += 8;
                stack.push(Obj::Opaque);
            }
            0x8c => {
                // SHORT_BINUNICODE
                need!(1);
                let n = buf[pos] as usize;
                pos += 1;
                need!(n);
                stack.push(Obj::Str(str_from(&buf[pos..pos + n])?));
                pos += n;
            }
            b'X' => {
                // BINUNICODE
                need!(4);
                let n = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
                    as usize;
                pos += 4;
                need!(n);
                stack.push(Obj::Str(str_from(&buf[pos..pos + n])?));
                pos += n;
            }
            b'q' => {
                // BINPUT
                need!(1);
                let idx = buf[pos] as u32;
                pos += 1;
                let obj = clone_top(&stack)?;
                memo.insert(idx, obj);
            }
            b'r' => {
                // LONG_BINPUT
                need!(4);
                let idx = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
                pos += 4;
                let obj = clone_top(&stack)?;
                memo.insert(idx, obj);
            }
            0x94 => {
                // MEMOIZE
                let obj = clone_top(&stack)?;
                memo.insert(memo_counter, obj);
                memo_counter += 1;
            }
            b'h' => {
                // BINGET
                need!(1);
                let idx = buf[pos] as u32;
                pos += 1;
                stack.push(memo_get(&memo, idx)?);
            }
            b'j' => {
                // LONG_BINGET
                need!(4);
                let idx = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
                pos += 4;
                stack.push(memo_get(&memo, idx)?);
            }
            b'c' => {
                // GLOBAL: module\nname\n
                let module = read_line(buf, &mut pos)?;
                let name = read_line(buf, &mut pos)?;
                stack.push(Obj::Global(format!("{module}.{name}")));
            }
            0x93 => {
                // STACK_GLOBAL: pop name, module
                let name = pop_str(&mut stack)?;
                let module = pop_str(&mut stack)?;
                stack.push(Obj::Global(format!("{module}.{name}")));
            }
            0x85 => tuple_n(&mut stack, 1)?, // TUPLE1
            0x86 => tuple_n(&mut stack, 2)?, // TUPLE2
            0x87 => tuple_n(&mut stack, 3)?, // TUPLE3
            b't' => {
                // TUPLE (mark based)
                let items = pop_to_mark(&mut stack)?;
                stack.push(Obj::Tuple(items));
            }
            b'a' => {
                // APPEND
                let v = stack.pop().ok_or_else(|| anyhow!("APPEND underflow"))?;
                if let Some(Obj::List(list)) = stack.last_mut() {
                    list.push(v);
                } else {
                    bail!("APPEND on non-list");
                }
            }
            b'e' => {
                // APPENDS
                let items = pop_to_mark(&mut stack)?;
                if let Some(Obj::List(list)) = stack.last_mut() {
                    list.extend(items);
                } else {
                    bail!("APPENDS on non-list");
                }
            }
            b's' => {
                // SETITEM
                let value = stack.pop().ok_or_else(|| anyhow!("SETITEM underflow"))?;
                let key = stack.pop().ok_or_else(|| anyhow!("SETITEM underflow"))?;
                if let Some(Obj::Dict(d)) = stack.last_mut() {
                    d.push((key, value));
                } else {
                    bail!("SETITEM on non-dict");
                }
            }
            b'u' => {
                // SETITEMS
                let items = pop_to_mark(&mut stack)?;
                if items.len() % 2 != 0 {
                    bail!("SETITEMS with odd number of items");
                }
                if let Some(Obj::Dict(d)) = stack.last_mut() {
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                } else {
                    bail!("SETITEMS on non-dict");
                }
            }
            b'Q' => {
                // BINPERSID: pop pid, resolve to storage
                let pid = stack.pop().ok_or_else(|| anyhow!("BINPERSID underflow"))?;
                stack.push(resolve_persid(pid)?);
            }
            b'R' => {
                // REDUCE
                let args = stack.pop().ok_or_else(|| anyhow!("REDUCE underflow"))?;
                let callable = stack.pop().ok_or_else(|| anyhow!("REDUCE underflow"))?;
                stack.push(apply_reduce(callable, args)?);
            }
            b'b' => {
                // BUILD: pop state, leave object (we ignore __setstate__ payloads)
                stack.pop().ok_or_else(|| anyhow!("BUILD underflow"))?;
            }
            other => bail!("unsupported pickle opcode: 0x{other:02x}"),
        }
    };

    // The top object is the state_dict (possibly nested). Collect tensor specs.
    let mut specs = Vec::new();
    collect_specs(&top, "", &mut specs)?;
    if specs.is_empty() {
        bail!("no tensors (via _rebuild_tensor_v2) were found in the checkpoint");
    }
    Ok(specs)
}

/// Recursively collect tensor specs from a (possibly nested) dict, prefixing
/// nested keys with their dotted path.
fn collect_specs(obj: &Obj, prefix: &str, out: &mut Vec<TensorSpec>) -> Result<()> {
    match obj {
        Obj::Dict(entries) => {
            for (k, v) in entries {
                let key = match k {
                    Obj::Str(s) => s.clone(),
                    Obj::Int(i) => i.to_string(),
                    _ => continue,
                };
                let path = if prefix.is_empty() {
                    key
                } else {
                    format!("{prefix}.{key}")
                };
                collect_specs(v, &path, out)?;
            }
            Ok(())
        }
        Obj::Tensor(p) => {
            out.push(TensorSpec {
                name: prefix.to_string(),
                storage_key: p.storage_key.clone(),
                dtype: p.dtype,
                storage_offset: p.storage_offset,
                size: p.size.clone(),
                stride: p.stride.clone(),
                requires_grad: p.requires_grad,
            });
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Resolve a persistent-id tuple `('storage', <StorageType>, <key>, <loc>, <numel>)`.
fn resolve_persid(pid: Obj) -> Result<Obj> {
    let items = match pid {
        Obj::Tuple(items) => items,
        _ => bail!("persistent id is not a tuple (unsupported checkpoint layout)"),
    };
    if items.len() < 3 {
        bail!("persistent id tuple too short");
    }
    match &items[0] {
        Obj::Str(s) if s == "storage" => {}
        _ => bail!("unsupported persistent id kind (expected 'storage')"),
    }
    let dtype = match &items[1] {
        Obj::Global(g) => storage_dtype(g)?,
        _ => bail!("persistent id storage type is not a global"),
    };
    let key = match &items[2] {
        Obj::Str(s) => s.clone(),
        Obj::Int(i) => i.to_string(),
        _ => bail!("persistent id storage key has unexpected type"),
    };
    Ok(Obj::Storage { dtype, key })
}

/// Map a `torch.*Storage` global to a supported dtype.
fn storage_dtype(global: &str) -> Result<TensorDType> {
    match global {
        "torch.FloatStorage" => Ok(TensorDType::F32),
        "torch.DoubleStorage" => Ok(TensorDType::F64),
        "torch.LongStorage" => Ok(TensorDType::I64),
        other => bail!(
            "unsupported storage type '{}': only FloatStorage, DoubleStorage and LongStorage are supported",
            other
        ),
    }
}

/// Apply a REDUCE for the callables we understand.
fn apply_reduce(callable: Obj, args: Obj) -> Result<Obj> {
    let name = match &callable {
        Obj::Global(g) => g.as_str(),
        _ => return Ok(Obj::Opaque),
    };
    match name {
        "torch._utils._rebuild_tensor_v2" | "torch._utils._rebuild_tensor" => {
            let a = match args {
                Obj::Tuple(a) => a,
                _ => bail!("_rebuild_tensor args are not a tuple"),
            };
            if a.len() < 4 {
                bail!("_rebuild_tensor expects at least 4 args, got {}", a.len());
            }
            let (dtype, storage_key) = match &a[0] {
                Obj::Storage { dtype, key } => (*dtype, key.clone()),
                _ => bail!("_rebuild_tensor first arg is not a storage"),
            };
            let storage_offset = as_usize(&a[1])?;
            let size = as_usize_tuple(&a[2])?;
            let stride = as_usize_tuple(&a[3])?;
            let requires_grad = matches!(a.get(4), Some(Obj::Bool(true)));
            Ok(Obj::Tensor(TensorSpecPartial {
                storage_key,
                dtype,
                storage_offset,
                size,
                stride,
                requires_grad,
            }))
        }
        // OrderedDict()/dict() constructors -> empty dict to be filled by SETITEMS.
        "collections.OrderedDict" | "builtins.dict" | "__builtin__.dict" => {
            Ok(Obj::Dict(Vec::new()))
        }
        _ => Ok(Obj::Opaque),
    }
}

/// Reconstruct a dense, row-major tensor from a storage buffer, honouring
/// `storage_offset`, `size` and `stride`.
fn reconstruct_tensor(spec: TensorSpec, storage: &[u8]) -> Result<PytorchTensor> {
    let elem = spec.dtype.elem_size();
    if storage.len() % elem != 0 {
        bail!(
            "storage '{}' length {} is not a multiple of element size {}",
            spec.storage_key,
            storage.len(),
            elem
        );
    }
    let n_storage_elems = storage.len() / elem;
    let numel: usize = spec.size.iter().product();

    if spec.size.len() != spec.stride.len() {
        bail!("tensor '{}' size/stride rank mismatch", spec.name);
    }

    // Row-major traversal of the logical index space, gathering from storage
    // through the provided strides.
    let mut indices = vec![0usize; spec.size.len()];
    let mut flat = Vec::with_capacity(numel);
    for _ in 0..numel {
        let mut storage_index = spec.storage_offset;
        for (dim, &idx) in indices.iter().enumerate() {
            storage_index += idx * spec.stride[dim];
        }
        if storage_index >= n_storage_elems {
            bail!(
                "tensor '{}' indexes element {} beyond storage of {} elements",
                spec.name,
                storage_index,
                n_storage_elems
            );
        }
        flat.push(storage_index);

        // Increment the multi-dimensional counter (row-major, last dim fastest).
        for dim in (0..spec.size.len()).rev() {
            indices[dim] += 1;
            if indices[dim] < spec.size[dim] {
                break;
            }
            indices[dim] = 0;
        }
    }

    let data = match spec.dtype {
        TensorDType::F32 => TensorData::F32(
            flat.iter()
                .map(|&i| {
                    let b = &storage[i * 4..i * 4 + 4];
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                })
                .collect(),
        ),
        TensorDType::F64 => TensorData::F64(
            flat.iter()
                .map(|&i| {
                    let b = &storage[i * 8..i * 8 + 8];
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                })
                .collect(),
        ),
        TensorDType::I64 => TensorData::I64(
            flat.iter()
                .map(|&i| {
                    let b = &storage[i * 8..i * 8 + 8];
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                })
                .collect(),
        ),
    };

    Ok(PytorchTensor {
        name: spec.name,
        dtype: spec.dtype,
        shape: spec.size,
        requires_grad: spec.requires_grad,
        data,
    })
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn read_signed_le(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut val: i64 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        val |= (b as i64) << (8 * i);
    }
    // Sign-extend from the top bit of the highest byte.
    let bits = bytes.len() * 8;
    if bits < 64 && (bytes[bytes.len() - 1] & 0x80) != 0 {
        val |= -1i64 << bits;
    }
    val
}

fn str_from(bytes: &[u8]) -> Result<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| anyhow!("invalid UTF-8 in pickle string"))
}

fn read_line(buf: &[u8], pos: &mut usize) -> Result<String> {
    let start = *pos;
    while *pos < buf.len() && buf[*pos] != b'\n' {
        *pos += 1;
    }
    if *pos >= buf.len() {
        bail!("unterminated GLOBAL line in pickle stream");
    }
    let s = str_from(&buf[start..*pos])?;
    *pos += 1; // consume '\n'
    Ok(s)
}

fn clone_top(stack: &[Obj]) -> Result<Obj> {
    stack
        .last()
        .cloned()
        .ok_or_else(|| anyhow!("memoize on empty stack"))
}

fn memo_get(memo: &HashMap<u32, Obj>, idx: u32) -> Result<Obj> {
    memo.get(&idx)
        .cloned()
        .ok_or_else(|| anyhow!("BINGET of unknown memo index {idx}"))
}

fn pop_str(stack: &mut Vec<Obj>) -> Result<String> {
    match stack.pop() {
        Some(Obj::Str(s)) => Ok(s),
        _ => bail!("expected a string on the pickle stack"),
    }
}

fn tuple_n(stack: &mut Vec<Obj>, n: usize) -> Result<()> {
    if stack.len() < n {
        bail!("TUPLE{n} underflow");
    }
    let items = stack.split_off(stack.len() - n);
    stack.push(Obj::Tuple(items));
    Ok(())
}

fn pop_to_mark(stack: &mut Vec<Obj>) -> Result<Vec<Obj>> {
    let mut items = Vec::new();
    while let Some(obj) = stack.pop() {
        if matches!(obj, Obj::Mark) {
            items.reverse();
            return Ok(items);
        }
        items.push(obj);
    }
    bail!("no MARK found on the pickle stack")
}

fn as_usize(obj: &Obj) -> Result<usize> {
    match obj {
        Obj::Int(i) if *i >= 0 => Ok(*i as usize),
        _ => bail!("expected a non-negative integer"),
    }
}

fn as_usize_tuple(obj: &Obj) -> Result<Vec<usize>> {
    match obj {
        Obj::Tuple(items) => items.iter().map(as_usize).collect(),
        _ => bail!("expected a tuple of integers (size/stride)"),
    }
}
