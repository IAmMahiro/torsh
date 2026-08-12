//! Real single-node / multi-process TCP collective backend (Gloo-equivalent).
//!
//! This backend implements genuine cross-rank collective communication using
//! the crate's [`TcpStore`](crate::store::TcpStore) for rendezvous **and** data
//! exchange. Unlike the historical mock backends it performs **no fabricated
//! reductions**: data physically crosses TCP sockets between ranks, and every
//! collective returns an honest error when it cannot complete.
//!
//! # Topology
//!
//! The data plane is a **star through the master** (rank 0 runs the TCP store
//! server; every rank writes its contribution as a key and reads its peers'
//! keys). This is O(N) traffic at the master — correct and simple, appropriate
//! for single-node multi-process training. It is *not* a bandwidth-optimal ring;
//! the "PyTorch-Gloo-style" reference is about the store-based **rendezvous**,
//! not the data plane.
//!
//! # Supported element types
//!
//! `f32` and `f64` tensors, with reduce ops `Sum`, `Product`, `Min`, `Max` and
//! `Mean`. Bitwise reductions (`Band`/`Bor`/`Bxor`) are rejected for floating
//! point data. Other element types produce an honest error.
//!
//! # SPMD ordering contract
//!
//! Collectives are matched across ranks by a per-engine monotonic sequence
//! number namespaced by the operation name (e.g. `torsh/all_reduce/7`). All
//! ranks MUST issue the same sequence of collectives in the same order (the
//! standard SPMD assumption). A desynchronised rank waits on a key that never
//! appears and fails with an `operation_timeout` naming the missing key, rather
//! than hanging forever.
//!
//! # Lazy rendezvous
//!
//! [`TcpBackend::init`] is cheap and never touches the network, so constructing
//! a process group (even with `world_size > 1`) in a single process succeeds.
//! The master server is started lazily on the first `world_size > 1` collective.
//! `world_size == 1` collectives short-circuit locally and never open a socket.

use crate::backend::{
    Backend, BackendCapabilities, BackendConfig, BackendStatus, BackendType, ReduceOp,
};
use crate::store::Store;
use crate::store::TcpStore;
use crate::{TorshDistributedError, TorshResult};
use async_trait::async_trait;
use dashmap::DashMap;
use std::any::Any;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;

const DTYPE_F32: u8 = 0;
const DTYPE_F64: u8 = 1;

/// A decoded numeric payload exchanged between ranks.
///
/// Only `f32` and `f64` are supported; these are the element types with a
/// well-defined wire format and reduction semantics for this backend.
#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    /// 32-bit floating point buffer.
    F32(Vec<f32>),
    /// 64-bit floating point buffer.
    F64(Vec<f64>),
}

impl Payload {
    fn len(&self) -> usize {
        match self {
            Payload::F32(v) => v.len(),
            Payload::F64(v) => v.len(),
        }
    }

    fn dtype_name(&self) -> &'static str {
        match self {
            Payload::F32(_) => "f32",
            Payload::F64(_) => "f64",
        }
    }

    /// Encode as `[dtype:u8][count:u32-le][elements-le...]`.
    fn encode(&self) -> Vec<u8> {
        match self {
            Payload::F32(v) => {
                let mut out = Vec::with_capacity(5 + v.len() * 4);
                out.push(DTYPE_F32);
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
                out
            }
            Payload::F64(v) => {
                let mut out = Vec::with_capacity(5 + v.len() * 8);
                out.push(DTYPE_F64);
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
                out
            }
        }
    }

    /// Move the inner buffer into a type-erased box (`Box<Vec<f32>>` /
    /// `Box<Vec<f64>>`) for downcasting by the caller.
    pub fn into_box(self) -> Box<dyn Any + Send> {
        match self {
            Payload::F32(v) => Box::new(v),
            Payload::F64(v) => Box::new(v),
        }
    }

    /// Decode from the wire format produced by [`Payload::encode`].
    pub fn decode(bytes: &[u8]) -> TorshResult<Payload> {
        if bytes.len() < 5 {
            return Err(TorshDistributedError::communication_error(
                "decode",
                "payload too short",
            ));
        }
        let dtype = bytes[0];
        let mut count_buf = [0u8; 4];
        count_buf.copy_from_slice(&bytes[1..5]);
        let count = u32::from_le_bytes(count_buf) as usize;
        let body = &bytes[5..];
        match dtype {
            DTYPE_F32 => {
                if body.len() != count * 4 {
                    return Err(TorshDistributedError::communication_error(
                        "decode",
                        "f32 payload length mismatch",
                    ));
                }
                let mut v = Vec::with_capacity(count);
                for chunk in body.chunks_exact(4) {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(chunk);
                    v.push(f32::from_le_bytes(b));
                }
                Ok(Payload::F32(v))
            }
            DTYPE_F64 => {
                if body.len() != count * 8 {
                    return Err(TorshDistributedError::communication_error(
                        "decode",
                        "f64 payload length mismatch",
                    ));
                }
                let mut v = Vec::with_capacity(count);
                for chunk in body.chunks_exact(8) {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(chunk);
                    v.push(f64::from_le_bytes(b));
                }
                Ok(Payload::F64(v))
            }
            other => Err(TorshDistributedError::communication_error(
                "decode",
                format!("unknown dtype tag {}", other),
            )),
        }
    }

    /// Fold `other` into `self` element-wise using `op` (accumulation step).
    fn reduce_with(&mut self, other: &Payload, op: ReduceOp) -> TorshResult<()> {
        if self.len() != other.len() {
            return Err(TorshDistributedError::tensor_shape_mismatch(
                vec![self.len()],
                vec![other.len()],
            ));
        }
        match (self, other) {
            (Payload::F32(a), Payload::F32(b)) => reduce_slice_f32(a, b, op),
            (Payload::F64(a), Payload::F64(b)) => reduce_slice_f64(a, b, op),
            (a, b) => Err(TorshDistributedError::invalid_argument(
                "dtype",
                format!(
                    "mixed dtypes in reduction: {} vs {}",
                    a.dtype_name(),
                    b.dtype_name()
                ),
                "all ranks must contribute the same element type",
            )),
        }
    }

    /// Divide every element by `n` (used to turn a `Sum` into a `Mean`).
    fn scale_mean(&mut self, n: usize) {
        if n <= 1 {
            return;
        }
        match self {
            Payload::F32(v) => {
                let d = n as f32;
                for x in v.iter_mut() {
                    *x /= d;
                }
            }
            Payload::F64(v) => {
                let d = n as f64;
                for x in v.iter_mut() {
                    *x /= d;
                }
            }
        }
    }
}

fn bitwise_error(op: ReduceOp) -> TorshDistributedError {
    TorshDistributedError::invalid_argument(
        "op",
        format!(
            "bitwise reduction {:?} is not defined for floating-point tensors",
            op
        ),
        "Sum, Product, Min, Max or Mean",
    )
}

fn reduce_slice_f32(a: &mut [f32], b: &[f32], op: ReduceOp) -> TorshResult<()> {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        ReduceOp::Product => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x *= y;
            }
        }
        ReduceOp::Min => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x = x.min(y);
            }
        }
        ReduceOp::Max => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x = x.max(y);
            }
        }
        ReduceOp::Band | ReduceOp::Bor | ReduceOp::Bxor => return Err(bitwise_error(op)),
    }
    Ok(())
}

fn reduce_slice_f64(a: &mut [f64], b: &[f64], op: ReduceOp) -> TorshResult<()> {
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        ReduceOp::Product => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x *= y;
            }
        }
        ReduceOp::Min => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x = x.min(y);
            }
        }
        ReduceOp::Max => {
            for (x, &y) in a.iter_mut().zip(b) {
                *x = x.max(y);
            }
        }
        ReduceOp::Band | ReduceOp::Bor | ReduceOp::Bxor => return Err(bitwise_error(op)),
    }
    Ok(())
}

/// Extract a [`Payload`] from a type-erased tensor buffer (`Vec<f32>`/`Vec<f64>`).
fn read_any(any: &(dyn Any + Send + Sync)) -> TorshResult<Payload> {
    if let Some(v) = any.downcast_ref::<Vec<f32>>() {
        Ok(Payload::F32(v.clone()))
    } else if let Some(v) = any.downcast_ref::<Vec<f64>>() {
        Ok(Payload::F64(v.clone()))
    } else {
        Err(TorshDistributedError::invalid_argument(
            "tensor",
            "unsupported element type for the TCP collective backend",
            "Vec<f32> or Vec<f64>",
        ))
    }
}

/// Write a [`Payload`] back into a type-erased tensor buffer.
fn write_any(any: &mut (dyn Any + Send + Sync), payload: &Payload) -> TorshResult<()> {
    match payload {
        Payload::F32(src) => {
            if let Some(v) = any.downcast_mut::<Vec<f32>>() {
                *v = src.clone();
                Ok(())
            } else {
                Err(TorshDistributedError::invalid_argument(
                    "tensor",
                    "received f32 payload but destination buffer is not Vec<f32>",
                    "matching element type across ranks",
                ))
            }
        }
        Payload::F64(src) => {
            if let Some(v) = any.downcast_mut::<Vec<f64>>() {
                *v = src.clone();
                Ok(())
            } else {
                Err(TorshDistributedError::invalid_argument(
                    "tensor",
                    "received f64 payload but destination buffer is not Vec<f64>",
                    "matching element type across ranks",
                ))
            }
        }
    }
}

/// Encode a type-erased tensor buffer (`Vec<f32>`/`Vec<f64>`) to wire bytes.
///
/// Used by the collectives layer (e.g. `scatter`) to serialise per-rank chunks.
pub fn encode_any(data: &(dyn Any + Send + Sync)) -> TorshResult<Vec<u8>> {
    Ok(read_any(data)?.encode())
}

fn resolve_addr(master_addr: &str) -> IpAddr {
    if let Ok(ip) = master_addr.parse::<IpAddr>() {
        return ip;
    }
    // This backend targets single-node (loopback) multi-process rendezvous;
    // any non-literal host resolves to loopback. For real multi-node use, pass
    // a literal IP address as MASTER_ADDR.
    IpAddr::V4(Ipv4Addr::LOCALHOST)
}

/// Lock-free, `Send + Sync` collective engine shared behind an `Arc`.
///
/// All network state lives here (an `Arc<TcpStore>` and an atomic sequence
/// counter), so the engine can be cloned out from under a backend lock and
/// awaited without holding any guard across `.await` — a hard requirement for
/// the DDP background-sync path which `tokio::spawn`s gradient all-reduces.
pub struct TcpEngine {
    rank: u32,
    world_size: u32,
    master_addr: IpAddr,
    master_port: u16,
    timeout: Duration,
    /// Per-namespace monotonic sequence counters. The empty namespace is the
    /// global process group; sub-group collectives use the group id so that
    /// ranks not participating in a sub-group do not desync the global counter.
    seqs: DashMap<String, AtomicU64>,
    store: OnceCell<Arc<TcpStore>>,
}

impl std::fmt::Debug for TcpEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpEngine")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .field("master_addr", &self.master_addr)
            .field("master_port", &self.master_port)
            .finish()
    }
}

impl TcpEngine {
    fn new(
        rank: u32,
        world_size: u32,
        master_addr: IpAddr,
        master_port: u16,
        timeout: Duration,
    ) -> Self {
        Self {
            rank,
            world_size,
            master_addr,
            master_port,
            timeout,
            seqs: DashMap::new(),
            store: OnceCell::new(),
        }
    }

    /// Lazily start (rank 0) / connect (others) to the master TCP store.
    async fn ensure_store(&self) -> TorshResult<Arc<TcpStore>> {
        let store = self
            .store
            .get_or_try_init(|| async {
                let mut store = TcpStore::new(
                    self.master_addr,
                    self.master_port,
                    self.timeout,
                    self.rank == 0,
                )?;
                if self.rank == 0 {
                    store.start().await?;
                }
                let store = Arc::new(store);

                // Wait until the master server answers, retrying transient
                // connection failures (peers may start in any order).
                let start = Instant::now();
                loop {
                    match store.contains("__torsh_rendezvous_ping__").await {
                        Ok(_) => break,
                        Err(_) => {
                            if start.elapsed() > self.timeout {
                                return Err(TorshDistributedError::operation_timeout(
                                    format!(
                                        "connect to master {}:{}",
                                        self.master_addr, self.master_port
                                    ),
                                    self.timeout.as_secs(),
                                ));
                            }
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                    }
                }
                Ok(store)
            })
            .await?;
        Ok(store.clone())
    }

    /// Allocate the next lockstep tag for operation `op` in namespace `ns`
    /// (`""` for the global group, or a sub-group id).
    fn next_tag(&self, ns: &str, op: &str) -> String {
        let seq = self
            .seqs
            .entry(ns.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::SeqCst);
        format!("torsh/{}/{}/{}", ns, op, seq)
    }

    async fn store_set(&self, store: &TcpStore, key: &str, value: &[u8]) -> TorshResult<()> {
        let start = Instant::now();
        loop {
            match store.set(key, value).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if start.elapsed() > self.timeout {
                        return Err(e);
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        }
    }

    /// Poll `key` until it is present (or the deadline elapses).
    async fn poll_get(&self, store: &TcpStore, key: &str) -> TorshResult<Vec<u8>> {
        let start = Instant::now();
        let mut interval = Duration::from_millis(2);
        loop {
            if let Ok(Some(v)) = store.get(key).await {
                return Ok(v);
            }
            if start.elapsed() > self.timeout {
                return Err(TorshDistributedError::operation_timeout(
                    format!("rendezvous waiting for key '{}'", key),
                    self.timeout.as_secs(),
                ));
            }
            tokio::time::sleep(interval).await;
            interval = (interval * 2).min(Duration::from_millis(50));
        }
    }

    /// Post `payload` under `tag/r{rank}` and gather every participant's payload
    /// (returned in `participants` order).
    async fn exchange(
        &self,
        tag: &str,
        participants: &[u32],
        payload: &[u8],
    ) -> TorshResult<Vec<Vec<u8>>> {
        let store = self.ensure_store().await?;
        let my_key = format!("{}/r{}", tag, self.rank);
        self.store_set(&store, &my_key, payload).await?;

        let mut gathered = Vec::with_capacity(participants.len());
        for &p in participants {
            if p == self.rank {
                gathered.push(payload.to_vec());
            } else {
                let key = format!("{}/r{}", tag, p);
                gathered.push(self.poll_get(&store, &key).await?);
            }
        }
        Ok(gathered)
    }

    /// Real all-reduce over `participants` (must be sorted, contain `self.rank`).
    ///
    /// `ns` namespaces the lockstep sequence (`""` for the global group).
    pub async fn all_reduce_any(
        &self,
        data: &mut (dyn Any + Send + Sync),
        op: ReduceOp,
        participants: &[u32],
        ns: &str,
    ) -> TorshResult<()> {
        let mine = read_any(data)?;
        // Reject bitwise ops before doing any network I/O.
        if matches!(op, ReduceOp::Band | ReduceOp::Bor | ReduceOp::Bxor) {
            return Err(bitwise_error(op));
        }
        let encoded = mine.encode();
        let tag = self.next_tag(ns, "all_reduce");
        let all = self.exchange(&tag, participants, &encoded).await?;

        let mut acc: Option<Payload> = None;
        for bytes in &all {
            let p = Payload::decode(bytes)?;
            match acc.as_mut() {
                None => acc = Some(p),
                Some(a) => a.reduce_with(&p, op)?,
            }
        }
        let mut acc = acc.ok_or_else(|| {
            TorshDistributedError::internal_error("all_reduce produced no contributions")
        })?;
        if matches!(op, ReduceOp::Mean) {
            acc.scale_mean(participants.len());
        }
        write_any(data, &acc)
    }

    /// Real broadcast of the root's buffer to every participant.
    pub async fn broadcast_any(
        &self,
        data: &mut (dyn Any + Send + Sync),
        root: u32,
        _participants: &[u32],
        ns: &str,
    ) -> TorshResult<()> {
        let store = self.ensure_store().await?;
        let tag = self.next_tag(ns, "broadcast");
        let key = format!("{}/root{}", tag, root);
        if self.rank == root {
            let payload = read_any(data)?.encode();
            self.store_set(&store, &key, &payload).await?;
        } else {
            let bytes = self.poll_get(&store, &key).await?;
            let payload = Payload::decode(&bytes)?;
            write_any(data, &payload)?;
        }
        Ok(())
    }

    /// Real all-gather: returns one [`Payload`] per participant (in order).
    pub async fn all_gather_payloads(
        &self,
        data: &(dyn Any + Send + Sync),
        participants: &[u32],
        ns: &str,
    ) -> TorshResult<Vec<Payload>> {
        let mine = read_any(data)?.encode();
        let tag = self.next_tag(ns, "all_gather");
        let all = self.exchange(&tag, participants, &mine).await?;
        all.iter().map(|b| Payload::decode(b)).collect()
    }

    /// Real reduce: only the destination rank receives the reduced buffer;
    /// other ranks leave their buffer untouched.
    pub async fn reduce_any(
        &self,
        data: &mut (dyn Any + Send + Sync),
        dst: u32,
        op: ReduceOp,
        participants: &[u32],
        ns: &str,
    ) -> TorshResult<()> {
        if matches!(op, ReduceOp::Band | ReduceOp::Bor | ReduceOp::Bxor) {
            return Err(bitwise_error(op));
        }
        let encoded = read_any(data)?.encode();
        let tag = self.next_tag(ns, "reduce");
        let all = self.exchange(&tag, participants, &encoded).await?;
        if self.rank != dst {
            return Ok(());
        }
        let mut acc: Option<Payload> = None;
        for bytes in &all {
            let p = Payload::decode(bytes)?;
            match acc.as_mut() {
                None => acc = Some(p),
                Some(a) => a.reduce_with(&p, op)?,
            }
        }
        let mut acc = acc.ok_or_else(|| {
            TorshDistributedError::internal_error("reduce produced no contributions")
        })?;
        if matches!(op, ReduceOp::Mean) {
            acc.scale_mean(participants.len());
        }
        write_any(data, &acc)
    }

    /// Real scatter: `src` posts one chunk per participant; each rank reads its
    /// own chunk. `chunks` must be `Some` on `src` with one payload per rank.
    pub async fn scatter_bytes(
        &self,
        chunks: Option<&[Vec<u8>]>,
        src: u32,
        participants: &[u32],
        ns: &str,
    ) -> TorshResult<Vec<u8>> {
        let store = self.ensure_store().await?;
        let tag = self.next_tag(ns, "scatter");
        if self.rank == src {
            let chunks = chunks.ok_or_else(|| {
                TorshDistributedError::invalid_argument(
                    "chunks",
                    "source rank must provide one chunk per participant",
                    "Some(slice of length world_size)",
                )
            })?;
            if chunks.len() != participants.len() {
                return Err(TorshDistributedError::invalid_argument(
                    "chunks",
                    format!(
                        "expected {} chunks, got {}",
                        participants.len(),
                        chunks.len()
                    ),
                    "one chunk per participant",
                ));
            }
            for (&p, chunk) in participants.iter().zip(chunks) {
                let key = format!("{}/r{}", tag, p);
                self.store_set(&store, &key, chunk).await?;
            }
        }
        let my_key = format!("{}/r{}", tag, self.rank);
        self.poll_get(&store, &my_key).await
    }

    /// Real all-to-all: `sends[k]` is delivered to `participants[k]`; returns
    /// the buffer received from each participant (in `participants` order).
    pub async fn all_to_all_bytes(
        &self,
        sends: &[Vec<u8>],
        participants: &[u32],
        ns: &str,
    ) -> TorshResult<Vec<Vec<u8>>> {
        if sends.len() != participants.len() {
            return Err(TorshDistributedError::invalid_argument(
                "sends",
                format!(
                    "expected {} send buffers, got {}",
                    participants.len(),
                    sends.len()
                ),
                "one send buffer per participant",
            ));
        }
        let store = self.ensure_store().await?;
        let tag = self.next_tag(ns, "all_to_all");
        // Post one directed message per destination.
        for (&dst, buf) in participants.iter().zip(sends) {
            let key = format!("{}/{}_to_{}", tag, self.rank, dst);
            self.store_set(&store, &key, buf).await?;
        }
        // Collect one directed message from each source.
        let mut received = Vec::with_capacity(participants.len());
        for &src in participants {
            if src == self.rank {
                // Message to self: read what we just posted.
                let idx = participants.iter().position(|&p| p == self.rank);
                match idx {
                    Some(i) => received.push(sends[i].clone()),
                    None => {
                        return Err(TorshDistributedError::internal_error(
                            "self rank missing from participants in all_to_all",
                        ))
                    }
                }
            } else {
                let key = format!("{}/{}_to_{}", tag, src, self.rank);
                received.push(self.poll_get(&store, &key).await?);
            }
        }
        Ok(received)
    }

    /// Real point-to-point send: post the buffer for `dst` to consume.
    pub async fn send_any(
        &self,
        data: &(dyn Any + Send + Sync),
        dst: u32,
        tag: u32,
    ) -> TorshResult<()> {
        let store = self.ensure_store().await?;
        let payload = read_any(data)?.encode();
        let key = format!("torsh/p2p/{}_{}_{}", self.rank, dst, tag);
        self.store_set(&store, &key, &payload).await
    }

    /// Real point-to-point receive: wait for `src`'s buffer, then consume it.
    pub async fn recv_bytes(&self, src: u32, tag: u32) -> TorshResult<Payload> {
        let store = self.ensure_store().await?;
        let key = format!("torsh/p2p/{}_{}_{}", src, self.rank, tag);
        let bytes = self.poll_get(&store, &key).await?;
        // Consume the message so the same (src,dst,tag) can be reused.
        let _ = store.delete(&key).await;
        Payload::decode(&bytes)
    }

    /// Real barrier: every participant posts a token and waits for all others.
    pub async fn barrier(&self, participants: &[u32], ns: &str) -> TorshResult<()> {
        let tag = self.next_tag(ns, "barrier");
        self.exchange(&tag, participants, &[1u8]).await.map(|_| ())
    }
}

/// Real TCP collective backend (the default `Gloo`-equivalent transport).
///
/// See the [module documentation](self) for topology, supported types and the
/// SPMD ordering contract.
pub struct TcpBackend {
    engine: Arc<TcpEngine>,
    initialized: bool,
}

impl std::fmt::Debug for TcpBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpBackend")
            .field("engine", &self.engine)
            .field("initialized", &self.initialized)
            .finish()
    }
}

impl TcpBackend {
    /// Create a new TCP backend. Construction never touches the network; the
    /// master store is started lazily on the first `world_size > 1` collective.
    pub fn new(rank: u32, world_size: u32, master_addr: &str, master_port: u16) -> Self {
        let addr = resolve_addr(master_addr);
        Self {
            engine: Arc::new(TcpEngine::new(
                rank,
                world_size,
                addr,
                master_port,
                Duration::from_secs(30),
            )),
            initialized: false,
        }
    }

    /// Clone the shared collective engine (used by the collectives layer to run
    /// real communication without holding the backend lock across `.await`).
    pub fn engine(&self) -> Arc<TcpEngine> {
        Arc::clone(&self.engine)
    }

    fn participants(&self) -> Vec<u32> {
        (0..self.engine.world_size).collect()
    }
}

#[async_trait]
impl Backend for TcpBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Gloo
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            async_operations: true,
            gpu_support: false,
            p2p_communication: true,
            custom_reduce_ops: false,
            max_tensor_size: None,
            supported_dtypes: vec!["f32".to_string(), "f64".to_string()],
        }
    }

    async fn init(&mut self, _config: BackendConfig) -> TorshResult<()> {
        // Cheap, network-free initialisation: only mark the backend ready. The
        // master server / client connection is established lazily on the first
        // real collective so that constructing a process group never binds a
        // socket (required by the single-process construction tests).
        //
        // We deliberately do NOT rebuild the engine here: an `init -> collective
        // -> cleanup -> init` sequence would otherwise orphan a running store
        // server and reset the lockstep sequence counters (silently desyncing
        // peers). The engine's rendezvous timeout is fixed at construction and
        // matches `BackendConfig::default().timeout`.
        self.initialized = true;
        Ok(())
    }

    async fn cleanup(&mut self) -> TorshResult<()> {
        self.initialized = false;
        Ok(())
    }

    fn status(&self) -> BackendStatus {
        BackendStatus {
            initialized: self.initialized,
            healthy: true,
            active_operations: 0,
            total_operations: 0,
            failed_operations: 0,
            last_error: None,
        }
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn rank(&self) -> u32 {
        self.engine.rank
    }

    fn world_size(&self) -> u32 {
        self.engine.world_size
    }

    async fn barrier(&mut self) -> TorshResult<()> {
        if self.engine.world_size <= 1 {
            return Ok(());
        }
        let participants = self.participants();
        self.engine.barrier(&participants, "").await
    }

    async fn all_reduce(
        &mut self,
        tensor: &mut (dyn Any + Send + Sync),
        op: ReduceOp,
    ) -> TorshResult<()> {
        if self.engine.world_size <= 1 {
            // Single-rank reduction is the identity for every supported op.
            return Ok(());
        }
        let participants = self.participants();
        self.engine
            .all_reduce_any(tensor, op, &participants, "")
            .await
    }

    async fn all_gather(
        &mut self,
        tensor: &(dyn Any + Send + Sync),
    ) -> TorshResult<Box<dyn Any + Send>> {
        let participants = self.participants();
        if self.engine.world_size <= 1 {
            // Single rank: the gather is just this rank's own buffer.
            let mine = read_any(tensor)?;
            return Ok(mine.into_box());
        }
        let payloads = self
            .engine
            .all_gather_payloads(tensor, &participants, "")
            .await?;
        // Concatenate into a single contiguous buffer (rank-major order).
        concat_payloads(payloads)
    }

    async fn broadcast(
        &mut self,
        tensor: &mut (dyn Any + Send + Sync),
        root: u32,
    ) -> TorshResult<()> {
        if root >= self.engine.world_size {
            return Err(TorshDistributedError::RankOutOfBounds {
                rank: root,
                world_size: self.engine.world_size,
            });
        }
        if self.engine.world_size <= 1 {
            return Ok(());
        }
        let participants = self.participants();
        self.engine
            .broadcast_any(tensor, root, &participants, "")
            .await
    }

    async fn send(
        &mut self,
        tensor: &(dyn Any + Send + Sync),
        dst: u32,
        tag: u32,
    ) -> TorshResult<()> {
        if dst >= self.engine.world_size {
            return Err(TorshDistributedError::RankOutOfBounds {
                rank: dst,
                world_size: self.engine.world_size,
            });
        }
        self.engine.send_any(tensor, dst, tag).await
    }

    async fn recv(&mut self, src: u32, tag: u32) -> TorshResult<Box<dyn Any + Send>> {
        if src >= self.engine.world_size {
            return Err(TorshDistributedError::RankOutOfBounds {
                rank: src,
                world_size: self.engine.world_size,
            });
        }
        let payload = self.engine.recv_bytes(src, tag).await?;
        Ok(payload.into_box())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn concat_payloads(payloads: Vec<Payload>) -> TorshResult<Box<dyn Any + Send>> {
    match payloads.first() {
        Some(Payload::F32(_)) => {
            let mut out = Vec::new();
            for p in payloads {
                match p {
                    Payload::F32(v) => out.extend(v),
                    Payload::F64(_) => {
                        return Err(TorshDistributedError::invalid_argument(
                            "dtype",
                            "mixed dtypes across ranks in all_gather",
                            "consistent element type",
                        ))
                    }
                }
            }
            Ok(Box::new(out))
        }
        Some(Payload::F64(_)) => {
            let mut out = Vec::new();
            for p in payloads {
                match p {
                    Payload::F64(v) => out.extend(v),
                    Payload::F32(_) => {
                        return Err(TorshDistributedError::invalid_argument(
                            "dtype",
                            "mixed dtypes across ranks in all_gather",
                            "consistent element type",
                        ))
                    }
                }
            }
            Ok(Box::new(out))
        }
        None => Ok(Box::new(Vec::<f32>::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_roundtrip_f32() {
        let p = Payload::F32(vec![1.0, -2.5, 3.25]);
        let bytes = p.encode();
        let decoded = Payload::decode(&bytes).expect("decode");
        assert_eq!(p, decoded);
    }

    #[test]
    fn test_payload_roundtrip_f64() {
        let p = Payload::F64(vec![1.0, -2.5, 3.25, 1e10]);
        let bytes = p.encode();
        let decoded = Payload::decode(&bytes).expect("decode");
        assert_eq!(p, decoded);
    }

    #[test]
    fn test_reduce_sum_f32() {
        let mut a = Payload::F32(vec![1.0, 2.0, 3.0]);
        let b = Payload::F32(vec![4.0, 5.0, 6.0]);
        a.reduce_with(&b, ReduceOp::Sum).expect("reduce");
        assert_eq!(a, Payload::F32(vec![5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_reduce_bitwise_rejected() {
        let mut a = Payload::F32(vec![1.0]);
        let b = Payload::F32(vec![2.0]);
        assert!(a.reduce_with(&b, ReduceOp::Band).is_err());
    }

    #[test]
    fn test_read_any_unsupported_type() {
        let v: Vec<i32> = vec![1, 2, 3];
        let any: &(dyn Any + Send + Sync) = &v;
        assert!(read_any(any).is_err());
    }
}
