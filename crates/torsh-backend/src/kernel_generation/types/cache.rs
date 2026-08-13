//! Kernel cache for storing compiled kernels.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::sync::MutexExt;

use super::common_types::{CacheStatistics, GeneratedKernel};

/// Kernel cache for storing compiled kernels
pub struct KernelCache {
    cache: Arc<Mutex<HashMap<String, GeneratedKernel>>>,
    max_size: usize,
    hit_count: Arc<Mutex<u64>>,
    miss_count: Arc<Mutex<u64>>,
}

impl KernelCache {
    /// Create a new kernel cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a kernel from cache
    pub fn get(&self, key: &str) -> Option<GeneratedKernel> {
        let cache = self.cache.lock_or_recover();
        if let Some(kernel) = cache.get(key) {
            *self.hit_count.lock_or_recover() += 1;
            Some(kernel.clone())
        } else {
            *self.miss_count.lock_or_recover() += 1;
            None
        }
    }

    /// Insert a kernel into cache
    pub fn insert(&self, key: String, kernel: GeneratedKernel) {
        let mut cache = self.cache.lock_or_recover();
        if cache.len() >= self.max_size {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
        cache.insert(key, kernel);
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        let hits = *self.hit_count.lock_or_recover();
        let misses = *self.miss_count.lock_or_recover();
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        CacheStatistics {
            hits,
            misses,
            total_requests: total,
            hit_rate,
            cache_size: self.cache.lock_or_recover().len(),
            max_cache_size: self.max_size,
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.lock_or_recover().clear();
        *self.hit_count.lock_or_recover() = 0;
        *self.miss_count.lock_or_recover() = 0;
    }
}
