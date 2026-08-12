//! Dimension-aware tensor operations (reductions, scans, sorting, arg-reductions)
//!
//! Every operation in this module follows PyTorch semantics:
//!
//! - Negative dimension indices count from the end (`-1` is the last axis).
//! - Reductions remove the reduced axis unless `keepdim` is requested, in which
//!   case the axis is kept with extent `1`.
//! - Scans (`cumsum`, `cumprod`) walk *along* the requested axis and restart the
//!   accumulator for every independent slice.
//! - `sort` returns `(values, indices)` where the indices are positions **inside
//!   the sorted dimension**, typed `i64` like `argmin`/`argmax`.
//!
//! The kernels share a single `outer / dim / inner` decomposition: for an axis
//! `d` of a contiguous tensor, element `(o, i, n)` lives at flat index
//! `o * dim_size * inner_size + i * inner_size + n`.  This makes every op here
//! stride-correct for arbitrary rank without materialising transposes.

use torsh_core::{
    dtype::{FloatElement, TensorElement},
    error::{Result, TorshError},
};

use crate::core_ops::Tensor;

/// Normalise a (possibly negative) dimension index against a tensor rank.
///
/// Returns `Err(TorshError::InvalidOperation)` when the index is out of range,
/// including every index for a 0-dimensional tensor.
pub(crate) fn normalize_dim(dim: i32, ndim: usize) -> Result<usize> {
    let resolved = if dim < 0 {
        i64::from(dim) + ndim as i64
    } else {
        i64::from(dim)
    };

    if resolved < 0 || resolved as usize >= ndim {
        return Err(TorshError::InvalidOperation(format!(
            "Dimension {} out of range for {}-dimensional tensor",
            dim, ndim
        )));
    }

    Ok(resolved as usize)
}

/// Split a shape into the `(outer, dim, inner)` extents around `dim`.
pub(crate) fn split_extents(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, dim_size, inner)
}

/// Shape of a reduction result over a single axis.
pub(crate) fn reduced_shape(shape: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    let mut out = shape.to_vec();
    if keepdim {
        out[dim] = 1;
    } else {
        out.remove(dim);
    }
    out
}

/// Reduce `data` along the `(outer, dim, inner)` decomposition, keeping the
/// element selected by `prefer_new` (e.g. `|new, best| new > best` for max).
fn extremum_values<T: Copy>(
    data: &[T],
    outer: usize,
    dim_size: usize,
    inner: usize,
    prefer_new: impl Fn(T, T) -> bool,
) -> Vec<T> {
    let mut result = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for n in 0..inner {
            let base = o * dim_size * inner + n;
            let mut best = data[base];
            for d in 1..dim_size {
                let value = data[base + d * inner];
                if prefer_new(value, best) {
                    best = value;
                }
            }
            result.push(best);
        }
    }
    result
}

/// Index of the extremum along the `(outer, dim, inner)` decomposition.
/// Ties resolve to the first occurrence, matching PyTorch.
fn extremum_indices<T: Copy>(
    data: &[T],
    outer: usize,
    dim_size: usize,
    inner: usize,
    prefer_new: impl Fn(T, T) -> bool,
) -> Vec<i64> {
    let mut result = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for n in 0..inner {
            let base = o * dim_size * inner + n;
            let mut best = data[base];
            let mut best_idx = 0i64;
            for d in 1..dim_size {
                let value = data[base + d * inner];
                if prefer_new(value, best) {
                    best = value;
                    best_idx = d as i64;
                }
            }
            result.push(best_idx);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Float-only extremum reductions
// ---------------------------------------------------------------------------

impl<T: FloatElement + Copy> Tensor<T> {
    /// Maximum element of the tensor, optionally reduced along a single axis.
    ///
    /// `dim == None` reduces every element to a scalar (or to an all-ones shape
    /// when `keepdim` is set).
    pub fn max(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        match dim {
            None => {
                let max_val =
                    self.with_contiguous_data(|data| global_extremum(data, |a, b| a > b, "max"))?;
                if keepdim {
                    let shape = vec![1; self.shape().dims().len()];
                    Self::from_data(vec![max_val], shape, self.device())
                } else {
                    Self::scalar(max_val)
                }
            }
            Some(axis) => self.extremum_along_dim(axis, keepdim, true),
        }
    }

    /// Maximum along a specified dimension (negative indices allowed).
    pub fn max_dim(&self, dim: i32, keepdim: bool) -> Result<Self> {
        let axis = normalize_dim(dim, self.shape().dims().len())?;
        self.extremum_along_dim(axis, keepdim, true)
    }

    /// Minimum along a specified dimension (negative indices allowed).
    pub fn min_dim(&self, dim: i32, keepdim: bool) -> Result<Self> {
        let axis = normalize_dim(dim, self.shape().dims().len())?;
        self.extremum_along_dim(axis, keepdim, false)
    }

    /// Minimum element of the tensor, optionally reduced along a single axis.
    ///
    /// Mirror image of [`Tensor::max`]; PyTorch calls this reduction `amin`.
    /// The parameter-less [`Tensor::min`] remains available for global minima.
    pub fn amin(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        match dim {
            None => {
                let min_val =
                    self.with_contiguous_data(|data| global_extremum(data, |a, b| a < b, "min"))?;
                if keepdim {
                    let shape = vec![1; self.shape().dims().len()];
                    Self::from_data(vec![min_val], shape, self.device())
                } else {
                    Self::scalar(min_val)
                }
            }
            Some(axis) => self.extremum_along_dim(axis, keepdim, false),
        }
    }

    /// Maximum element of the tensor, optionally reduced along a single axis.
    ///
    /// PyTorch-compatible alias of [`Tensor::max`], provided so that generic
    /// code can use the symmetric `amin`/`amax` pair.
    pub fn amax(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        self.max(dim, keepdim)
    }

    /// Shared implementation of the dimension-wise extremum reductions.
    fn extremum_along_dim(&self, axis: usize, keepdim: bool, want_max: bool) -> Result<Self> {
        let shape_binding = self.shape();
        let input_shape = shape_binding.dims();

        if axis >= input_shape.len() {
            return Err(TorshError::InvalidOperation(format!(
                "Axis {} out of bounds for {}-dimensional tensor",
                axis,
                input_shape.len()
            )));
        }

        let (outer, dim_size, inner) = split_extents(input_shape, axis);
        if dim_size == 0 {
            return Err(TorshError::InvalidOperation(
                "Cannot reduce a dimension with zero size".to_string(),
            ));
        }

        let output_shape = reduced_shape(input_shape, axis, keepdim);
        let result_data = self.with_contiguous_data(|data| {
            Ok(if want_max {
                extremum_values(data, outer, dim_size, inner, |new, best| new > best)
            } else {
                extremum_values(data, outer, dim_size, inner, |new, best| new < best)
            })
        })?;

        Self::from_data(result_data, output_shape, self.device())
    }
}

/// Fold a slice to its extremum, returning an error for empty input.
fn global_extremum<T: Copy>(data: &[T], prefer_new: impl Fn(T, T) -> bool, op: &str) -> Result<T> {
    let mut iter = data.iter().copied();
    let first = iter.next().ok_or_else(|| {
        TorshError::InvalidOperation(format!("Cannot compute {} of empty tensor", op))
    })?;
    Ok(iter.fold(first, |best, x| if prefer_new(x, best) { x } else { best }))
}

// ---------------------------------------------------------------------------
// Generic dimension-aware operations
// ---------------------------------------------------------------------------

impl<T: TensorElement + Copy> Tensor<T> {
    /// Compute the sum along the specified dimensions.
    ///
    /// Every listed dimension is reduced (duplicates and negative indices are
    /// accepted); with `keepdim` the reduced axes are kept with extent `1`.
    /// An empty `dims` slice reduces the whole tensor, matching [`Tensor::sum`].
    pub fn sum_dim(&self, dims: &[i32], keepdim: bool) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        if dims.is_empty() {
            return self.sum();
        }

        let shape_binding = self.shape();
        let input_shape = shape_binding.dims().to_vec();
        let ndim = input_shape.len();

        let mut reduce_dims = Vec::with_capacity(dims.len());
        for &dim in dims {
            reduce_dims.push(normalize_dim(dim, ndim)?);
        }
        reduce_dims.sort_unstable();
        reduce_dims.dedup();

        // Single-axis fast path: contiguous outer/dim/inner walk.
        if reduce_dims.len() == 1 {
            let axis = reduce_dims[0];
            let (outer, dim_size, inner) = split_extents(&input_shape, axis);
            let output_shape = reduced_shape(&input_shape, axis, keepdim);

            let result_data = self.with_contiguous_data(|data| {
                let mut result = vec![<T as num_traits::Zero>::zero(); outer * inner];
                for o in 0..outer {
                    for n in 0..inner {
                        let base = o * dim_size * inner + n;
                        let mut acc = <T as num_traits::Zero>::zero();
                        for d in 0..dim_size {
                            acc = acc + data[base + d * inner];
                        }
                        result[o * inner + n] = acc;
                    }
                }
                Ok(result)
            })?;

            return Self::from_data(result_data, output_shape, self.device());
        }

        // General multi-axis reduction: map every input element onto the flat
        // index of the (keepdim-shaped) output slot it accumulates into.
        let mut is_reduced = vec![false; ndim];
        for &d in &reduce_dims {
            is_reduced[d] = true;
        }

        let input_strides = contiguous_strides(&input_shape);
        let mut output_shape_keepdim = input_shape.clone();
        for &d in &reduce_dims {
            output_shape_keepdim[d] = 1;
        }
        let output_strides = contiguous_strides(&output_shape_keepdim);
        let output_size: usize = output_shape_keepdim.iter().product();

        let result_data = self.with_contiguous_data(|data| {
            let mut acc = vec![<T as num_traits::Zero>::zero(); output_size];
            for (flat_idx, &value) in data.iter().enumerate() {
                let mut remaining = flat_idx;
                let mut out_flat = 0usize;
                for (dim, &reduced) in is_reduced.iter().enumerate() {
                    let coord = remaining / input_strides[dim];
                    remaining %= input_strides[dim];
                    if !reduced {
                        out_flat += coord * output_strides[dim];
                    }
                }
                acc[out_flat] = acc[out_flat] + value;
            }
            Ok(acc)
        })?;

        let final_shape = if keepdim {
            output_shape_keepdim
        } else {
            input_shape
                .into_iter()
                .zip(is_reduced.iter())
                .filter(|(_, &reduced)| !reduced)
                .map(|(size, _)| size)
                .collect::<Vec<_>>()
        };

        Self::from_data(result_data, final_shape, self.device())
    }

    /// Compute the mean along the specified dimensions.
    ///
    /// `dims == None` averages every element. Out-of-range dimensions return an
    /// error instead of panicking.
    pub fn mean(&self, dims: Option<&[usize]>, keepdim: bool) -> Result<Self>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + num_traits::Zero
            + num_traits::One
            + num_traits::FromPrimitive,
    {
        let ndim = self.shape().ndim();

        // Validate and normalise the requested axes before any indexing.
        let reduce_dims: Option<Vec<usize>> = match dims {
            Some(requested) => {
                for &dim in requested {
                    if dim >= ndim {
                        return Err(TorshError::InvalidOperation(format!(
                            "Dimension {} out of range for {}-dimensional tensor",
                            dim, ndim
                        )));
                    }
                }
                let mut normalized = requested.to_vec();
                normalized.sort_unstable();
                normalized.dedup();
                Some(normalized)
            }
            None => None,
        };

        let sum = match &reduce_dims {
            Some(reduce) => {
                let as_i32: Vec<i32> = reduce.iter().map(|&d| d as i32).collect();
                self.sum_dim(&as_i32, keepdim)?
            }
            None => {
                let scalar_sum = self.sum()?;
                if keepdim {
                    let keepdim_shape = vec![1; ndim];
                    scalar_sum.view(&keepdim_shape)?
                } else {
                    scalar_sum
                }
            }
        };

        let count = match &reduce_dims {
            Some(reduce) => {
                let dims_binding = self.shape();
                let shape = dims_binding.dims();
                reduce.iter().map(|&d| shape[d]).product::<usize>() as f64
            }
            None => self.numel() as f64,
        };

        let mut result = sum.div_scalar(
            <T as num_traits::FromPrimitive>::from_f64(count)
                .unwrap_or_else(|| <T as num_traits::One>::one()),
        )?;

        // Propagate requires_grad and record operation for autograd
        if crate::should_record_grad(self.requires_grad) {
            result.requires_grad = true;
            result.operation = crate::core_ops::Operation::Mean {
                input: std::sync::Arc::new(self.clone()),
                count,
            };
        }

        Ok(result)
    }

    /// Cumulative sum along a dimension.
    ///
    /// The accumulator restarts for every independent slice, so `cumsum(0)` on a
    /// `[2, 3]` tensor accumulates down the rows and `cumsum(-1)` accumulates
    /// across each row separately.
    pub fn cumsum(&self, dim: i32) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero + Copy,
    {
        let shape_binding = self.shape();
        let shape = shape_binding.dims().to_vec();
        let axis = normalize_dim(dim, shape.len())?;
        let (outer, dim_size, inner) = split_extents(&shape, axis);

        let mut result_data = self.data()?;
        for o in 0..outer {
            for n in 0..inner {
                let base = o * dim_size * inner + n;
                let mut acc = <T as num_traits::Zero>::zero();
                for d in 0..dim_size {
                    let idx = base + d * inner;
                    acc = acc + result_data[idx];
                    result_data[idx] = acc;
                }
            }
        }

        Self::from_data(result_data, shape, self.device())
    }

    /// Cumulative product along a dimension.
    pub fn cumprod(&self, dim: i32) -> Result<Self>
    where
        T: std::ops::Mul<Output = T> + num_traits::One + Copy,
    {
        let shape_binding = self.shape();
        let shape = shape_binding.dims().to_vec();
        let axis = normalize_dim(dim, shape.len())?;
        let (outer, dim_size, inner) = split_extents(&shape, axis);

        // `data()` already yields an owned, view-ordered buffer: mutate it directly.
        let mut result_data = self.data()?;
        for o in 0..outer {
            for n in 0..inner {
                let base = o * dim_size * inner + n;
                let mut acc = <T as num_traits::One>::one();
                for d in 0..dim_size {
                    let idx = base + d * inner;
                    acc = acc * result_data[idx];
                    result_data[idx] = acc;
                }
            }
        }

        Self::from_data(result_data, shape, self.device())
    }

    /// Sort the tensor along a dimension.
    ///
    /// Returns `(values, indices)` where `indices` are positions inside the
    /// sorted dimension (PyTorch semantics). `dim == None` sorts the last
    /// dimension. Ties keep their original relative order (stable sort).
    pub fn sort(&self, dim: Option<i32>, descending: bool) -> Result<(Self, Tensor<i64>)>
    where
        T: PartialOrd + Copy,
    {
        let shape_binding = self.shape();
        let shape = shape_binding.dims().to_vec();

        if shape.is_empty() {
            // 0-d tensors are already sorted; the only valid index is 0.
            let values = Self::from_data(self.data()?, shape.clone(), self.device())?;
            let indices = Tensor::<i64>::from_data(vec![0], shape, self.device())?;
            return Ok((values, indices));
        }

        let axis = match dim {
            Some(d) => normalize_dim(d, shape.len())?,
            None => shape.len() - 1,
        };
        let (outer, dim_size, inner) = split_extents(&shape, axis);

        let data = self.data()?;
        if data.is_empty() {
            let values = Self::from_data(Vec::new(), shape.clone(), self.device())?;
            let indices = Tensor::<i64>::from_data(Vec::new(), shape, self.device())?;
            return Ok((values, indices));
        }

        let mut sorted_values = vec![data[0]; data.len()];
        let mut sorted_indices = vec![0i64; data.len()];
        let mut slice: Vec<(usize, T)> = Vec::with_capacity(dim_size);

        for o in 0..outer {
            for n in 0..inner {
                let base = o * dim_size * inner + n;
                slice.clear();
                slice.extend((0..dim_size).map(|d| (d, data[base + d * inner])));

                if descending {
                    slice
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    slice
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }

                for (d, (src_idx, value)) in slice.iter().enumerate() {
                    let dst = base + d * inner;
                    sorted_values[dst] = *value;
                    sorted_indices[dst] = *src_idx as i64;
                }
            }
        }

        let values = Self::from_data(sorted_values, shape.clone(), self.device())?;
        let indices = Tensor::<i64>::from_data(sorted_indices, shape, self.device())?;
        Ok((values, indices))
    }

    /// Indices of the minimum values along a dimension.
    ///
    /// `dim == Some(d)` returns a tensor shaped like the input with axis `d`
    /// removed; `dim == None` returns the flat index of the global minimum as a
    /// 0-dimensional tensor.
    pub fn argmin(&self, dim: Option<i32>) -> Result<Tensor<i64>>
    where
        T: PartialOrd + Copy,
    {
        match dim {
            Some(d) => self.arg_extremum_dim(d, false, false),
            None => self.arg_extremum_global(false),
        }
    }

    /// Indices of the maximum values along a dimension.
    pub fn argmax(&self, dim: Option<i32>) -> Result<Tensor<i64>>
    where
        T: PartialOrd + Copy,
    {
        match dim {
            Some(d) => self.arg_extremum_dim(d, true, false),
            None => self.arg_extremum_global(true),
        }
    }

    /// Indices of the minimum values along `dim`, with PyTorch's `keepdim`.
    pub fn argmin_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<i64>>
    where
        T: PartialOrd + Copy,
    {
        self.arg_extremum_dim(dim, false, keepdim)
    }

    /// Indices of the maximum values along `dim`, with PyTorch's `keepdim`.
    pub fn argmax_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<i64>>
    where
        T: PartialOrd + Copy,
    {
        self.arg_extremum_dim(dim, true, keepdim)
    }

    fn arg_extremum_dim(&self, dim: i32, want_max: bool, keepdim: bool) -> Result<Tensor<i64>>
    where
        T: PartialOrd + Copy,
    {
        let shape_binding = self.shape();
        let shape = shape_binding.dims().to_vec();
        let axis = normalize_dim(dim, shape.len())?;
        let (outer, dim_size, inner) = split_extents(&shape, axis);

        if dim_size == 0 {
            return Err(TorshError::InvalidOperation(
                "Cannot compute arg-reduction over a zero-sized dimension".to_string(),
            ));
        }

        let result_data = self.with_contiguous_data(|data| {
            Ok(if want_max {
                extremum_indices(data, outer, dim_size, inner, |new, best| new > best)
            } else {
                extremum_indices(data, outer, dim_size, inner, |new, best| new < best)
            })
        })?;

        Tensor::<i64>::from_data(
            result_data,
            reduced_shape(&shape, axis, keepdim),
            self.device(),
        )
    }

    fn arg_extremum_global(&self, want_max: bool) -> Result<Tensor<i64>>
    where
        T: PartialOrd + Copy,
    {
        let index = self.with_contiguous_data(|data| {
            let mut iter = data.iter().copied().enumerate();
            let (mut best_idx, mut best) = iter.next().ok_or_else(|| {
                TorshError::InvalidOperation(
                    "Cannot compute arg-reduction on empty tensor".to_string(),
                )
            })?;
            for (idx, value) in iter {
                let better = if want_max { value > best } else { value < best };
                if better {
                    best = value;
                    best_idx = idx;
                }
            }
            Ok(best_idx as i64)
        })?;

        Tensor::<i64>::from_data(vec![index], vec![], self.device())
    }
}

/// Row-major (C-contiguous) strides for a shape.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn normalize_dim_handles_negative_and_range() {
        assert_eq!(normalize_dim(-1, 3).expect("valid dim"), 2);
        assert_eq!(normalize_dim(0, 3).expect("valid dim"), 0);
        assert!(normalize_dim(3, 3).is_err());
        assert!(normalize_dim(-4, 3).is_err());
        assert!(normalize_dim(0, 0).is_err());
    }

    #[test]
    fn amin_amax_are_symmetric() {
        let t = Tensor::from_data(vec![1.0f32, 5.0, 3.0, 2.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        let amax_dim1 = t.amax(Some(1), false).expect("amax should succeed");
        let amin_dim1 = t.amin(Some(1), false).expect("amin should succeed");
        assert_eq!(amax_dim1.to_vec().expect("to_vec"), vec![5.0, 3.0]);
        assert_eq!(amin_dim1.to_vec().expect("to_vec"), vec![1.0, 2.0]);

        let amax_all = t.amax(None, false).expect("amax should succeed");
        let amin_all = t.amin(None, false).expect("amin should succeed");
        assert_eq!(amax_all.item().expect("item"), 5.0);
        assert_eq!(amin_all.item().expect("item"), 1.0);

        // keepdim keeps the rank
        let kept = t.amin(Some(0), true).expect("amin should succeed");
        assert_eq!(kept.shape().dims(), &[1, 2]);
    }

    #[test]
    fn min_dim_and_max_dim_agree_with_amin_amax() {
        let t = Tensor::from_data(
            vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 0.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation should succeed");

        assert_eq!(
            t.min_dim(-1, false).expect("min_dim").to_vec().expect("v"),
            t.amin(Some(1), false).expect("amin").to_vec().expect("v")
        );
        assert_eq!(
            t.max_dim(-1, false).expect("max_dim").to_vec().expect("v"),
            t.amax(Some(1), false).expect("amax").to_vec().expect("v")
        );
    }

    #[test]
    fn argmax_dim_keepdim_shape() {
        let t = Tensor::from_data(
            vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 0.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation should succeed");
        let out = t.argmax_dim(1, true).expect("argmax_dim should succeed");
        assert_eq!(out.shape().dims(), &[2, 1]);
        assert_eq!(out.to_vec().expect("to_vec"), vec![1i64, 1]);
    }

    #[test]
    fn sort_indices_are_i64_positions_within_dim() {
        let t = Tensor::from_data(
            vec![3.0f32, 1.0, 2.0, 0.0, 5.0, 4.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation should succeed");
        let (values, indices) = t.sort(Some(1), false).expect("sort should succeed");
        assert_eq!(
            values.to_vec().expect("v"),
            vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0]
        );
        assert_eq!(indices.to_vec().expect("v"), vec![1i64, 2, 0, 0, 2, 1]);
    }

    #[test]
    fn sort_on_a_view_uses_view_order() {
        let base = Tensor::from_data(
            vec![3.0f32, 1.0, 2.0, 0.0, 5.0, 4.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation should succeed");
        let view = base.transpose_view(0, 1).expect("transpose_view");
        // view = [[3,0],[1,5],[2,4]]; sort along dim 1 -> [[0,3],[1,5],[2,4]]
        let (values, indices) = view.sort(Some(1), false).expect("sort should succeed");
        assert_eq!(
            values.to_vec().expect("v"),
            vec![0.0, 3.0, 1.0, 5.0, 2.0, 4.0]
        );
        assert_eq!(indices.to_vec().expect("v"), vec![1i64, 0, 0, 1, 0, 1]);
    }
}
