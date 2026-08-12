//! Spatial interpolation methods for image processing

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{Result, VisionError};
use torsh_tensor::Tensor;
// Note: interpolation module not available in scirs2_spatial, will implement basic interpolation
use scirs2_core::ndarray::{arr1, arr2, Array1, Array2, ArrayView2};

/// Spatial interpolation configuration
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    pub method: InterpolationMethod,
    pub kernel: KernelType,
    pub power: f64,
    pub smoothing: f64,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            // Inverse-distance weighting is the default because it is defined
            // for arbitrary scattered samples; NaturalNeighbor is not implemented.
            method: InterpolationMethod::InverseDistanceWeighting,
            kernel: KernelType::Gaussian,
            power: 2.0,
            smoothing: 0.0,
        }
    }
}

/// Supported interpolation methods
#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    NaturalNeighbor,
    RadialBasisFunction,
    InverseDistanceWeighting,
    Bilinear,
    Bicubic,
}

/// Kernel types for RBF interpolation
#[derive(Debug, Clone)]
pub enum KernelType {
    Gaussian,
    Multiquadric,
    InverseMultiquadric,
    ThinPlateSpline,
}

/// Spatial interpolator for image and point data
pub struct SpatialInterpolator {
    config: InterpolationConfig,
}

impl SpatialInterpolator {
    /// Create a new spatial interpolator
    pub fn new(config: InterpolationConfig) -> Self {
        Self { config }
    }

    /// Interpolate sparse data points to a regular grid
    pub fn interpolate_to_grid(
        &self,
        points: &Array2<f64>,
        values: &Array1<f64>,
        grid_points: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        if points.nrows() != values.len() {
            return Err(VisionError::InvalidArgument(
                "Number of points must match number of values".to_string(),
            ));
        }

        if points.ncols() != grid_points.ncols() {
            return Err(VisionError::InvalidArgument(
                "Points and grid points must have same dimensionality".to_string(),
            ));
        }

        match self.config.method {
            InterpolationMethod::NaturalNeighbor => {
                self.natural_neighbor_interpolation(points, values, grid_points)
            }
            InterpolationMethod::RadialBasisFunction => {
                self.rbf_interpolation(points, values, grid_points)
            }
            InterpolationMethod::InverseDistanceWeighting => {
                self.idw_interpolation(points, values, grid_points)
            }
            InterpolationMethod::Bilinear => {
                self.bilinear_interpolation(points, values, grid_points)
            }
            InterpolationMethod::Bicubic => self.bicubic_interpolation(points, values, grid_points),
        }
    }

    fn natural_neighbor_interpolation(
        &self,
        _points: &Array2<f64>,
        _values: &Array1<f64>,
        _grid_points: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        // Natural-neighbour interpolation needs a Voronoi/Delaunay tessellation
        // with Sibson weights, which this crate does not implement yet. Report it
        // honestly instead of silently returning a field of zeros.
        Err(VisionError::UnsupportedOperation(
            "Natural neighbor interpolation is not implemented; use \
             InterpolationMethod::InverseDistanceWeighting or RadialBasisFunction"
                .to_string(),
        ))
    }

    /// Radial basis function interpolation
    ///
    /// Solves the RBF interpolation system `Phi * w = values` for the weights with
    /// Gaussian elimination (partial pivoting) and evaluates
    /// `f(g) = sum_i w_i * phi(||g - p_i||)` on the grid. `config.smoothing` is
    /// added to the diagonal as Tikhonov regularisation.
    fn rbf_interpolation(
        &self,
        points: &Array2<f64>,
        values: &Array1<f64>,
        grid_points: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let n = points.nrows();
        if n == 0 {
            return Err(VisionError::InvalidArgument(
                "RBF interpolation requires at least one sample point".to_string(),
            ));
        }

        // Shape parameter: the mean nearest-neighbour spacing keeps the kernel
        // well conditioned regardless of the data scale.
        let epsilon = self.rbf_shape_parameter(points);

        // Build the (regularised) interpolation matrix.
        let mut matrix = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                let distance = euclidean_distance(&points.row(i), &points.row(j));
                let mut phi = self.rbf_kernel(distance, epsilon);
                if i == j {
                    phi += self.config.smoothing;
                }
                matrix[i][j] = phi;
            }
            matrix[i][n] = values[i];
        }

        let weights = solve_linear_system(&mut matrix, n)?;

        let mut interpolated = Array1::zeros(grid_points.nrows());
        for (g, grid_point) in grid_points.outer_iter().enumerate() {
            let mut acc = 0.0;
            for (i, data_point) in points.outer_iter().enumerate() {
                let distance = euclidean_distance(&grid_point, &data_point);
                acc += weights[i] * self.rbf_kernel(distance, epsilon);
            }
            interpolated[g] = acc;
        }

        Ok(interpolated)
    }

    /// Evaluate the configured radial basis kernel
    fn rbf_kernel(&self, distance: f64, epsilon: f64) -> f64 {
        let scaled = distance / epsilon;
        match self.config.kernel {
            KernelType::Gaussian => (-scaled * scaled).exp(),
            KernelType::Multiquadric => (1.0 + scaled * scaled).sqrt(),
            KernelType::InverseMultiquadric => 1.0 / (1.0 + scaled * scaled).sqrt(),
            KernelType::ThinPlateSpline => {
                if distance <= f64::EPSILON {
                    0.0
                } else {
                    distance * distance * distance.ln()
                }
            }
        }
    }

    /// Mean nearest-neighbour distance, used as the RBF shape parameter
    fn rbf_shape_parameter(&self, points: &Array2<f64>) -> f64 {
        let n = points.nrows();
        if n < 2 {
            return 1.0;
        }

        let mut total = 0.0;
        let mut counted = 0usize;
        for i in 0..n {
            let mut nearest = f64::INFINITY;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let distance = euclidean_distance(&points.row(i), &points.row(j));
                if distance < nearest {
                    nearest = distance;
                }
            }
            if nearest.is_finite() && nearest > 0.0 {
                total += nearest;
                counted += 1;
            }
        }

        if counted == 0 || total <= 0.0 {
            1.0
        } else {
            total / counted as f64
        }
    }

    fn idw_interpolation(
        &self,
        points: &Array2<f64>,
        values: &Array1<f64>,
        grid_points: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        // Simple inverse distance weighting implementation
        let mut interpolated = Array1::zeros(grid_points.nrows());

        for (i, grid_point) in grid_points.outer_iter().enumerate() {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (j, data_point) in points.outer_iter().enumerate() {
                let diff = &grid_point - &data_point;
                let distance = (diff.mapv(|x| x * x).sum()).sqrt();

                if distance < 1e-10 {
                    // Exact match
                    interpolated[i] = values[j];
                    weight_sum = 1.0;
                    weighted_sum = values[j];
                    break;
                } else {
                    let weight = 1.0 / distance.powf(self.config.power);
                    weighted_sum += weight * values[j];
                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                interpolated[i] = weighted_sum / weight_sum;
            }
        }

        Ok(interpolated)
    }

    fn bilinear_interpolation(
        &self,
        _points: &Array2<f64>,
        _values: &Array1<f64>,
        _grid_points: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        // Bilinear interpolation is only defined on a regular lattice; this entry
        // point receives scattered samples. Use `super_resolution` /
        // `warp_image` for image-lattice bilinear sampling.
        Err(VisionError::UnsupportedOperation(
            "Bilinear interpolation requires samples on a regular grid; use \
             InterpolationMethod::InverseDistanceWeighting or RadialBasisFunction \
             for scattered data"
                .to_string(),
        ))
    }

    fn bicubic_interpolation(
        &self,
        _points: &Array2<f64>,
        _values: &Array1<f64>,
        _grid_points: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        Err(VisionError::UnsupportedOperation(
            "Bicubic interpolation requires samples on a regular grid; use \
             InterpolationMethod::InverseDistanceWeighting or RadialBasisFunction \
             for scattered data"
                .to_string(),
        ))
    }

    /// Interpolate missing pixels in an image
    ///
    /// `image` is a 2D `(H, W)` or 3D `(C, H, W)` tensor and `mask` has the same
    /// spatial extent (either `(H, W)` or matching the image shape). Mask values
    /// greater than `0.5` mark *known* pixels; every other pixel is reconstructed
    /// with inverse-distance weighting over the known pixels found in a growing
    /// square neighbourhood.
    pub fn interpolate_image_gaps(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let dims = image.shape().dims().to_vec();
        let (channels, height, width) = match dims.len() {
            2 => (1usize, dims[0], dims[1]),
            3 => (dims[0], dims[1], dims[2]),
            other => {
                return Err(VisionError::InvalidShape(format!(
                    "interpolate_image_gaps expects a 2D (H, W) or 3D (C, H, W) tensor, got {}D",
                    other
                )))
            }
        };

        if height == 0 || width == 0 {
            return Err(VisionError::InvalidArgument(
                "Cannot inpaint an image with an empty spatial extent".to_string(),
            ));
        }

        let plane_len = height * width;
        let mask_data = mask.to_vec()?;
        if mask_data.len() != plane_len && mask_data.len() != plane_len * channels {
            return Err(VisionError::InvalidShape(format!(
                "Mask must cover the image plane ({} elements) or the whole image ({} elements), \
                 got {}",
                plane_len,
                plane_len * channels,
                mask_data.len()
            )));
        }

        let data = image.to_vec()?;
        let mut output = data.clone();
        let power = if self.config.power > 0.0 {
            self.config.power
        } else {
            2.0
        };
        let max_radius = height.max(width);

        for c in 0..channels {
            let base = c * plane_len;
            let mask_base = if mask_data.len() == plane_len {
                0
            } else {
                base
            };

            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    if mask_data[mask_base + idx] > 0.5 {
                        continue; // Known pixel: keep it.
                    }

                    let mut weighted_sum = 0.0f64;
                    let mut weight_sum = 0.0f64;

                    // Grow the neighbourhood until enough known pixels are found.
                    let mut radius = 1usize;
                    while radius <= max_radius {
                        let y_lo = y.saturating_sub(radius);
                        let y_hi = (y + radius).min(height - 1);
                        let x_lo = x.saturating_sub(radius);
                        let x_hi = (x + radius).min(width - 1);

                        weighted_sum = 0.0;
                        weight_sum = 0.0;
                        for ny in y_lo..=y_hi {
                            for nx in x_lo..=x_hi {
                                let nidx = ny * width + nx;
                                if mask_data[mask_base + nidx] <= 0.5 {
                                    continue;
                                }
                                let dy = ny as f64 - y as f64;
                                let dx = nx as f64 - x as f64;
                                let distance = (dx * dx + dy * dy).sqrt();
                                if distance < 1e-12 {
                                    continue;
                                }
                                let weight = 1.0 / distance.powf(power);
                                weighted_sum += weight * data[base + nidx] as f64;
                                weight_sum += weight;
                            }
                        }

                        if weight_sum > 0.0 {
                            break;
                        }
                        radius *= 2;
                    }

                    if weight_sum > 0.0 {
                        output[base + idx] = (weighted_sum / weight_sum) as f32;
                    }
                }
            }
        }

        Tensor::from_vec(output, &dims).map_err(VisionError::TensorError)
    }

    /// Super-resolution using spatial interpolation
    ///
    /// Upscales a 3D `(C, H, W)` tensor by `scale_factor` using the interpolation
    /// method configured on this interpolator (bicubic and bilinear both map onto
    /// the corresponding resize kernel, everything else falls back to bilinear).
    pub fn super_resolution(&self, low_res_image: &Tensor, scale_factor: f64) -> Result<Tensor> {
        if scale_factor <= 1.0 {
            return Err(VisionError::InvalidArgument(
                "Scale factor must be greater than 1.0".to_string(),
            ));
        }
        if !scale_factor.is_finite() {
            return Err(VisionError::InvalidArgument(
                "Scale factor must be finite".to_string(),
            ));
        }

        let dims = low_res_image.shape().dims().to_vec();
        let (height, width) = match dims.len() {
            3 => (dims[1], dims[2]),
            4 => (dims[2], dims[3]),
            other => {
                return Err(VisionError::InvalidShape(format!(
                    "super_resolution expects a 3D (C, H, W) or 4D (N, C, H, W) tensor, got {}D",
                    other
                )))
            }
        };

        let target_height = ((height as f64) * scale_factor).round() as usize;
        let target_width = ((width as f64) * scale_factor).round() as usize;

        let mode = match self.config.method {
            InterpolationMethod::Bicubic => crate::ops::InterpolationMode::Bicubic,
            _ => crate::ops::InterpolationMode::Bilinear,
        };

        crate::ops::resize_with_mode(low_res_image, (target_width, target_height), mode)
    }
}

/// Euclidean distance between two coordinate rows
fn euclidean_distance(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc.sqrt()
}

/// Solve a dense `n x n` linear system given as an augmented `n x (n + 1)` matrix
///
/// Uses Gaussian elimination with partial pivoting.
fn solve_linear_system(matrix: &mut [Vec<f64>], n: usize) -> Result<Vec<f64>> {
    for col in 0..n {
        // Partial pivoting for numerical stability.
        let mut pivot = col;
        for row in (col + 1)..n {
            if matrix[row][col].abs() > matrix[pivot][col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][col].abs() < 1e-12 {
            return Err(VisionError::InvalidArgument(
                "Interpolation system is singular; the sample points are degenerate".to_string(),
            ));
        }
        matrix.swap(col, pivot);

        let diagonal = matrix[col][col];
        for row in (col + 1)..n {
            let factor = matrix[row][col] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for k in col..=n {
                matrix[row][k] -= factor * matrix[col][k];
            }
        }
    }

    let mut solution = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut acc = matrix[row][n];
        for col in (row + 1)..n {
            acc -= matrix[row][col] * solution[col];
        }
        solution[row] = acc / matrix[row][row];
    }

    Ok(solution)
}

/// Image warping using spatial interpolation
pub struct ImageWarper {
    interpolator: SpatialInterpolator,
}

impl ImageWarper {
    /// Create a new image warper
    pub fn new(config: InterpolationConfig) -> Self {
        Self {
            interpolator: SpatialInterpolator::new(config),
        }
    }

    /// Warp image using a displacement field
    ///
    /// `image` is a 2D `(H, W)` or 3D `(C, H, W)` tensor. `displacement_field` has
    /// one `(dx, dy)` row per pixel in row-major order (`H * W` rows, 2 columns);
    /// the output pixel `(x, y)` is sampled from the source at
    /// `(x + dx, y + dy)` with clamped bilinear interpolation.
    pub fn warp_image(&self, image: &Tensor, displacement_field: &Array2<f64>) -> Result<Tensor> {
        let dims = image.shape().dims().to_vec();
        let (height, width) = match dims.len() {
            2 => (dims[0], dims[1]),
            3 => (dims[1], dims[2]),
            other => {
                return Err(VisionError::InvalidShape(format!(
                    "warp_image expects a 2D (H, W) or 3D (C, H, W) tensor, got {}D",
                    other
                )))
            }
        };

        if displacement_field.ncols() != 2 {
            return Err(VisionError::InvalidArgument(format!(
                "Displacement field must have 2 columns (dx, dy), got {}",
                displacement_field.ncols()
            )));
        }
        if displacement_field.nrows() != height * width {
            return Err(VisionError::InvalidArgument(format!(
                "Displacement field must have one row per pixel ({}), got {}",
                height * width,
                displacement_field.nrows()
            )));
        }

        crate::ops::common::utils::inverse_warp_chw(image, |x, y| {
            let idx = (y as usize) * width + (x as usize);
            (
                x + displacement_field[[idx, 0]] as f32,
                y + displacement_field[[idx, 1]] as f32,
            )
        })
    }

    /// Apply barrel distortion correction
    ///
    /// Barrel distortion pushes pixels towards the centre, so the correction
    /// samples the source at the *distorted* radius
    /// `r_src = r * (1 + k1 * r^2 + k2 * r^4 + ...)`, with `r` normalised by half
    /// the image diagonal so the coefficients are resolution independent.
    pub fn correct_barrel_distortion(
        &self,
        image: &Tensor,
        distortion_coeffs: &Array1<f64>,
    ) -> Result<Tensor> {
        radial_undistort(image, distortion_coeffs, 1.0)
    }

    /// Apply pincushion distortion correction
    ///
    /// The mirror of [`Self::correct_barrel_distortion`]: the polynomial is
    /// applied with the opposite sign so positive coefficients describe
    /// pincushion rather than barrel distortion.
    pub fn correct_pincushion_distortion(
        &self,
        image: &Tensor,
        distortion_coeffs: &Array1<f64>,
    ) -> Result<Tensor> {
        radial_undistort(image, distortion_coeffs, -1.0)
    }
}

/// Shared radial (de)distortion warp used by the barrel/pincushion correctors
fn radial_undistort(image: &Tensor, coeffs: &Array1<f64>, sign: f64) -> Result<Tensor> {
    if coeffs.is_empty() {
        return Err(VisionError::InvalidArgument(
            "At least one radial distortion coefficient is required".to_string(),
        ));
    }

    let dims = image.shape().dims().to_vec();
    let (height, width) = match dims.len() {
        2 => (dims[0], dims[1]),
        3 => (dims[1], dims[2]),
        other => {
            return Err(VisionError::InvalidShape(format!(
                "Distortion correction expects a 2D (H, W) or 3D (C, H, W) tensor, got {}D",
                other
            )))
        }
    };
    if height == 0 || width == 0 {
        return Err(VisionError::InvalidArgument(
            "Cannot correct distortion on an image with an empty spatial extent".to_string(),
        ));
    }

    let cx = (width as f64 - 1.0) / 2.0;
    let cy = (height as f64 - 1.0) / 2.0;
    // Normalise by half the diagonal so the coefficients do not depend on the
    // image resolution.
    let norm = ((cx * cx + cy * cy).sqrt()).max(1e-12);

    let coefficients: Vec<f64> = coeffs.iter().copied().collect();

    crate::ops::common::utils::inverse_warp_chw(image, |x, y| {
        let dx = x as f64 - cx;
        let dy = y as f64 - cy;
        let r = ((dx * dx + dy * dy).sqrt()) / norm;

        let mut factor = 1.0;
        let mut r_pow = 1.0;
        for k in &coefficients {
            r_pow *= r * r;
            factor += sign * k * r_pow;
        }

        ((cx + dx * factor) as f32, (cy + dy * factor) as f32)
    })
}

/// Dense optical flow using spatial interpolation
pub struct OpticalFlowInterpolator {
    config: InterpolationConfig,
}

impl OpticalFlowInterpolator {
    /// Create a new optical flow interpolator
    pub fn new(config: InterpolationConfig) -> Self {
        Self { config }
    }

    /// Interpolate sparse optical flow to dense flow field
    pub fn interpolate_flow(
        &self,
        sparse_points: &Array2<f64>,
        flow_vectors: &Array2<f64>,
        image_size: (usize, usize),
    ) -> Result<Array2<f64>> {
        if sparse_points.nrows() != flow_vectors.nrows() {
            return Err(VisionError::InvalidArgument(
                "Number of points must match number of flow vectors".to_string(),
            ));
        }

        // Create dense grid
        let mut grid_points = Vec::new();
        for y in 0..image_size.1 {
            for x in 0..image_size.0 {
                grid_points.push([x as f64, y as f64]);
            }
        }

        let grid_array = Array2::from_shape_vec(
            (image_size.0 * image_size.1, 2),
            grid_points.into_iter().flatten().collect(),
        )
        .map_err(|e| VisionError::Other(anyhow::anyhow!("Grid creation failed: {}", e)))?;

        // Interpolate flow components separately
        let interpolator = SpatialInterpolator::new(self.config.clone());

        let flow_x = flow_vectors.column(0).to_owned();
        let flow_y = flow_vectors.column(1).to_owned();

        let interpolated_x =
            interpolator.interpolate_to_grid(sparse_points, &flow_x, &grid_array)?;
        let interpolated_y =
            interpolator.interpolate_to_grid(sparse_points, &flow_y, &grid_array)?;

        // Combine interpolated components
        let mut dense_flow = Array2::zeros((grid_array.nrows(), 2));
        for i in 0..grid_array.nrows() {
            dense_flow[[i, 0]] = interpolated_x[i];
            dense_flow[[i, 1]] = interpolated_y[i];
        }

        Ok(dense_flow)
    }

    /// Smooth an optical flow field
    ///
    /// Applies a 1D binomial (`[1, 2, 1] / 4`) smoothing pass along the sample
    /// order of the field, independently per flow component. The field is
    /// `N x 2` with one `(dx, dy)` row per sample.
    pub fn smooth_flow_field(&self, flow_field: &Array2<f64>) -> Result<Array2<f64>> {
        if flow_field.ncols() != 2 {
            return Err(VisionError::InvalidArgument(format!(
                "Flow field must have 2 columns (dx, dy), got {}",
                flow_field.ncols()
            )));
        }

        let n = flow_field.nrows();
        let mut smoothed = flow_field.clone();
        if n < 3 {
            return Ok(smoothed);
        }

        for component in 0..2 {
            for i in 1..(n - 1) {
                smoothed[[i, component]] = 0.25 * flow_field[[i - 1, component]]
                    + 0.5 * flow_field[[i, component]]
                    + 0.25 * flow_field[[i + 1, component]];
            }
        }

        Ok(smoothed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // arr1, arr2 imported above

    #[test]
    fn test_interpolation_config_default() {
        let config = InterpolationConfig::default();
        assert!(matches!(
            config.method,
            InterpolationMethod::InverseDistanceWeighting
        ));
        assert!(matches!(config.kernel, KernelType::Gaussian));
    }

    #[test]
    fn test_spatial_interpolator_creation() {
        let config = InterpolationConfig::default();
        let interpolator = SpatialInterpolator::new(config);
        assert!(matches!(
            interpolator.config.method,
            InterpolationMethod::InverseDistanceWeighting
        ));
    }

    #[test]
    fn test_idw_interpolation() {
        let config = InterpolationConfig {
            method: InterpolationMethod::InverseDistanceWeighting,
            ..Default::default()
        };

        let interpolator = SpatialInterpolator::new(config);

        let points = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
        let values = arr1(&[0.0, 1.0]);
        let grid_points = arr2(&[[0.5, 0.5]]);

        let result = interpolator.interpolate_to_grid(&points, &values, &grid_points);
        assert!(result.is_ok());
    }

    #[test]
    fn test_image_warper_creation() {
        let config = InterpolationConfig::default();
        let warper = ImageWarper::new(config);
        assert!(matches!(
            warper.interpolator.config.method,
            InterpolationMethod::InverseDistanceWeighting
        ));
    }

    #[test]
    fn test_optical_flow_interpolator() {
        let config = InterpolationConfig::default();
        let flow_interpolator = OpticalFlowInterpolator::new(config);
        assert!(matches!(
            flow_interpolator.config.method,
            InterpolationMethod::InverseDistanceWeighting
        ));
    }
}
