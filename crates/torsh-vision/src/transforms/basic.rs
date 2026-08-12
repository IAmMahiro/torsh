use super::core::Transform;
use crate::{Result, VisionError};
use torsh_tensor::Tensor;

/// Resize transform
///
/// Resizes input images to a specified size. This is one of the most commonly used
/// transforms for standardizing input dimensions.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Resize, Transform};
///
/// let resize = Resize::new((224, 224));
/// // Apply to tensor: result = resize.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct Resize {
    size: (usize, usize),
}

impl Resize {
    /// Create a new Resize transform
    ///
    /// # Arguments
    ///
    /// * `size` - Target size as (width, height)
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }

    /// Get the target size
    pub fn size(&self) -> (usize, usize) {
        self.size
    }
}

impl Transform for Resize {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::resize(input, self.size)
    }

    fn name(&self) -> &'static str {
        "Resize"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("size", format!("({}, {})", self.size.0, self.size.1))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Resize::new(self.size))
    }
}

/// Center crop transform
///
/// Crops the input image at the center to the specified size. Useful for creating
/// uniform image sizes while preserving the central content.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{CenterCrop, Transform};
///
/// let crop = CenterCrop::new((224, 224));
/// // Apply to tensor: result = crop.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct CenterCrop {
    size: (usize, usize),
}

impl CenterCrop {
    /// Create a new CenterCrop transform
    ///
    /// # Arguments
    ///
    /// * `size` - Target crop size as (width, height)
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }

    /// Get the crop size
    pub fn size(&self) -> (usize, usize) {
        self.size
    }
}

impl Transform for CenterCrop {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::center_crop(input, self.size)
    }

    fn name(&self) -> &'static str {
        "CenterCrop"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("size", format!("({}, {})", self.size.0, self.size.1))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(CenterCrop::new(self.size))
    }
}

/// Memory layout of an image tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageLayout {
    /// Height x Width x Channels — the layout produced by image decoders
    /// (`image::DynamicImage`, PIL, OpenCV). This is the default input layout
    /// of [`ToTensor`], matching torchvision.
    #[default]
    Hwc,
    /// Channels x Height x Width — the layout every ToRSh vision op expects.
    Chw,
}

/// Convert a decoded image tensor into the CHW tensor layout used by the models
///
/// This mirrors `torchvision.transforms.ToTensor`: the input is interpreted as an
/// interleaved image (HWC, or NHWC when batched) holding 8-bit sample values in
/// `0..=255`, and the output is a planar CHW (or NCHW) tensor scaled into
/// `0.0..=1.0`.
///
/// Both steps are configurable and applied unconditionally — the transform never
/// inspects the data to guess whether it should scale, so the same pipeline always
/// produces the same normalisation.
///
/// * Inputs with 2 dimensions are treated as a single-channel `H x W` image and
///   gain a leading channel axis.
/// * Use [`ToTensor::from_chw`] when the input is already planar and only the
///   `1/255` scaling is wanted.
/// * Use [`ToTensor::with_scale`] to disable the scaling for inputs that already
///   live in `0.0..=1.0`.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{ToTensor, Transform};
///
/// // torchvision semantics: HWC in 0..=255 -> CHW in 0.0..=1.0
/// let to_tensor = ToTensor::new();
///
/// // Already planar, still needs the 1/255 scaling
/// let planar = ToTensor::from_chw();
///
/// // Planar and already scaled: a no-op pass-through
/// let identity = ToTensor::from_chw().with_scale(false);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ToTensor {
    layout: ImageLayout,
    scale: bool,
}

impl Default for ToTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToTensor {
    /// Create a new ToTensor transform with torchvision semantics
    ///
    /// The input is treated as HWC/NHWC in `0..=255` and the output is CHW/NCHW
    /// in `0.0..=1.0`.
    pub fn new() -> Self {
        Self {
            layout: ImageLayout::Hwc,
            scale: true,
        }
    }

    /// Create a ToTensor transform for input that is already in CHW/NCHW layout
    ///
    /// Only the `1/255` scaling is applied.
    pub fn from_chw() -> Self {
        Self {
            layout: ImageLayout::Chw,
            scale: true,
        }
    }

    /// Create a ToTensor transform for an explicit input layout
    pub fn with_layout(layout: ImageLayout) -> Self {
        Self {
            layout,
            scale: true,
        }
    }

    /// Enable or disable the `1/255` scaling
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Get the expected input layout
    pub fn layout(&self) -> ImageLayout {
        self.layout
    }

    /// Whether the transform divides the samples by 255
    pub fn scales(&self) -> bool {
        self.scale
    }
}

impl Transform for ToTensor {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let dims = input.shape().dims().to_vec();

        let planar = match (dims.len(), self.layout) {
            // Single-channel image: add the channel axis.
            (2, _) => input.view(&[1, dims[0] as i32, dims[1] as i32])?,
            (3, ImageLayout::Hwc) => input.permute(&[2, 0, 1])?.contiguous()?,
            (4, ImageLayout::Hwc) => input.permute(&[0, 3, 1, 2])?.contiguous()?,
            (3, ImageLayout::Chw) | (4, ImageLayout::Chw) => input.clone(),
            _ => {
                return Err(VisionError::InvalidShape(format!(
                    "ToTensor expects a 2D (H, W), 3D (H, W, C) or 4D (N, H, W, C) tensor, got {}D",
                    dims.len()
                )))
            }
        };

        if self.scale {
            Ok(planar.div_scalar(255.0)?)
        } else {
            Ok(planar)
        }
    }

    fn name(&self) -> &'static str {
        "ToTensor"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("layout", format!("{:?}", self.layout)),
            ("scale", format!("{}", self.scale)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(*self)
    }
}

/// Normalize transform
///
/// Normalizes tensor images with mean and standard deviation. This is typically
/// applied as the final preprocessing step before feeding data to neural networks.
///
/// The normalization formula is: `(input - mean) / std`
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Normalize, Transform};
///
/// // ImageNet normalization
/// let normalize = Normalize::new(
///     vec![0.485, 0.456, 0.406],  // RGB means
///     vec![0.229, 0.224, 0.225]   // RGB standard deviations
/// );
/// // Apply to tensor: result = normalize.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    /// Create a new Normalize transform
    ///
    /// # Arguments
    ///
    /// * `mean` - Per-channel means for normalization
    /// * `std` - Per-channel standard deviations for normalization
    ///
    /// # Panics
    ///
    /// Panics if `mean` and `std` have different lengths
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(
            mean.len(),
            std.len(),
            "Mean and std must have the same length"
        );
        Self { mean, std }
    }

    /// Fallible variant of [`Self::new`]
    ///
    /// Returns [`VisionError::InvalidArgument`] instead of panicking when `mean`
    /// and `std` have different lengths, when either is empty, or when any
    /// standard deviation is zero (which would make the normalisation divide by
    /// zero).
    pub fn try_new(mean: Vec<f32>, std: Vec<f32>) -> Result<Self> {
        if mean.len() != std.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Normalize: mean and std must have the same length, got {} and {}",
                mean.len(),
                std.len()
            )));
        }
        if mean.is_empty() {
            return Err(VisionError::InvalidArgument(
                "Normalize: mean and std must not be empty".to_string(),
            ));
        }
        if let Some(idx) = std.iter().position(|s| *s == 0.0) {
            return Err(VisionError::InvalidArgument(format!(
                "Normalize: std[{}] is 0.0, which would divide by zero",
                idx
            )));
        }
        Ok(Self { mean, std })
    }

    /// Create ImageNet normalization (RGB)
    pub fn imagenet() -> Self {
        Self::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
    }

    /// Create CIFAR normalization (RGB)
    pub fn cifar() -> Self {
        Self::new(vec![0.4914, 0.4822, 0.4465], vec![0.2023, 0.1994, 0.2010])
    }

    /// Get the normalization mean
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Get the normalization standard deviation
    pub fn std(&self) -> &[f32] {
        &self.std
    }
}

impl Transform for Normalize {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::normalize(
            input,
            crate::ops::color::NormalizationConfig {
                method: crate::ops::color::NormalizationMethod::Custom,
                mean: Some(self.mean.clone()),
                std: Some(self.std.clone()),
                per_channel: true,
                eps: 1e-8,
            },
        )
    }

    fn name(&self) -> &'static str {
        "Normalize"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("mean", format!("{:?}", self.mean)),
            ("std", format!("{:?}", self.std)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Normalize::new(self.mean.clone(), self.std.clone()))
    }
}

/// Padding transform
///
/// Pads the input tensor with a specified value. Useful for increasing image
/// size before random cropping or for maintaining spatial dimensions.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Pad, Transform};
///
/// // Pad with 4 pixels on all sides, filled with black (0.0)
/// let pad = Pad::symmetric(4, 0.0);
///
/// // Asymmetric padding: (left, top, right, bottom)
/// let pad_custom = Pad::new((2, 4, 2, 4), 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct Pad {
    padding: (usize, usize, usize, usize), // (left, top, right, bottom)
    fill: f32,
}

impl Pad {
    /// Create a new Pad transform with asymmetric padding
    ///
    /// # Arguments
    ///
    /// * `padding` - Padding amounts as (left, top, right, bottom)
    /// * `fill` - Fill value for padded regions
    pub fn new(padding: (usize, usize, usize, usize), fill: f32) -> Self {
        Self { padding, fill }
    }

    /// Create symmetric padding (same amount on all sides)
    ///
    /// # Arguments
    ///
    /// * `pad` - Padding amount for all sides
    /// * `fill` - Fill value for padded regions
    pub fn symmetric(pad: usize, fill: f32) -> Self {
        Self {
            padding: (pad, pad, pad, pad),
            fill,
        }
    }

    /// Create padding for specific sides
    ///
    /// # Arguments
    ///
    /// * `horizontal` - Padding for left and right sides
    /// * `vertical` - Padding for top and bottom sides
    /// * `fill` - Fill value for padded regions
    pub fn sides(horizontal: usize, vertical: usize, fill: f32) -> Self {
        Self {
            padding: (horizontal, vertical, horizontal, vertical),
            fill,
        }
    }

    /// Get the padding configuration
    pub fn padding(&self) -> (usize, usize, usize, usize) {
        self.padding
    }

    /// Get the fill value
    pub fn fill(&self) -> f32 {
        self.fill
    }
}

impl Transform for Pad {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::pad(
            input,
            self.padding,
            crate::ops::PaddingMode::Zero,
            self.fill,
        )
    }

    fn name(&self) -> &'static str {
        "Pad"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "padding",
                format!(
                    "({}, {}, {}, {})",
                    self.padding.0, self.padding.1, self.padding.2, self.padding.3
                ),
            ),
            ("fill", format!("{:.2}", self.fill)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Pad::new(self.padding, self.fill))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_resize_creation() {
        let resize = Resize::new((224, 224));
        assert_eq!(resize.size(), (224, 224));
        assert_eq!(resize.name(), "Resize");

        let params = resize.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "size");
        assert_eq!(params[0].1, "(224, 224)");
    }

    #[test]
    fn test_center_crop_creation() {
        let crop = CenterCrop::new((128, 128));
        assert_eq!(crop.size(), (128, 128));
        assert_eq!(crop.name(), "CenterCrop");

        let params = crop.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "size");
        assert_eq!(params[0].1, "(128, 128)");
    }

    #[test]
    fn test_to_tensor_creation() {
        let to_tensor = ToTensor::new();
        assert_eq!(to_tensor.name(), "ToTensor");
        assert_eq!(to_tensor.layout(), ImageLayout::Hwc);
        assert!(to_tensor.scales());

        let params = to_tensor.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "layout");
        assert_eq!(params[1].0, "scale");
    }

    #[test]
    fn test_to_tensor_default() {
        let to_tensor = ToTensor::default();
        assert_eq!(to_tensor.name(), "ToTensor");
        assert_eq!(to_tensor.layout(), ImageLayout::Hwc);
        assert!(to_tensor.scales());
    }

    #[test]
    fn test_to_tensor_forward_hwc_to_chw() {
        let to_tensor = ToTensor::new();
        // 4x8 image with 3 interleaved channels.
        let input = creation::ones(&[4, 8, 3]).expect("creation should succeed");

        let result = to_tensor
            .forward(&input)
            .expect("forward pass should succeed");
        assert_eq!(result.shape().dims(), &[3, 4, 8]);
        assert!(
            (result
                .get(&[0, 0, 0])
                .expect("element retrieval should succeed for valid index")
                - 1.0 / 255.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_to_tensor_forward_chw_passthrough() {
        let to_tensor = ToTensor::from_chw().with_scale(false);
        let input = creation::ones(&[3, 32, 32]).expect("creation should succeed");

        let result = to_tensor
            .forward(&input)
            .expect("forward pass should succeed");
        assert_eq!(result.shape().dims(), &[3, 32, 32]);
        assert_eq!(
            result
                .get(&[0, 0, 0])
                .expect("element retrieval should succeed for valid index"),
            1.0
        );
    }

    #[test]
    fn test_to_tensor_rejects_unsupported_rank() {
        let to_tensor = ToTensor::new();
        let input = creation::ones(&[5]).expect("creation should succeed");
        assert!(to_tensor.forward(&input).is_err());
    }

    #[test]
    fn test_normalize_try_new_rejects_mismatched_lengths() {
        assert!(Normalize::try_new(vec![0.5, 0.5], vec![0.5, 0.5, 0.5]).is_err());
        assert!(Normalize::try_new(vec![], vec![]).is_err());
        assert!(Normalize::try_new(vec![0.5], vec![0.0]).is_err());
        assert!(Normalize::try_new(vec![0.5], vec![0.25]).is_ok());
    }

    #[test]
    fn test_normalize_creation() {
        let normalize = Normalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]);
        assert_eq!(normalize.mean(), &[0.5, 0.5, 0.5]);
        assert_eq!(normalize.std(), &[0.5, 0.5, 0.5]);
        assert_eq!(normalize.name(), "Normalize");
    }

    #[test]
    fn test_normalize_imagenet() {
        let normalize = Normalize::imagenet();
        assert_eq!(normalize.mean(), &[0.485, 0.456, 0.406]);
        assert_eq!(normalize.std(), &[0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_normalize_cifar() {
        let normalize = Normalize::cifar();
        assert_eq!(normalize.mean(), &[0.4914, 0.4822, 0.4465]);
        assert_eq!(normalize.std(), &[0.2023, 0.1994, 0.2010]);
    }

    #[test]
    #[should_panic(expected = "Mean and std must have the same length")]
    fn test_normalize_mismatched_lengths() {
        Normalize::new(vec![0.5, 0.5], vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_normalize_parameters() {
        let normalize = Normalize::new(vec![0.1, 0.2], vec![0.3, 0.4]);
        let params = normalize.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "mean");
        assert_eq!(params[1].0, "std");
    }

    #[test]
    fn test_pad_new() {
        let pad = Pad::new((1, 2, 3, 4), 0.5);
        assert_eq!(pad.padding(), (1, 2, 3, 4));
        assert_eq!(pad.fill(), 0.5);
        assert_eq!(pad.name(), "Pad");
    }

    #[test]
    fn test_pad_symmetric() {
        let pad = Pad::symmetric(5, 1.0);
        assert_eq!(pad.padding(), (5, 5, 5, 5));
        assert_eq!(pad.fill(), 1.0);
    }

    #[test]
    fn test_pad_sides() {
        let pad = Pad::sides(3, 7, 0.25);
        assert_eq!(pad.padding(), (3, 7, 3, 7));
        assert_eq!(pad.fill(), 0.25);
    }

    #[test]
    fn test_pad_parameters() {
        let pad = Pad::new((1, 2, 3, 4), 0.8);
        let params = pad.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "padding");
        assert_eq!(params[0].1, "(1, 2, 3, 4)");
        assert_eq!(params[1].0, "fill");
        assert_eq!(params[1].1, "0.80");
    }

    #[test]
    fn test_clone_transforms() {
        let resize = Resize::new((100, 100));
        let cloned = resize.clone_transform();
        assert_eq!(cloned.name(), "Resize");

        let normalize = Normalize::new(vec![0.1], vec![0.2]);
        let cloned = normalize.clone_transform();
        assert_eq!(cloned.name(), "Normalize");

        let pad = Pad::symmetric(2, 0.0);
        let cloned = pad.clone_transform();
        assert_eq!(cloned.name(), "Pad");
    }
}
