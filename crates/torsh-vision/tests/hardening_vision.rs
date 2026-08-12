//! Regression tests for the torsh-vision production-hardening campaign.
//!
//! Every test in this file corresponds to an audit finding (F-id in the name).

use scirs2_core::ndarray::{arr1, arr2, Array1};
use torsh_tensor::Tensor;
use torsh_vision::feature_detection_advanced::{AttentionMatcher, AttentionMatcherConfig, Feature};
use torsh_vision::ops;
use torsh_vision::spatial::interpolation::{ImageWarper, InterpolationConfig, SpatialInterpolator};
use torsh_vision::spatial::transforms::{GeometricProcessor, InterpolationMethod};
use torsh_vision::transforms::{ToTensor, Transform};

/// F283: bilinear upsampling must clamp at the top/left borders instead of
/// linearly extrapolating past them.
#[test]
fn f283_bilinear_upscale_never_extrapolates() {
    let input = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[1, 2, 2])
        .expect("input tensor creation should succeed");
    let out = ops::resize(&input, (8, 8)).expect("resize should succeed");

    for y in 0..8 {
        for x in 0..8 {
            let v = out.get(&[0, y, x]).expect("get should succeed");
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&v),
                "resize produced {v} at ({y},{x}), outside the input range [0, 1]"
            );
        }
    }
}

/// F284: `random_crop` must be able to select the bottom-right-most origin.
#[test]
fn f284_random_crop_reaches_bottom_right_origin() {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let img = Tensor::from_vec(data, &[1, 4, 4]).expect("tensor creation should succeed");

    let mut seen_bottom_right = false;
    for _ in 0..1000 {
        let crop = ops::random_crop(&img, (3, 3)).expect("random_crop should succeed");
        let top_left = crop.to_vec().expect("to_vec should succeed")[0];
        if (top_left - 5.0).abs() < 1e-6 {
            seen_bottom_right = true;
            break;
        }
    }

    assert!(
        seen_bottom_right,
        "random_crop never selected the bottom-right-most valid origin (1, 1)"
    );
}

/// F287: `resize` must handle 4D tensors with batch_size > 1.
#[test]
fn f287_resize_supports_batches_larger_than_one() {
    let mut data = vec![0.0f32; 2 * 1 * 4 * 4];
    for v in data.iter_mut().skip(16) {
        *v = 1.0;
    }
    let batch = Tensor::from_vec(data, &[2, 1, 4, 4]).expect("tensor creation should succeed");

    let out = ops::resize(&batch, (2, 2)).expect("batched resize should succeed");
    assert_eq!(out.shape().dims(), &[2, 1, 2, 2]);

    let out_data = out.to_vec().expect("to_vec should succeed");
    for (i, v) in out_data.iter().enumerate() {
        let expected = if i < 4 { 0.0 } else { 1.0 };
        assert!(
            (v - expected).abs() < 1e-5,
            "batched resize mixed up images: index {i} is {v}, expected {expected}"
        );
    }
}

/// F286: `ToTensor` must reorder HWC -> CHW and scale by 1/255.
#[test]
fn f286_to_tensor_converts_hwc_and_scales() {
    let data: Vec<f32> = (0..12).map(|i| (i as f32) * 20.0).collect();
    let img = Tensor::from_vec(data.clone(), &[2, 2, 3]).expect("tensor creation should succeed");

    let out = ToTensor::new()
        .forward(&img)
        .expect("ToTensor should succeed");

    assert_eq!(
        out.shape().dims(),
        &[3, 2, 2],
        "ToTensor must produce CHW output"
    );

    // (row 0, col 0) -> data[0..3], (row 1, col 1) -> data[9..12]
    for c in 0..3 {
        let got = out.get(&[c, 0, 0]).expect("get should succeed");
        let want = data[c] / 255.0;
        assert!(
            (got - want).abs() < 1e-6,
            "channel {c} at (0,0): got {got}, want {want}"
        );

        let got = out.get(&[c, 1, 1]).expect("get should succeed");
        let want = data[9 + c] / 255.0;
        assert!(
            (got - want).abs() < 1e-6,
            "channel {c} at (1,1): got {got}, want {want}"
        );
    }
}

/// F282 (pre-fix proof): the infallible builder aborts on an invalid parameter.
#[test]
fn f282_color_jitter_new_still_panics_on_invalid_brightness() {
    use torsh_vision::transforms::ColorJitter;

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(|| {
        let _ = ColorJitter::new().brightness(-1.0);
    });
    std::panic::set_hook(previous);

    assert!(
        outcome.is_err(),
        "expected the infallible builder to panic on a negative brightness"
    );
}

/// F285: a translation affine transform must actually move image content.
#[test]
fn f285_affine_transform_is_not_identity() {
    let processor = GeometricProcessor::new(InterpolationMethod::Bilinear);
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let img = Tensor::from_vec(data.clone(), &[1, 4, 4]).expect("tensor creation should succeed");

    // Translate by (+1, +1).
    let matrix = arr2(&[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]);
    let out = processor
        .apply_affine_transform(&img, &matrix)
        .expect("affine transform should succeed");

    let out_data = out.to_vec().expect("to_vec should succeed");
    assert_ne!(
        out_data, data,
        "apply_affine_transform returned the input unchanged"
    );
}

/// F285: warping with a non-zero displacement field must change the image.
#[test]
fn f285_warp_image_is_not_identity() {
    let warper = ImageWarper::new(InterpolationConfig::default());
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let img = Tensor::from_vec(data.clone(), &[1, 4, 4]).expect("tensor creation should succeed");

    // Displacement field: one (dx, dy) row per pixel.
    let field = scirs2_core::ndarray::Array2::from_elem((16, 2), 1.0);
    let out = warper
        .warp_image(&img, &field)
        .expect("warp_image should succeed");

    let out_data = out.to_vec().expect("to_vec should succeed");
    assert_ne!(out_data, data, "warp_image returned the input unchanged");
}

/// F285: barrel-distortion correction with non-zero coefficients must change the image.
#[test]
fn f285_barrel_distortion_is_not_identity() {
    let warper = ImageWarper::new(InterpolationConfig::default());
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let img = Tensor::from_vec(data.clone(), &[1, 8, 8]).expect("tensor creation should succeed");

    let coeffs: Array1<f64> = arr1(&[0.2, 0.05]);
    let out = warper
        .correct_barrel_distortion(&img, &coeffs)
        .expect("barrel correction should succeed");

    let out_data = out.to_vec().expect("to_vec should succeed");
    assert_ne!(
        out_data, data,
        "correct_barrel_distortion returned the input unchanged"
    );
}

/// F285: super-resolution must actually upscale.
#[test]
fn f285_super_resolution_upscales() {
    let interpolator = SpatialInterpolator::new(InterpolationConfig::default());
    let img = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[1, 2, 2])
        .expect("tensor creation should succeed");

    let out = interpolator
        .super_resolution(&img, 2.0)
        .expect("super_resolution should succeed");

    assert_eq!(
        out.shape().dims(),
        &[1, 4, 4],
        "super_resolution did not upscale the image"
    );
}

/// F281: `num_heads` must actually influence attention-based matching.
#[test]
fn f281_attention_matcher_uses_num_heads() {
    fn make(seed: f32, n: usize, dim: usize) -> Vec<Feature> {
        (0..n)
            .map(|i| {
                let descriptor = Array1::from_shape_fn(dim, |k| {
                    ((i as f32 + seed) * 0.37 + k as f32 * 0.11).sin()
                });
                Feature {
                    x: i as f32,
                    y: 0.0,
                    response: 1.0,
                    descriptor,
                    scale: 1.0,
                    orientation: 0.0,
                }
            })
            .collect()
    }

    let f1 = make(0.0, 6, 8);
    let f2 = make(0.5, 6, 8);

    let one_head = AttentionMatcher::new(AttentionMatcherConfig {
        num_heads: 1,
        match_threshold: 0.0,
        mutual_match: false,
        ..Default::default()
    })
    .expect("matcher creation should succeed");
    let many_heads = AttentionMatcher::new(AttentionMatcherConfig {
        num_heads: 4,
        match_threshold: 0.0,
        mutual_match: false,
        ..Default::default()
    })
    .expect("matcher creation should succeed");

    let m1 = one_head
        .match_features(&f1, &f2)
        .expect("matching should succeed");
    let m4 = many_heads
        .match_features(&f1, &f2)
        .expect("matching should succeed");

    let confidences1: Vec<f32> = m1.iter().map(|m| m.confidence).collect();
    let confidences4: Vec<f32> = m4.iter().map(|m| m.confidence).collect();

    assert_ne!(
        confidences1, confidences4,
        "num_heads has no effect on AttentionMatcher results"
    );
}

/// F176: LayerNorm2d must reject non-4D input with an error instead of panicking.
#[test]
fn f176_layer_norm_2d_rejects_non_4d_input() {
    use torsh_nn::Module;
    use torsh_vision::models::advanced_cnns::LayerNorm2d;

    let norm = LayerNorm2d::new(4).expect("LayerNorm2d creation should succeed");
    let input =
        Tensor::from_vec(vec![0.0f32; 12], &[4, 3]).expect("tensor creation should succeed");

    let result = norm.forward(&input);
    assert!(
        result.is_err(),
        "LayerNorm2d accepted a 2D input instead of returning an error"
    );
}

/// F176: LayerNorm2d must reject empty spatial dimensions instead of emitting NaN.
#[test]
fn f176_layer_norm_2d_rejects_empty_spatial_dims() {
    use torsh_nn::Module;
    use torsh_vision::models::advanced_cnns::LayerNorm2d;

    let norm = LayerNorm2d::new(2).expect("LayerNorm2d creation should succeed");
    let input =
        Tensor::from_vec(Vec::<f32>::new(), &[1, 2, 0, 4]).expect("tensor creation should succeed");

    let result = norm.forward(&input);
    assert!(
        result.is_err(),
        "LayerNorm2d accepted an empty spatial extent instead of returning an error"
    );
}

/// F176: LayerNorm2d must keep the autograd graph alive.
#[test]
fn f176_layer_norm_2d_preserves_autograd() {
    use torsh_nn::Module;
    use torsh_vision::models::advanced_cnns::LayerNorm2d;

    let norm = LayerNorm2d::new(2).expect("LayerNorm2d creation should succeed");
    let input = Tensor::from_vec((0..16).map(|i| i as f32).collect(), &[1, 2, 2, 4])
        .expect("tensor creation should succeed")
        .requires_grad_(true);

    let out = norm.forward(&input).expect("forward should succeed");
    assert!(
        out.requires_grad(),
        "LayerNorm2d severed the autograd graph"
    );
}

/// F282: fallible constructors validate user-supplied augmentation parameters.
#[test]
fn f282_fallible_constructors_reject_invalid_parameters() {
    use torsh_vision::transforms::{
        AutoAugment, ColorJitter, CutMix, Cutout, GaussianBlur, MixUp, RandAugment, RandomErasing,
        RandomHorizontalFlip, RandomResizedCrop, RandomRotation, RandomVerticalFlip,
    };

    assert!(ColorJitter::new().try_brightness(-1.0).is_err());
    assert!(ColorJitter::new().try_contrast(-0.5).is_err());
    assert!(ColorJitter::new().try_saturation(-0.5).is_err());
    assert!(ColorJitter::new().try_hue(0.9).is_err());
    assert!(ColorJitter::new().try_brightness(0.2).is_ok());

    assert!(GaussianBlur::try_new(4, 1.0).is_err());
    assert!(GaussianBlur::try_new(3, 0.0).is_err());
    assert!(GaussianBlur::try_new(3, 1.0).is_ok());

    assert!(RandomErasing::try_new(1.5).is_err());
    assert!(RandomErasing::try_new(0.5)
        .and_then(|t| t.try_with_scale((0.8, 0.2)))
        .is_err());
    assert!(RandomErasing::try_new(0.5)
        .and_then(|t| t.try_with_ratio((2.0, 0.5)))
        .is_err());

    assert!(Cutout::try_new(0, 1).is_err());
    assert!(Cutout::try_new(16, 0).is_err());
    assert!(Cutout::try_new(16, 2).is_ok());

    assert!(RandomHorizontalFlip::try_new(-0.1).is_err());
    assert!(RandomVerticalFlip::try_new(1.1).is_err());
    assert!(RandomRotation::try_new((10.0, -10.0)).is_err());
    assert!(RandomResizedCrop::new((224, 224))
        .try_with_scale((0.5, 0.2))
        .is_err());
    assert!(RandomResizedCrop::new((224, 224))
        .try_with_ratio((2.0, 0.5))
        .is_err());

    assert!(MixUp::try_new(-1.0).is_err());
    assert!(MixUp::try_new(0.2).is_ok());
    assert!(CutMix::try_new(-1.0).is_err());

    assert!(RandAugment::try_new(0, 5.0).is_err());
    assert!(RandAugment::try_new(2, 20.0).is_err());
    assert!(RandAugment::try_new(2, 5.0).is_ok());
    assert!(RandAugment::try_with_transforms(2, 5.0, vec![]).is_err());

    assert!(AutoAugment::try_with_policies(vec![]).is_err());
    assert!(AutoAugment::try_with_policies(vec![vec![]]).is_err());
    assert!(AutoAugment::try_with_policies(vec![vec![("rotate".to_string(), 2.0)]]).is_err());
    assert!(AutoAugment::try_with_policies(vec![vec![("rotate".to_string(), 0.5)]]).is_ok());
}

/// F285: hue adjustment must actually rotate colours.
#[test]
fn f285_adjust_hue_changes_colours() {
    use torsh_vision::AdvancedTransforms;

    let transforms = AdvancedTransforms::auto_detect().expect("hardware detection should succeed");
    // 2x2 interleaved RGB image with saturated colours.
    let data = vec![
        1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    ];
    let img = Tensor::from_vec(data.clone(), &[2, 2, 3]).expect("tensor creation should succeed");

    let out = transforms
        .random_hue(&img, (0.3, 0.31))
        .expect("random_hue should succeed");

    let out_data = out.to_vec().expect("to_vec should succeed");
    assert_eq!(out.shape().dims(), &[2, 2, 3]);
    assert_ne!(out_data, data, "adjust_hue returned the input unchanged");
}

/// F285: rectify_image must apply the homography.
#[test]
fn f285_rectify_image_applies_homography() {
    let processor = GeometricProcessor::new(InterpolationMethod::Bilinear);
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let img = Tensor::from_vec(data.clone(), &[1, 4, 4]).expect("tensor creation should succeed");

    // Pure translation expressed as a homography.
    let homography = arr2(&[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let out = processor
        .rectify_image(&img, &homography)
        .expect("rectify_image should succeed");

    let out_data = out.to_vec().expect("to_vec should succeed");
    assert_ne!(out_data, data, "rectify_image returned the input unchanged");
}

/// F285: gap interpolation must fill masked-out pixels.
#[test]
fn f285_interpolate_image_gaps_fills_holes() {
    let interpolator = SpatialInterpolator::new(InterpolationConfig::default());

    let mut data = vec![1.0f32; 16];
    data[5] = 999.0; // The hole, marked unknown below.
    let img = Tensor::from_vec(data, &[1, 4, 4]).expect("tensor creation should succeed");

    let mut mask = vec![1.0f32; 16];
    mask[5] = 0.0;
    let mask = Tensor::from_vec(mask, &[4, 4]).expect("tensor creation should succeed");

    let out = interpolator
        .interpolate_image_gaps(&img, &mask)
        .expect("inpainting should succeed");

    let filled = out.to_vec().expect("to_vec should succeed")[5];
    assert!(
        (filled - 1.0).abs() < 1e-4,
        "masked pixel was not reconstructed from its neighbours, got {filled}"
    );
}

/// F285: scattered-data interpolation methods that are not implemented must say so.
#[test]
fn f285_unimplemented_scattered_methods_return_errors() {
    use torsh_vision::spatial::interpolation::InterpolationMethod as ScatteredMethod;

    let points = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
    let values = arr1(&[0.0, 1.0]);
    let grid = arr2(&[[0.5, 0.5]]);

    for method in [
        ScatteredMethod::NaturalNeighbor,
        ScatteredMethod::Bilinear,
        ScatteredMethod::Bicubic,
    ] {
        let interpolator = SpatialInterpolator::new(InterpolationConfig {
            method,
            ..Default::default()
        });
        assert!(
            interpolator
                .interpolate_to_grid(&points, &values, &grid)
                .is_err(),
            "unimplemented scattered interpolation must return an error, not zeros"
        );
    }

    // RBF is implemented and must reproduce the samples it was fitted on.
    let rbf = SpatialInterpolator::new(InterpolationConfig {
        method: ScatteredMethod::RadialBasisFunction,
        ..Default::default()
    });
    let fitted = rbf
        .interpolate_to_grid(&points, &values, &points)
        .expect("RBF interpolation should succeed");
    assert!((fitted[0] - 0.0).abs() < 1e-6 && (fitted[1] - 1.0).abs() < 1e-6);
}
