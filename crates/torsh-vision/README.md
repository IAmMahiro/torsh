# torsh-vision

Computer vision utilities and models for ToRSh, leveraging scirs2-vision for optimized image operations.

## Overview

This crate provides comprehensive computer vision functionality:

- **Image I/O**: Loading, saving, and format conversion
- **Transforms**: Data augmentation and preprocessing
- **Datasets**: Common vision datasets (ImageNet, COCO, etc.)
- **Models**: Pre-trained vision models
- **Operations**: Image processing and computer vision algorithms
- **Visualization**: Image display and annotation utilities

Note: This crate integrates with scirs2-vision for optimized image processing operations.

## Usage

### Image I/O

```rust
use torsh_vision::prelude::*;

// Load images
let image = read_image("path/to/image.jpg")?;
let batch = read_image_batch(&["img1.jpg", "img2.jpg", "img3.jpg"])?;

// Save images
write_image(&tensor, "output.png")?;
write_video(&frames, "output.mp4", fps=30)?;

// Format conversion
let rgb = bgr_to_rgb(&bgr_image)?;
let gray = rgb_to_grayscale(&rgb_image)?;
let hsv = rgb_to_hsv(&rgb_image)?;
```

### Transforms

```rust
use torsh_vision::transforms::*;

// Create transform pipeline
let transform = Compose::new(vec![
    Box::new(Resize::new(256)),
    Box::new(CenterCrop::new(224)),
    Box::new(ToTensor::new()),
    Box::new(Normalize::new(
        vec![0.485, 0.456, 0.406],  // ImageNet mean
        vec![0.229, 0.224, 0.225],  // ImageNet std
    )),
]);

let transformed = transform.apply(&image)?;

// Data augmentation
let augment = Compose::new(vec![
    Box::new(RandomResizedCrop::new(224, scale=(0.08, 1.0))),
    Box::new(RandomHorizontalFlip::new(0.5)),
    Box::new(ColorJitter::new(0.4, 0.4, 0.4, 0.1)),
    Box::new(RandomErasing::new(0.2)),
    Box::new(ToTensor::new()),
    Box::new(Normalize::imagenet()),
]);

// Advanced augmentations
let mixup = MixUp::new(alpha=1.0);
let cutmix = CutMix::new(alpha=1.0);
let augmix = AugMix::new(severity=3, width=3, depth=-1, alpha=1.0);
```

### Datasets

```rust
use torsh_vision::datasets_impl::*;

// ImageNet dataset — NOTE: currently a placeholder. It prints a warning and
// returns a single dummy zero tensor rather than loading real ImageNet data.
let imagenet = ImageNet::new("./data/imagenet", true)?;

// COCO dataset (CocoDataset) — also a placeholder implementation today.
let coco = CocoDataset::new("./data/coco", true)?;

// CIFAR-10 — real loader (parses the actual binary batch files)
let cifar10 = CIFAR10::new("./data", true, true)?;

// Custom folder dataset — real loader, scans subdirectories as classes
let dataset = ImageFolder::new("./data/custom")?;
```

There is currently no `VideoFolder` dataset type; video-specific data loading utilities
live in `torsh_vision::video` (see Video Processing below).

### Pre-trained Models

```rust
use torsh_vision::models::*;

// Classification models. NOTE: `ModelConfig` carries a `pretrained` flag, but
// weight download/loading is not wired up yet — models always initialize
// with fresh, randomly-initialized weights (see TODO.md "Model Zoo").
let resnet = ResNet::resnet50(ModelConfig::default())?;
let efficientnet = EfficientNet::efficientnet_b0(ModelConfig::default())?;
let vit = VisionTransformer::vit_base_patch16_224(ModelConfig::default())?;

// Object detection — real factory functions (YOLOv5 / RetinaNet / SSD).
// There is no Faster R-CNN or DeepLabV3 implementation in this crate yet.
let yolo = yolo_v5_small(80)?;
let retina_net = retina_net_resnet50(80)?;
let ssd = ssd_300(80)?;
```

### Image Operations

```rust
use torsh_vision::ops::*;

// Basic operations (leveraging scirs2-vision)
let resized = resize(&image, (224, 224))?;
let flipped = horizontal_flip(&image)?;
let rotated = rotate(&image, 45.0)?;

// Filtering
let blurred = gaussian_blur(&image, 1.0)?;
let edge = sobel_edge_detection(&image)?;

// Color adjustments
let bright = adjust_brightness(&image, 1.5)?;
let contrast = adjust_contrast(&image, 1.5)?;
let saturated = adjust_saturation(&image, 1.5)?;
```

Note: there is no generic `crop()`, `adjust_sharpness()`, `slic_superpixels()`, or
`dense_optical_flow()` free function today. Cropping is available via the
`CenterCrop`/`RandomCrop` transforms, and Lucas-Kanade optical flow is available
through `torsh_vision::video::OpticalFlow`.

### Object Detection Utilities

```rust
use torsh_vision::{box_iou, nms, generate_anchors};
use torsh_vision::ops::detection::{roi_pool, ROIPoolConfig, AnchorConfig};

// Bounding box IoU (single pair) and NMS
let iou = box_iou(&box1, &box2); // f32, not a batched tensor op
let kept = nms(detections, NMSConfig::default())?;

// Anchor generation is a function, not a builder struct
let anchors = generate_anchors(feature_height, feature_width, AnchorConfig::default())?;

// ROI pooling (there is no roi_align yet)
let pooled = roi_pool(&features, &rois, ROIPoolConfig::default())?;
```

### Visualization

```rust
use torsh_vision::utils::*;

// Draw bounding boxes (mutates `image` in place)
draw_bounding_boxes(&mut image, &boxes, Some(&labels), None, None)?;

// Create image grid (tensors, nrow, padding)
let grid = make_grid(&tensor_list, 8, 2)?;

// Save a tensor as an image (tensor, path, normalize)
save_tensor_as_image(&grid, "visualization.png", true)?;
```

Note: there is currently no `draw_segmentation_masks()` or `draw_keypoints()` helper
(and no `COCO_PERSON_SKELETON` constant) in this crate.

### Video Processing

```rust
use torsh_vision::video::*;

// There is no read_video()/write_video() free function (and no audio
// support) yet — video I/O goes through the VideoReader/VideoWriter
// traits and their Simple* implementations.
let reader = SimpleVideoReader::from_images(&["frame1.png", "frame2.png"], 30.0)?;
let mut writer = SimpleVideoWriter::new("output.mp4", 30.0);

// Apply a frame-wise transform across a video
let video_transform = VideoTransform::new(my_frame_transform);
```

### Feature Extraction and Similarity

There is currently no `create_feature_extractor()`, `cosine_similarity()`, or
`ImageRetrieval` public API in this crate — feature-map extraction is limited to
whatever a given model architecture exposes directly (e.g. intermediate forward
outputs), and there is no built-in CBIR/image-retrieval system yet.

## Integration with SciRS2

This crate leverages scirs2-vision for:
- Optimized image processing operations
- Efficient data augmentation
- Hardware-accelerated transforms
- Computer vision algorithms

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.