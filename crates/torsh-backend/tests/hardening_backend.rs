//! Regression tests for the Wave-3 backend hardening pass.
//!
//! These cover the compilable fixes: real dilated/grouped CPU convolution
//! (F090/F193), non-panicking `Clone` for the generic kernel/buffer handles
//! (F085), and the honest pure-Rust CUDA fallback (F002/F191).

use torsh_backend::convolution::algorithms::DirectConvolution;
use torsh_backend::{BufferHandle, KernelHandle};

/// Plain 2x2-over-3x3 convolution, groups=1, dilation=1: sanity baseline.
#[test]
fn conv2d_direct_plain_is_correct() {
    // 3x3 input.
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // 2x2 kernel = [[1,0],[0,1]] (diagonal).
    let kernel = vec![1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0; 4];

    DirectConvolution::conv2d_direct(
        &input,
        &kernel,
        &mut output,
        &[1, 1, 3, 3],
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        (1, 1),
        (0, 0),
        (1, 1),
        1,
    )
    .expect("plain conv should succeed");

    // out[i,j] = in[i,j] + in[i+1,j+1]
    assert_eq!(output, vec![6.0, 8.0, 12.0, 14.0]);
}

/// Dilation must space the kernel taps apart (F193). With dilation (2,2) a 2x2
/// all-ones kernel over a 3x3 input samples exactly the four corners.
#[test]
fn conv2d_direct_honours_dilation() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let kernel = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 ones
    let mut output = vec![0.0; 1];

    DirectConvolution::conv2d_direct(
        &input,
        &kernel,
        &mut output,
        &[1, 1, 3, 3],
        &[1, 1, 2, 2],
        &[1, 1, 1, 1], // effective extent 3x3 -> single output
        (1, 1),
        (0, 0),
        (2, 2),
        1,
    )
    .expect("dilated conv should succeed");

    // corners: 1 + 3 + 7 + 9
    assert_eq!(output, vec![20.0]);

    // A dilation-1 run over the same buffers would instead sum the top-left
    // 2x2 block (1+2+4+5 = 12), proving dilation is not silently ignored.
    let mut dense = vec![0.0; 4];
    DirectConvolution::conv2d_direct(
        &input,
        &kernel,
        &mut dense,
        &[1, 1, 3, 3],
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        (1, 1),
        (0, 0),
        (1, 1),
        1,
    )
    .expect("dense conv should succeed");
    assert_eq!(dense[0], 12.0);
    assert_ne!(dense[0], output[0]);
}

/// Grouped convolution must keep each output channel confined to its group's
/// input channels (F193). Two groups, one channel each, 1x1 kernels.
#[test]
fn conv2d_direct_honours_groups() {
    // N=1, C_in=2, H=1, W=2. Channel 0 = [1,2], channel 1 = [10,20].
    let input = vec![1.0, 2.0, 10.0, 20.0];
    // C_out=2, C_in/groups=1, 1x1 kernels: oc0 weight 2, oc1 weight 3.
    let kernel = vec![2.0, 3.0];
    let mut output = vec![0.0; 4];

    DirectConvolution::conv2d_direct(
        &input,
        &kernel,
        &mut output,
        &[1, 2, 1, 2],
        &[2, 1, 1, 1],
        &[1, 2, 1, 2],
        (1, 1),
        (0, 0),
        (1, 1),
        2,
    )
    .expect("grouped conv should succeed");

    // oc0 sees only channel 0: [1*2, 2*2]; oc1 sees only channel 1: [10*3, 20*3].
    assert_eq!(output, vec![2.0, 4.0, 30.0, 60.0]);
}

/// Depthwise convolution is grouped convolution with groups == in_channels.
#[test]
fn conv2d_direct_depthwise_via_groups() {
    // N=1, C=2, 2x2 spatial. Channel 0 all 1s, channel 1 all 2s.
    let input = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
    // Per-channel 2x2 all-ones kernels: [C_out=2, 1, 2, 2].
    let kernel = vec![1.0; 8];
    let mut output = vec![0.0; 2];

    DirectConvolution::conv2d_direct(
        &input,
        &kernel,
        &mut output,
        &[1, 2, 2, 2],
        &[2, 1, 2, 2],
        &[1, 2, 1, 1],
        (1, 1),
        (0, 0),
        (1, 1),
        2, // groups == channels -> depthwise
    )
    .expect("depthwise conv should succeed");

    // channel 0: sum of four 1s = 4; channel 1: sum of four 2s = 8.
    assert_eq!(output, vec![4.0, 8.0]);
}

/// Invalid group counts must error rather than silently compute nonsense.
#[test]
fn conv2d_direct_rejects_bad_groups() {
    let input = vec![0.0; 4];
    let kernel = vec![0.0; 2];
    let mut output = vec![0.0; 2];

    let err = DirectConvolution::conv2d_direct(
        &input,
        &kernel,
        &mut output,
        &[1, 2, 1, 2],
        &[2, 1, 1, 1],
        &[1, 2, 1, 2],
        (1, 1),
        (0, 0),
        (1, 1),
        3, // 3 does not divide in_channels = 2
    );
    assert!(
        err.is_err(),
        "groups that do not divide channels must error"
    );
}

/// An undersized buffer must produce an error, never an out-of-bounds panic
/// from inside the kernel (Rule 4: no new panic paths).
#[test]
fn conv2d_direct_rejects_undersized_buffers() {
    let input = vec![0.0; 3]; // needs 3*3 = 9 for a 3x3 input
    let kernel = vec![0.0; 4];
    let mut output = vec![0.0; 4];

    let err = DirectConvolution::conv2d_direct(
        &input,
        &kernel,
        &mut output,
        &[1, 1, 3, 3],
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        (1, 1),
        (0, 0),
        (1, 1),
        1,
    );
    assert!(
        err.is_err(),
        "undersized input buffer must error, not panic"
    );
}

/// A `Generic` kernel handle must clone without panicking (F085).
#[test]
fn generic_kernel_handle_clones_without_panic() {
    let handle = KernelHandle::Generic {
        handle: std::sync::Arc::new(1234u32),
    };
    let cloned = handle.clone();
    assert!(matches!(cloned, KernelHandle::Generic { .. }));
}

/// A `Generic` buffer handle must clone without panicking (F085).
#[test]
fn generic_buffer_handle_clones_without_panic() {
    let handle = BufferHandle::Generic {
        handle: std::sync::Arc::new([0u8; 16]),
        size: 16,
    };
    let cloned = handle.clone();
    assert_eq!(cloned.size(), 16);
    assert!(matches!(cloned, BufferHandle::Generic { .. }));
}

/// torsh-backend's CUDA surface is an honest pure-Rust fallback: it never
/// claims a device is available and never fabricates success (F002/F191).
#[cfg(feature = "cuda")]
#[test]
fn cuda_fallback_is_honest() {
    assert!(!torsh_backend::cuda::is_available());
    assert_eq!(
        torsh_backend::cuda::device_count().expect("device_count is infallible"),
        0
    );
    assert!(torsh_backend::cuda::init().is_err());
    assert!(torsh_backend::cuda::current_device().is_err());
}
