//! Error functions
//!
//! This module used to carry a second, hand-rolled implementation of the error
//! and Fresnel functions. It was measurably wrong — the Fresnel small-argument
//! series had a factor-of-two error in `S`, both branches negated an already
//! signed value so `C(-x)`/`S(-x)` came back with the wrong sign, and the
//! `erfcx` asymptotic returned 0.705 at `x = 1` against a true value of 0.4276
//! (a 65% error).
//!
//! There is now exactly one implementation path: the `scirs2-special` backed
//! functions in [`crate::scirs2_integration`], which this module re-exports so
//! existing callers keep working.

pub use crate::scirs2_integration::{erf, erfc, erfcx, erfinv, fresnel, fresnel_c, fresnel_s};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TorshResult;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;
    use torsh_tensor::Tensor;

    #[test]
    fn test_erf() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;
        let result = erf(&x)?;
        let data = result.data()?;

        // Known values: erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) ≈ -0.8427
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(data[1], 0.8427, epsilon = 1e-3);
        assert_relative_eq!(data[2], -0.8427, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_erfc() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;
        let result = erfc(&x)?;
        let data = result.data()?;

        // Known values: erfc(0) = 1, erfc(1) ≈ 0.1573, erfc(-1) ≈ 1.8427
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(data[1], 0.1573, epsilon = 1e-3);
        assert_relative_eq!(data[2], 1.8427, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_erfcx() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, 2.0], vec![3], device)?;
        let result = erfcx(&x)?;
        let data = result.data()?;

        // erfcx(0) = 1, erfcx(1) = 0.4275836, erfcx(2) = 0.2553956
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(data[1], 0.427_583_6, epsilon = 1e-4);
        assert_relative_eq!(data[2], 0.255_395_6, epsilon = 1e-4);
        Ok(())
    }

    #[test]
    fn test_erfinv() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 0.5, -0.5], vec![3], device)?;
        let result = erfinv(&x)?;
        let data = result.data()?;

        // erfinv(0) = 0, erfinv should be antisymmetric
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-4);
        assert!(data[1] > 0.0);
        assert!(data[2] < 0.0);
        assert_relative_eq!(data[1], -data[2], epsilon = 1e-4);
        Ok(())
    }

    #[test]
    fn test_fresnel_integrals() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;

        let s_result = fresnel_s(&x)?;
        let c_result = fresnel_c(&x)?;

        let s_data = s_result.data()?;
        let c_data = c_result.data()?;

        // Fresnel integrals at x=0 should be 0
        assert_relative_eq!(s_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(c_data[0], 0.0, epsilon = 1e-6);

        // S(1) = 0.4382591, C(1) = 0.7798934, and both are odd functions
        assert_relative_eq!(s_data[1], 0.438_259_1, epsilon = 1e-5);
        assert_relative_eq!(c_data[1], 0.779_893_4, epsilon = 1e-5);
        assert_relative_eq!(s_data[1], -s_data[2], epsilon = 1e-5);
        assert_relative_eq!(c_data[1], -c_data[2], epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_erf_erfc_complement() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.5, 1.0, 1.5], vec![3], device)?;

        let erf_result = erf(&x)?;
        let erfc_result = erfc(&x)?;

        let erf_data = erf_result.data()?;
        let erfc_data = erfc_result.data()?;

        // Test that erf(x) + erfc(x) = 1
        for i in 0..erf_data.len() {
            assert_relative_eq!(erf_data[i] + erfc_data[i], 1.0, epsilon = 1e-6);
        }
        Ok(())
    }
}
