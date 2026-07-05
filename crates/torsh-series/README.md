# torsh-series

Time series analysis and forecasting components for ToRSh - powered by SciRS2.

## Overview

This crate provides comprehensive time series analysis, forecasting, and modeling capabilities with PyTorch-compatible neural network integration. It leverages `scirs2-series` for classical time series methods while enabling deep learning approaches through ToRSh's neural network modules.

## Features

- **Forecasting Models**: ARIMA, SARIMA, Exponential Smoothing
- **Neural Forecasting**: LSTM, GRU, Temporal CNNs, and Transformer forward/forecast
  passes (training (`fit()`) is not implemented yet — see Neural Forecasting Models below).
  There is no Prophet, N-BEATS, or DeepAR implementation in this crate.
- **Decomposition**: STL, classical additive/multiplicative, EMD/EEMD, VMD, SSA
- **Anomaly Detection**: Isolation forest, statistical detector, LSTM-based detector
- **State Space Models**: Kalman, Extended Kalman, Unscented Kalman, particle filters,
  Dynamic Linear Models (no Hidden Markov Model implementation)
- **Feature Engineering**: Lag/difference/interaction features, rolling statistics,
  trend/seasonality/spectral features (no Fourier-features or calendar-features helper)
- **Changepoint Detection**: PELT, Binary Segmentation, window-based (CUSUM is a
  window statistic option, not a standalone Bayesian method; no BOCD implementation)
- **Frequency Analysis**: FFT/IFFT, PSD (periodogram/Welch/multitaper), coherence
- **Utilities**: Differencing, Box-Cox, detrending, stationarity tests (ADF, KPSS)

There is no GPU/CUDA acceleration in this crate.

## Usage

> Several examples below use illustrative helpers (`load_ts(...)`, `generate_seasonal_data(...)`,
> `plot_decomposition(...)`) to keep the snippets short. These are not real functions in this
> crate — build a `TimeSeries` directly (see the first example) from your own data instead.

### Basic Time Series Operations

```rust
use torsh_series::prelude::*;
use torsh_tensor::prelude::*;

// Create time series data
let data = tensor![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
let timestamps = tensor![0, 1, 2, 3, 4, 5, 6, 7];

// Create TimeSeries object
let ts = TimeSeries::new(data, Some(timestamps), None)?;

println!("Length: {}", ts.len());
println!("Start: {:?}, End: {:?}", ts.start(), ts.end());

// Basic statistics
println!("Mean: {:.4}", ts.mean()?);
println!("Std: {:.4}", ts.std()?);
```

### Time Series Decomposition

#### STL Decomposition (Seasonal-Trend decomposition using Loess)

```rust
use torsh_series::decomposition::*;

// Perform STL decomposition (real type is `STLDecomposition`, builder only
// exposes `.robust(bool)` — no seasonal_window/trend_window/degree setters)
let stl = STLDecomposition::new(/* period */ 12).robust(false);
let result = stl.fit(&data)?;
```

#### Classical Decomposition

```rust
use torsh_series::decomposition::*;

// Additive / multiplicative decomposition are separate types, not a single
// `seasonal_decompose(data, period, mode)` free function
let decomp_add = AdditiveDecomposition::new(/* period */ 12);
let decomp_mul = MultiplicativeDecomposition::new(/* period */ 12);
```

### ARIMA Models

```rust
use torsh_series::forecast::*;

let data = load_ts("co2_levels.csv")?;

// Auto ARIMA - automatically selects best (p,d,q) parameters
let auto_model = AutoARIMA::new()
    .seasonal(true)
    .m(12)  // Seasonal period
    .max_p(5)
    .max_q(5)
    .max_d(2)
    .information_criterion("aic");

let model = auto_model.fit(&data)?;
println!("Selected model: ARIMA{:?}", model.order());

// Forecast next 24 steps
let forecast = model.predict(24, None)?;
let conf_int = model.predict_interval(24, 0.95)?;  // 95% confidence interval

// Manual ARIMA(2,1,1)
let arima = ARIMA::new(2, 1, 1)?;
arima.fit(&data)?;

let forecast = arima.predict(12, None)?;
```

### SARIMA (Seasonal ARIMA)

```rust
use torsh_series::forecast::*;

let data = load_ts("monthly_sales.csv")?;

// SARIMA(1,1,1)(1,1,1,12)
// (p,d,q) = non-seasonal parameters
// (P,D,Q,m) = seasonal parameters (m=12 for monthly data)
let sarima = SARIMA::new(1, 1, 1, 1, 1, 1, 12)?;

sarima.fit(&data)?;

// Forecast with confidence intervals
let forecast = sarima.predict(24, None)?;
let (lower, upper) = sarima.predict_interval(24, 0.95)?;

// Model diagnostics
let residuals = sarima.residuals()?;
let aic = sarima.aic()?;
let bic = sarima.bic()?;

println!("AIC: {:.4}, BIC: {:.4}", aic, bic);

// Check residuals for white noise
let ljung_box = ljung_box_test(&residuals, 10)?;
println!("Ljung-Box p-value: {:.4}", ljung_box.p_value);
```

### Exponential Smoothing

```rust
use torsh_series::forecast::*;

let data = load_ts("demand.csv")?;

// Simple Exponential Smoothing
let ses = SimpleExpSmoothing::new(0.3)?;  // alpha=0.3
ses.fit(&data)?;
let forecast = ses.predict(12)?;

// Holt's Linear Trend
let holt = Holt::new(0.8, 0.2)?;  // alpha=0.8, beta=0.2
holt.fit(&data)?;
let forecast = holt.predict(12)?;

// Holt-Winters (Triple Exponential Smoothing)
let hw = HoltWinters::new()
    .seasonal_periods(12)
    .trend("add")
    .seasonal("add")
    .alpha(0.8)
    .beta(0.2)
    .gamma(0.3);

hw.fit(&data)?;
let forecast = hw.predict(24)?;
```

Note: there is no `AutoETS` (automatic Error-Trend-Seasonal selection) in this crate today.

### Neural Forecasting Models

**Status**: `LSTMForecaster`, `GRUForecaster`, `CNNForecaster`, and `TransformerForecaster`
(in `torsh_series::forecast::deep`) have real forward passes — the transformer block
implements genuine scaled dot-product self-attention with residual connections. **Training
is not wired up yet**: only `LSTMForecaster` has a `fit()` method at all, and it is
currently a no-op stub (every parameter is ignored; see `forecast/deep.rs`).
`GRUForecaster`/`CNNForecaster`/`TransformerForecaster` have no `fit()` method at all today.
There is no `NBEATS` or `DeepAR` implementation in this crate. Forecasting uses
`forecast(series, steps)`, not `predict(...)`, and none of these types use a builder
pattern — constructor arguments are positional.

```rust
use torsh_series::forecast::neural::*;

let data: TimeSeries = /* ... */;

// LSTM — forward pass and forecast() are real; fit() does NOT actually train yet
let mut lstm = LSTMForecaster::new(/* input_size */ 1, /* hidden_size */ 64, /* num_layers */ 2)?
    .with_dropout(0.2);
lstm.fit(&data, 100, 0.001); // currently a no-op — does not train the model
let forecast = lstm.forecast(&data, 10)?;

// GRU / CNN ("TCN") / Transformer forecasters — forward + forecast() only, no fit() yet
let gru = GRUForecaster::new(1, 64, 2)?;
let cnn = CNNForecaster::new(vec![1, 32, 32, 32], vec![3, 3, 3])?;
let transformer = TransformerForecaster::new(/* d_model */ 64, /* nhead */ 8, /* num_layers */ 3)?;
```

### Anomaly Detection

```rust
use torsh_series::anomaly::*;

let data = load_ts("sensor_data.csv")?;

// Statistical anomaly detection (real type is `StatisticalDetector`, not
// `StatisticalAnomalyDetector`; constructor takes an `AnomalyMethod`, not a builder chain)
let detector = StatisticalDetector::new(method);
let anomalies = detector.detect(&data)?;

// Isolation Forest (new() takes no args; defaults to 100 estimators, 0.1 contamination)
let iforest = IsolationForest::new();

// LSTM-based anomaly detection (real type is `LSTMAnomaly`, not `LSTMAutoencoder`)
let ae = LSTMAnomaly::new(/* sequence_length */ 50, /* hidden_size */ 16);
```

Note: there is no `SeasonalESD` (Seasonal Hybrid ESD) implementation in this crate.

### Changepoint Detection

```rust
use torsh_series::changepoint::*;

let data = load_ts("regime_changes.csv")?;

// PELT (Pruned Exact Linear Time)
let pelt = PELT::new()
    .model("rbf")
    .min_size(2)
    .jump(1)
    .penalty(Some(3.0));

let changepoints = pelt.detect(&data)?;
println!("Detected changepoints at: {:?}", changepoints);

// Binary Segmentation (constructor takes a threshold, not a builder chain)
let binseg = BinarySegmentation::new(/* threshold */ 5.0);
let changepoints = binseg.detect(&data)?;

// CUSUM is a `WindowStatistic` option on `WindowDetector`, not its own type,
// and there is no Bayesian Online Changepoint Detection in this crate.
let window_detector = WindowDetector::new(/* window_size */ 10, /* threshold */ 5.0)
    .with_statistic(WindowStatistic::CUSUM);
let changepoints = window_detector.detect(&data)?;
```

### State Space Models

#### Kalman Filter

```rust
use torsh_series::state_space::*;

// Linear Kalman Filter (constructor takes dimensions, not a builder chain of
// matrices — the matrices/covariances are set via separate setter methods)
let kf = KalmanFilter::new(/* state_dim */ 4, /* obs_dim */ 2);
```

#### Particle Filter

```rust
use torsh_series::state_space::*;

// Particle Filter for nonlinear/non-Gaussian systems
let pf = ParticleFilter::new(/* num_particles */ 1000, /* state_dim */ 4);
```

There is also an `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, and `DynamicLinearModel`
(Bayesian state space with discount factors) in `torsh_series::state_space`. There is no
Hidden Markov Model (HMM) implementation in this crate.

### Frequency Domain Analysis

```rust
use torsh_series::frequency::*;

// Fast Fourier Transform — analyzer object, not a free `fft()` function
let fft_analyzer = FFTAnalyzer::new(/* sampling_rate */ 1.0);
let fft_result = fft_analyzer.fft(&data)?;
let power_spectrum = fft_result.power();

// Dominant frequency via PeriodogramAnalyzer
let periodogram_analyzer = PeriodogramAnalyzer::new(1.0);
let dominant_freq = periodogram_analyzer.dominant_frequency(&data)?;

// Power spectral density (Welch's method is one `PSDMethod` option)
let psd_estimator = PSDEstimator::new(PSDMethod::Welch, 1.0);
let psd_result = psd_estimator.estimate(&data)?;
```

Note: there is no `find_dominant_frequencies()`, `WaveletTransform`, or `spectrogram()` in
`torsh_series::frequency` today. Wavelet decomposition lives in `torsh_series::decomposition`
instead.

### Feature Engineering

```rust
use torsh_series::utils::features::*;

// Lag features (real name is create_lag_features, module is utils::features)
let lag_features = create_lag_features(&data, &[1, 7, 30])?;

// Rolling statistics — one call returns mean/std/min/max/median together
let rolling = rolling_statistics(&data, /* window_size */ 7, None);

// Difference, trend, seasonality, spectral, and autocorrelation features are
// also available: create_difference_features, trend_features,
// seasonality_features, spectral_features, statistical_features, autocorrelation.
```

Note: there is no expanding-window (`expanding_mean`/`expanding_std`), exponentially
weighted (`ewm_mean`/`ewm_std`), `TimeSeriesFeatures`, `fourier_features()`, or
`calendar_features()` API in this crate today.

### Stationarity and Transformations

```rust
use torsh_series::utils::statistical_tests::*;
use torsh_series::utils::preprocessing::*;

// Test for stationarity (real name has a `_test` suffix)
let adf_result = augmented_dickey_fuller_test(&data, "c", None)?;

// KPSS test
let kpss_result = kpss_test(&data, "c", None)?;

// Differencing (order-based; no separate periods/seasonal variant)
let diff1 = diff(&data, 1);
let diff2 = diff(&data, 2); // second-order differencing

// Box-Cox transformation — lambda must be supplied (not estimated automatically)
let transformed = box_cox(&data, 0.5);
let original = inv_box_cox(&transformed, 0.5);

// Detrending
let detrended = detrend(&data, "linear");
```

### Model Selection and Validation

```rust
use torsh_series::utils::validation::*;

// The real cross-validation type is `TimeSeriesCV` (there is also
// `PurgedTimeSeriesCV`, `CombinatorialPurgedCV`, `NestedTimeSeriesCV`,
// `ScoredTimeSeriesCV`, and `BlockedTimeSeriesCV`). There is no plain
// `train_test_split()` or `ExpandingWindowSplit` type — expanding-window and
// walk-forward evaluation are free functions instead:
let tscv = TimeSeriesCV::new(/* n_splits */ 5);

let errors = walk_forward_validation(&data, /* window_size */ 100, /* step_size */ 10, |train| {
    // ... fit a model on `train` and return a single prediction ...
    0.0
})?;
```

Note: there is no `grid_search_arima()` hyperparameter search helper in this crate.

### Evaluation Metrics

```rust
use torsh_series::utils::metrics::*;

let y_true: TimeSeries = test_data;
let y_pred: TimeSeries = forecasts;

// Real function names are shorter and return a plain f64 (not Result<f64>)
let mae_score = mae(&y_true, &y_pred);
let rmse_score = rmse(&y_true, &y_pred);
let mape_score = mape(&y_true, &y_pred);
let smape_score = smape(&y_true, &y_pred);
let mase_score = mase(&y_true, &y_pred, /* seasonal_period */ 1);

// Also available: r2, theil_u, directional_accuracy, max_error, evaluate_forecast.
// There is no forecast_bias() or tracking_signal() in this crate.
```

## Integration with SciRS2

This crate leverages the SciRS2 ecosystem for:

- Classical time series methods through `scirs2-series`
- Signal processing via `scirs2-signal`
- Statistical functions from `scirs2-stats`
- Neural network components via ToRSh modules
- Optimized tensor operations through `scirs2-core`

All implementations follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for consistent APIs and optimal performance.

## Examples

There is currently no `examples/` directory in this crate; see the doc comments and the
`tests/` under each module (e.g. `src/forecast/`, `src/anomaly.rs`, `src/state_space/`)
for runnable usage patterns.

## Performance Tips

1. **Apply differencing** to make series stationary before ARIMA
2. **Cache decomposition results** when used multiple times

Note: there is no GPU acceleration, and neural forecaster training (`fit()`) is not
implemented yet (see Neural Forecasting Models above), so "normalize before training" and
"parallelize cross-validation" tips do not yet apply in practice.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
