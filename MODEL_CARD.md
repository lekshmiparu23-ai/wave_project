# Model Card: CNN-LSTM Wave Height Predictor

This model card documents the features, training setup, evaluation metrics, and operational limitations of the CNN-LSTM Wave Height Prediction model.

---

## Model Overview
The model is a hybrid **CNN-LSTM** architecture designed to predict the significant wave height ($H_s$) at the next hour ($t+1$) based on a sliding window of historical meteorological observations.

- **Architecture:** 
  1. **Conv1D Feature Extractor:** 64 filters, kernel size 2, ReLU activation. Extracts short-term temporal features from the multivariable sequence.
  2. **MaxPooling1D:** Pool size 2. Reduces dimensionality and highlights dominant features.
  3. **LSTM Sequence Learner:** 50 units, tanh activation. Captures long-term sequential dependencies.
  4. **Dense Output:** 1 unit, linear activation. Outputs the single predicted wave height at $t+1$.

---

## Training Details

### Features Used
The model uses 4 features representing meteorology and ocean state, retrieved from NOAA NDBC Buoy Station 46059:
1. **`wave_height` (Hs):** Significant wave height (meters) - *Target variable*
2. **`wind_speed`:** Wind speed (m/s)
3. **`pressure`:** Atmospheric pressure (hPa)
4. **`sst`:** Sea Surface Temperature (°C)

### Configuration
- **Sliding Window Size:** 10 hours ($t-9$ to $t$).
- **Normalization:** Features are scaled using `MinMaxScaler` fitted solely on the training partition.
- **Train/Test Split:** Chronological split (first 80% of chronological observations for training, remaining 20% for testing) with **no shuffling** to prevent temporal data leakage.
- **Optimization:** Adam optimizer, Mean Squared Error (MSE) loss, with EarlyStopping on validation loss (patience 5) restoring the best weights.

---

## Performance Evaluation

To contextualize the model's accuracy, it is compared side-by-side against a trivial **Persistence Baseline**. The baseline predicts that the wave height at $t+1$ is exactly equal to the last observed wave height at $t$.

### Metrics on Test Partition
All metrics are computed in original physical units (meters):

| Metric | CNN-LSTM Model | Persistence Baseline |
| :--- | :--- | :--- |
| **RMSE** (Root Mean Squared Error) | **0.0784 m** | **0.0494 m** |
| **MAE** (Mean Absolute Error) | **0.0510 m** | **0.0343 m** |
| **$R^2$ Score** (Coeff. of Determination) | **0.9967** | **0.9987** |

### Insights from the Baseline
* **High Autocorrelation:** The persistence baseline achieves an $R^2$ of **0.9987** and an RMSE of **0.0494 m**. This indicates that hourly wave height is extremely autocorrelated.
* **Model Value-Add:** The CNN-LSTM model's raw 1-step metrics are slightly higher (RMSE of **0.0784 m**) compared to persistence. This highlights the difficulty in beating simple persistence for hourly steps, making the comparison critical for transparency.

---

## Operational Limitations & Multi-Step Forecasting

> [!WARNING]
> **Single-Step Model Extension:**
> This model was trained **exclusively as a 1-hour-ahead (single-step) predictor**. It does NOT have a native understanding of multi-step sequence horizons.

### Recursive Monte Carlo Rollout
To generate a 24-hour forecast, we employ a **recursive feedback rollout** (`forecast_multistep`):
1. The predicted wave height at $t+1$ is appended to the input window, and the oldest timestep is discarded.
2. Contextual features (`wind_speed`, `pressure`, `sst`) are held persistent at their last known values under the assumption that weather remains stable over short 24-hour periods.
3. This updated window is fed back into the model to predict $t+2$, repeating the process up to $t+24$.
4. **Gaussian Noise Injection:** To simulate error accumulation (compounding drift), Gaussian noise based on the test set prediction error is injected at each recursive step. A 50-run Monte Carlo simulation generates a distribution of trajectories, yielding a mean projection and a **10th–90th percentile uncertainty band**.
5. As expected with recursive rollouts, the uncertainty band **visibly widens** over the 24-hour forecast horizon as prediction errors propagate.
