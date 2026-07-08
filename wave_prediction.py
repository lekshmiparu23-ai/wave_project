"""
.gitignore Content:
__pycache__/
*.pyc
.venv/
wave_model.h5
scaler.pkl
metrics.pkl
data/*.gz
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Import modules from the project
from step2_dataset import load_and_process_data
from step3_preprocessing import create_sequences

def inverse_transform_target(scaled_y, scaler):
    """
    Helper function to inverse-transform wave height (target at index 0)
    using the 4-feature scaler.
    """
    dummy = np.zeros((len(scaled_y), 4))
    dummy[:, 0] = scaled_y.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

def simulate_trajectory(model, last_window, horizon_hours, rmse_scaled):
    """
    Simulates a single forecast trajectory using recursive rollout.
    Features wind_speed, pressure, sst are held constant at their last known values.
    Since short-term weather is persistent, holding context features constant is
    a reasonable assumption over 24 hours.
    """
    current_window = last_window.copy()
    trajectory = []
    for _ in range(horizon_hours):
        # Prepare input shape (1, window_size, features)
        x_in = np.expand_dims(current_window, axis=0)
        
        # Predict next step (in scaled space)
        pred_scaled = model(x_in, training=False).numpy()[0, 0]
        
        # Inject Gaussian noise in scaled space
        noise = np.random.normal(0, rmse_scaled)
        next_val_scaled = pred_scaled + noise
        
        # Clip scaled wave height to [0.0, 1.0] to prevent extrapolation drift
        next_val_scaled = np.clip(next_val_scaled, 0.0, 1.0)
        
        trajectory.append(next_val_scaled)
        
        # Update window: shift and append new step
        # [wave_height, wind_speed, pressure, sst]
        # Keep wind_speed, pressure, sst persistent
        new_row = np.array([
            next_val_scaled,
            current_window[-1, 1],
            current_window[-1, 2],
            current_window[-1, 3]
        ])
        current_window = np.vstack([current_window[1:], new_row])
        
    return np.array(trajectory)

def forecast_multistep(model, last_window, scaler, rmse_scaled, horizon_hours=24, n_simulations=50):
    """
    Performs Monte Carlo simulations of multi-step forecasting.
    """
    all_simulations = []
    
    for _ in range(n_simulations):
        traj = simulate_trajectory(model, last_window, horizon_hours, rmse_scaled)
        all_simulations.append(traj)
        
    all_sims_scaled = np.array(all_simulations) # shape (n_simulations, horizon_hours)
    
    # Inverse transform each simulation to original units (meters)
    all_sims_meters = np.zeros_like(all_sims_scaled)
    for i in range(n_simulations):
        all_sims_meters[i] = inverse_transform_target(all_sims_scaled[i], scaler)
        
    mean_forecast = np.mean(all_sims_meters, axis=0)
    lower_bound = np.percentile(all_sims_meters, 10, axis=0) # 10th percentile
    upper_bound = np.percentile(all_sims_meters, 90, axis=0) # 90th percentile
    
    return mean_forecast, lower_bound, upper_bound, all_sims_meters

def main():
    print("TensorFlow Version:", tf.__version__)
    
    # 1. Load Data
    FILEPATH = os.path.join('data', '46059h2023.txt.gz')
    if not os.path.exists(FILEPATH):
        raise FileNotFoundError(f"Data file not found at {FILEPATH}. Please run download_data.py first.")
        
    df = load_and_process_data(FILEPATH)
    
    # Select features
    features = ['wave_height', 'wind_speed', 'pressure', 'sst']
    data_values = df[features].values
    
    # 2. Split Data (Chronological split to prevent temporal data leakage)
    # FIX: scaler fitted only on train to prevent data leakage
    split_idx = int(len(data_values) * 0.8)
    train_raw, test_raw = data_values[:split_idx], data_values[split_idx:]
    
    # 3. Normalize Data
    print("Normalizing data using MinMaxScaler...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)
    
    # 4. Create Sequences
    # Use 10-hour window as specified by user requirements
    WINDOW_SIZE = 10
    TARGET_COL_IDX = 0  # wave_height is at index 0
    
    print(f"Creating sequences with window size {WINDOW_SIZE}...")
    X_train, y_train = create_sequences(train_scaled, TARGET_COL_IDX, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, TARGET_COL_IDX, WINDOW_SIZE)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # 5. Build CNN-LSTM Model
    print("Building CNN-LSTM model...")
    model_cnn_lstm = Sequential([
        # CNN Feature Extraction
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(WINDOW_SIZE, X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        # LSTM Sequence Learning
        # FIX: removed relu, LSTM default tanh prevents exploding gradients
        LSTM(50),
        # Dense Output Layer
        Dense(1)
    ])
    
    model_cnn_lstm.compile(optimizer='adam', loss='mse')
    model_cnn_lstm.summary()
    
    # 6. Train the Model
    # IMPROVED: EarlyStopping prevents overfitting, restores best weights
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Starting training...")
    history = model_cnn_lstm.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 7. Evaluate the Model
    print("Evaluating model...")
    y_pred_scaled = model_cnn_lstm.predict(X_test)
    
    # Inverse transform to original unit (meters)
    y_test_inv = inverse_transform_target(y_test, scaler)
    y_pred_inv = inverse_transform_target(y_pred_scaled, scaler)
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    m_cnn_lstm = [rmse, mae, r2]
    
    # Calculate persistence baseline metrics on test set
    y_persistence_scaled = X_test[:, -1, TARGET_COL_IDX]
    y_persistence_inv = inverse_transform_target(y_persistence_scaled, scaler)
    p_rmse = np.sqrt(mean_squared_error(y_test_inv, y_persistence_inv))
    p_mae = mean_absolute_error(y_test_inv, y_persistence_inv)
    p_r2 = r2_score(y_test_inv, y_persistence_inv)
    
    # Calculate RMSE in scaled space for Monte Carlo simulations
    rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    
    print("\n" + "="*50)
    print("EVALUATION METRICS COMPARISON:")
    print(f"{'Metric':<10} | {'CNN-LSTM':<12} | {'Persistence Baseline':<20}")
    print("-" * 50)
    print(f"{'RMSE':<10} | {rmse:<12.4f} | {p_rmse:<20.4f}")
    print(f"{'MAE':<10} | {mae:<12.4f} | {p_mae:<20.4f}")
    print(f"{'R²':<10} | {r2:<12.4f} | {p_r2:<20.4f}")
    print("="*50)
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # 8. Plot and Save Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='#00d4d4')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#00f5ff')
    plt.title('CNN-LSTM Model Learning Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'learning_curve.png'), dpi=300)
    plt.close()
    print(f"Learning curve saved to {os.path.join(models_dir, 'learning_curve.png')}.")
    
    # 9. Plot and Save Predictions vs Actuals (Subset of test data for readability)
    plt.figure(figsize=(12, 6))
    subset_len = 150  # Plot first 150 hours of the test set
    plt.plot(y_test_inv[:subset_len], label='Actual Wave Height', color='#ffffff', linewidth=2)
    plt.plot(y_pred_inv[:subset_len], label='Predicted Wave Height', color='#00f5ff', linestyle='--', linewidth=2)
    plt.title('Actual vs Predicted Wave Heights (Test Set - First 150 Hours)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Significant Wave Height (m)', fontsize=12)
    # Style styling (mimics dark ocean theme for matplotlib)
    ax = plt.gca()
    ax.set_facecolor('#0a1628')
    fig = plt.gcf()
    fig.patch.set_facecolor('#0a1628')
    ax.spines['bottom'].set_color('#00d4d4')
    ax.spines['top'].set_color('#00d4d4')
    ax.spines['left'].set_color('#00d4d4')
    ax.spines['right'].set_color('#00d4d4')
    ax.xaxis.label.set_color('#7fffd4')
    ax.yaxis.label.set_color('#7fffd4')
    ax.title.set_color('#00f5ff')
    ax.tick_params(colors='#7fffd4')
    plt.grid(True, linestyle=':', alpha=0.3, color='#00d4d4')
    plt.legend(facecolor='#0d2137', labelcolor='#ffffff', edgecolor='#00f5ff')
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'prediction_plot.png'), dpi=300)
    plt.close()
    print(f"Prediction plot saved to {os.path.join(models_dir, 'prediction_plot.png')}.")
    
    # 10. Save Model, Scaler, and Metrics
    # DEPLOYMENT: saves model + scaler + metrics to models/ for Streamlit app
    model_cnn_lstm.save(os.path.join(models_dir, 'wave_model.keras'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump({
        'rmse': rmse, 
        'mae': mae, 
        'r2': r2,
        'p_rmse': p_rmse,
        'p_mae': p_mae,
        'p_r2': p_r2,
        'rmse_scaled': rmse_scaled
    }, os.path.join(models_dir, 'metrics.pkl'))
    print("Model, scaler, and metrics saved in models/ directory.")

if __name__ == "__main__":
    main()
