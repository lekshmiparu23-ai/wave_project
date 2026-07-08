import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reuse load_and_process_data and create_sequences from project modules
from step2_dataset import load_and_process_data
from step3_preprocessing import create_sequences

np.random.seed(42)
tf.random.set_seed(42)

def inverse_transform_target(scaled_y, scaler):
    """
    Helper function to inverse-transform wave height (target at index 0)
    using the 4-feature scaler.
    """
    dummy = np.zeros((len(scaled_y), 4))
    dummy[:, 0] = scaled_y.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

def batch_rollout(model, batch_windows, horizon):
    """
    Performs recursive multi-step rollout in parallel for a batch of starting windows.
    Keeps wind_speed, pressure, sst held constant at their last known real values.
    """
    current_windows = batch_windows.copy()
    for _ in range(horizon):
        # Predict next step (in scaled space)
        preds = model(current_windows, training=False).numpy()[:, 0]
        preds = np.clip(preds, 0.0, 1.0)
        
        # Prepare the new rows to append: [pred_wave_height, last_wind, last_pressure, last_sst]
        new_rows = np.zeros((len(preds), 1, 4))
        new_rows[:, 0, 0] = preds
        new_rows[:, 0, 1] = current_windows[:, -1, 1]
        new_rows[:, 0, 2] = current_windows[:, -1, 2]
        new_rows[:, 0, 3] = current_windows[:, -1, 3]
        
        # Shift window and concatenate new row
        current_windows = np.concatenate([current_windows[:, 1:, :], new_rows], axis=1)
        
    return preds

def main():
    # 1. Paths and parameters
    model_path = os.path.join('models', 'wave_model.keras')
    scaler_path = os.path.join('models', 'scaler.pkl')
    data_path = os.path.join('data', '46059h2023.txt.gz')
    WINDOW_SIZE = 10
    TARGET_COL_IDX = 0
    
    # 2. Load model and scaler
    print("Loading model and scaler...")
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    
    # 3. Load and split data (exact same split as wave_prediction.py)
    print("Loading and preparing test data...")
    df = load_and_process_data(data_path)
    features = ['wave_height', 'wind_speed', 'pressure', 'sst']
    data_values = df[features].values
    
    split_idx = int(len(data_values) * 0.8)
    test_raw = data_values[split_idx:]
    
    # Transform test set using the fitted scaler
    test_scaled = scaler.transform(test_raw)
    
    horizons = [1, 3, 6, 12, 24]
    results = []
    
    print("\nEvaluating horizons...")
    for H in horizons:
        # Determine number of valid starting points for horizon H
        num_samples = len(test_scaled) - WINDOW_SIZE - H + 1
        
        # Gather all starting windows of shape (num_samples, WINDOW_SIZE, 4)
        batch_windows = np.array([test_scaled[i : i + WINDOW_SIZE] for i in range(num_samples)])
        
        # Baseline: last known wave height in starting window (index WINDOW_SIZE - 1)
        baseline_scaled = np.array([test_scaled[i + WINDOW_SIZE - 1, 0] for i in range(num_samples)])
        
        # Actual target value H steps after lookback window
        actual_scaled = np.array([test_scaled[i + WINDOW_SIZE - 1 + H, 0] for i in range(num_samples)])
        
        # Model predictions via rollout
        preds_scaled = batch_rollout(model, batch_windows, H)
        
        # Inverse-transform to physical units (meters)
        actual_inv = inverse_transform_target(actual_scaled, scaler)
        pred_inv = inverse_transform_target(preds_scaled, scaler)
        baseline_inv = inverse_transform_target(baseline_scaled, scaler)
        
        # Compute metrics
        model_rmse = np.sqrt(mean_squared_error(actual_inv, pred_inv))
        model_mae = mean_absolute_error(actual_inv, pred_inv)
        
        baseline_rmse = np.sqrt(mean_squared_error(actual_inv, baseline_inv))
        baseline_mae = mean_absolute_error(actual_inv, baseline_inv)
        
        skill_score = 1.0 - (model_rmse / baseline_rmse)
        
        results.append({
            'Horizon (hours)': H,
            'Model RMSE': model_rmse,
            'Persistence RMSE': baseline_rmse,
            'Model MAE': model_mae,
            'Persistence MAE': baseline_mae,
            'Skill Score': skill_score
        })
    
    # 4. Create summary dataframe
    summary_df = pd.DataFrame(results)
    
    # Format the numeric values for display
    display_df = summary_df.copy()
    for col in display_df.columns:
        if col != 'Horizon (hours)':
            display_df[col] = display_df[col].map('{:.4f}'.format)
            
    print("\n" + "="*85)
    print("MULTI-HORIZON SKILL SCORE EVALUATION:")
    print(display_df.to_string(index=False))
    print("="*85)
    
    # 5. Save results to models/horizon_comparison.csv
    os.makedirs('models', exist_ok=True)
    csv_save_path = os.path.join('models', 'horizon_comparison.csv')
    summary_df.to_csv(csv_save_path, index=False)
    print(f"\nResults successfully saved to: {csv_save_path}")

if __name__ == '__main__':
    main()
