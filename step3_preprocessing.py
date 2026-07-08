import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from step2_dataset import load_and_process_data
import os

def create_sequences(data, target_col_idx, window_size=10):
    """
    Creates sequences for time-series forecasting.
    X: (samples, window_size, features)
    y: (samples,) - target value at the next time step
    """
    X = []
    y = []
    
    # data is a numpy array
    for i in range(len(data) - window_size):
        # Input sequence: from i to i+window_size
        X.append(data[i:i+window_size])
        
        # Target: value at i+window_size (next step)
        # target_col_idx is the index of the target column
        y.append(data[i+window_size, target_col_idx])
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    FILEPATH = os.path.join('data', '46059h2023.txt.gz')
    
    # 1. Load Data
    df = load_and_process_data(FILEPATH)
    
    # Select features
    features = ['wave_height', 'wind_speed', 'pressure', 'sst']
    data_values = df[features].values
    
    # 2. Split Data (Chronological split to prevent temporal data leakage)
    split_idx = int(len(data_values) * 0.8)
    train_raw, test_raw = data_values[:split_idx], data_values[split_idx:]
    
    # 3. Normalize Data
    print("\nNormalizing data using MinMaxScaler...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)
    
    # 4. Create Sequences
    WINDOW_SIZE = 10
    # Target is wave_height, which is at index 0
    TARGET_COL_IDX = 0
    
    print(f"Creating sequences with window size {WINDOW_SIZE}...")
    X_train, y_train = create_sequences(train_scaled, TARGET_COL_IDX, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, TARGET_COL_IDX, WINDOW_SIZE)
    
    # 5. Print Shapes
    print("\nData Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 6. Explain Dimensions
    print("\nExplanation:")
    print(f"(samples, time_steps, features) -> ({X_train.shape[0]}, {X_train.shape[1]}, {X_train.shape[2]})")
    print("samples: Number of data points available for training/testing.")
    print("time_steps: The sequence length (how far back the model looks).")
    print("features: Number of input variables (wave_height, wind_speed, pressure, sst).")

