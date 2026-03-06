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
    
    # 2. Normalize Data
    print("\nNormalizing data using MinMaxScaler...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # 3. Create Sequences
    WINDOW_SIZE = 10
    # Target is wave_height, which is at index 0
    TARGET_COL_IDX = 0
    
    print(f"Creating sequences with window size {WINDOW_SIZE}...")
    X, y = create_sequences(data_scaled, TARGET_COL_IDX, WINDOW_SIZE)
    
    # 4. Print Shapes
    print("\nData Shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # 5. Explain Dimensions
    print("\nExplanation:")
    print(f"(samples, time_steps, features) -> ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
    print("samples: Number of data points available for training/testing.")
    print("time_steps: The sequence length (how far back the model looks).")
    print("features: Number of input variables (wave_height, wind_speed, pressure, sst).")
