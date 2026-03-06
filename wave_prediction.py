import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os

# Import helper functions from previous steps
from step2_dataset import load_and_process_data
from step3_preprocessing import create_sequences

# -------------------------------------------------------
# STEP 4: CNN–LSTM MODEL
# -------------------------------------------------------
def build_cnn_lstm_model(input_shape):
    """
    Builds a CNN-LSTM hybrid model.
    """
    model = Sequential([
        # 1. CNN Layer (Feature Extraction)
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        # 2. MaxPooling (Downsampling)
        MaxPooling1D(pool_size=2),
        # 3. LSTM Layer (Temporal Dependency)
        LSTM(50, activation='relu'),
        # 4. Dense Output Layer
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def main():
    # -------------------------------------------------------
    # DATA PREPARATION (Recap of Steps 1-3)
    # -------------------------------------------------------
    FILEPATH = os.path.join('data', '46059h2023.txt.gz')
    if not os.path.exists(FILEPATH):
        print("Data file not found. Please run download_data.py first.")
        return

    # Load and clean data
    df = load_and_process_data(FILEPATH)
    
    # Select features
    features = ['wave_height', 'wind_speed', 'pressure', 'sst']
    data_values = df[features].values
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # Create sequences
    WINDOW_SIZE = 10
    TARGET_COL_IDX = 0 # wave_height
    X, y = create_sequences(data_scaled, TARGET_COL_IDX, WINDOW_SIZE)
    
    print(f"\nData Shapes: X={X.shape}, y={y.shape}")
    
    # -------------------------------------------------------
    # STEP 5: TRAINING
    # -------------------------------------------------------
    # 1. Split data 80/20 (Time-series split, no shuffling)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Define Input Shape for Model (time_steps, features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build Model
    model_cnn_lstm = build_cnn_lstm_model(input_shape)
    
    # 4. Show model summary
    print("\nCNN-LSTM Model Summary:")
    model_cnn_lstm.summary()
    
    # 2. Train model
    print("\nStarting training CNN-LSTM...")
    history_cnn_lstm = model_cnn_lstm.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # 3. Plot training vs validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history_cnn_lstm.history['loss'], label='CNN-LSTM Train Loss')
    plt.plot(history_cnn_lstm.history['val_loss'], label='CNN-LSTM Val Loss', linestyle='--')
    plt.title('CNN-LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('learning_curve.png')
    print("Saved learning_curve.png")

    # -------------------------------------------------------
    # STEP 6: EVALUATION
    # -------------------------------------------------------
    # 1. Predict on test data
    y_pred_cnn_lstm = model_cnn_lstm.predict(X_test)
    
    def inverse_transform_target(y_scaled, scaler, target_idx, n_features):
        dummy = np.zeros((len(y_scaled), n_features))
        dummy[:, target_idx] = y_scaled.flatten()
        return scaler.inverse_transform(dummy)[:, target_idx]
    
    y_test_inv = inverse_transform_target(y_test, scaler, TARGET_COL_IDX, data_scaled.shape[1])
    y_pred_cnn_lstm_inv = inverse_transform_target(y_pred_cnn_lstm, scaler, TARGET_COL_IDX, data_scaled.shape[1])
    
    # 2. Calculate Metrics
    def get_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    m_cnn_lstm = get_metrics(y_test_inv, y_pred_cnn_lstm_inv)
    
    # 3. Print metrics
    print("\n-------------------------------------------------------")
    print("EVALUATION METRICS (CNN-LSTM)")
    print("-------------------------------------------------------")
    print(f"{'Metric':<10} | {'Value':<12}")
    print("-" * 25)
    print(f"{'RMSE':<10} | {m_cnn_lstm[0]:<12.4f}")
    print(f"{'MAE':<10} | {m_cnn_lstm[1]:<12.4f}")
    print(f"{'R2':<10} | {m_cnn_lstm[2]:<12.4f}")
    
    # -------------------------------------------------------
    # SHOW EXAMPLES
    # -------------------------------------------------------
    test_start_idx = split_idx + WINDOW_SIZE
    test_dates = df.index[test_start_idx : test_start_idx + len(y_test)]
    
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test_inv.flatten(),
        'CNN-LSTM': y_pred_cnn_lstm_inv.flatten()
    })
    
    print("\n-------------------------------------------------------")
    print("ACTUAL VS PREDICTED SAMPLES (First 10)")
    print("-------------------------------------------------------")
    print(results_df.head(10).to_string(index=False))
    
    # 4. Plot Actual vs Predicted Comparison
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inv, label='Actual', color='black', alpha=0.5, linewidth=2)
    plt.plot(y_pred_cnn_lstm_inv, label='CNN-LSTM Prediction', linestyle='--')
    plt.title('Wave Height Prediction: Actual vs CNN-LSTM')
    plt.xlabel('Time Steps')
    plt.ylabel('Wave Height (m)')
    plt.legend()
    plt.savefig('prediction_plot.png')
    print("Saved prediction_plot.png")
    
    # -------------------------------------------------------
    # STEP 7: RESEARCH OUTPUT
    # -------------------------------------------------------
    print("\n-------------------------------------------------------")
    print("RESEARCH SUMMARY")
    print("-------------------------------------------------------")
    print(f"CNN-LSTM Model: RMSE={m_cnn_lstm[0]:.4f}, R2={m_cnn_lstm[2]:.4f}")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
