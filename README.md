# Underwater Wave Prediction Project

A deep learning project to predict wave heights using a CNN-LSTM hybrid model on NOAA buoy data (Station 46059).

## Prerequisites

- **Python 3.11** (Required for TensorFlow compatibility)
- **VS Code** (Recommended IDE)

## Setup

1.  **Open the project folder** in VS Code: `d:\projects\wave_project`
2.  **Open a Terminal** in VS Code (Ctrl+` or Terminal > New Terminal).
3.  **Create a Virtual Environment** (if not already done):
    ```powershell
    py -3.11 -m venv .venv
    ```
4.  **Activate the Virtual Environment**:
    ```powershell
    .venv\Scripts\activate
    ```
    (You should see `(.venv)` at the start of your terminal line).

5.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

## Usage

### 1. Download Data
First, download the historical data from NOAA:
```powershell
python download_data.py
```

### 2. Run the Model
To process data, train the model, and generate predictions:
```powershell
python wave_prediction.py
```

## Results

After running the script, check the project folder for:
- `learning_curve.png`: Plot of Training vs Validation Loss.
- `prediction_plot.png`: Plot of Actual vs Predicted Wave Heights.

The script will also print evaluation metrics (RMSE, MAE, R2) to the terminal.
