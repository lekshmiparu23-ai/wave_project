import pandas as pd
import numpy as np

def load_and_process_data(filepath):
    print(f"Loading data from {filepath}...")
    
    # Load data
    # stored as .txt.gz
    # sep is whitespace
    # Row 0 is header, Row 1 is units (we skip row 1)
    df = pd.read_csv(filepath, compression='gzip', sep='\s+', skiprows=[1])
    
    # 4. Rename columns
    rename_map = {
        'WVHT': 'wave_height',
        'WSPD': 'wind_speed',
        'PRES': 'pressure',
        'WTMP': 'sst'  # SST is water temperature
    }
    
    # Check if columns exist before renaming to avoid errors
    # Note: Column names in file might have '#' prefix on first column
    df.rename(columns={'#YY': 'YY'}, inplace=True)
    
    # Filter columns that exist
    available_cols = [c for c in rename_map.keys() if c in df.columns]
    df.rename(columns=rename_map, inplace=True)
    
    print("Columns renamed.")
    
    # 5. Combine date columns
    # NDBC has YYYY MM DD hh mm
    df['date'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H-%M')
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Keep only relevant columns
    features = ['wave_height', 'wind_speed', 'pressure', 'sst']
    df = df[features]
    
    # 6. Handle missing values
    # NDBC uses 99.0, 999.0, 9999.0 as missing values
    # We replace them based on standard NDBC missing value codes or simple logic
    # WVHT: 99.00
    # WSPD: 99.00
    # PRES: 9999.0
    # WTMP: 999.0
    
    df['wave_height'] = df['wave_height'].replace(99.00, np.nan)
    df['wind_speed'] = df['wind_speed'].replace(99.00, np.nan)
    df['pressure'] = df['pressure'].replace(9999.0, np.nan)
    df['sst'] = df['sst'].replace(999.0, np.nan)
    
    # Interpolate to fill missing values (linear interpolation)
    df.interpolate(method='time', inplace=True)
    
    # Drop any remaining NaNs at the start/end
    df.dropna(inplace=True)
    
    print("Missing values handled.")
    
    return df

if __name__ == "__main__":
    import os
    FILEPATH = os.path.join('data', '46059h2023.txt.gz')
    
    if os.path.exists(FILEPATH):
        df = load_and_process_data(FILEPATH)
        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print(f"File not found: {FILEPATH}")
