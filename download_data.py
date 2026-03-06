import requests
import os

def download_data(station_id, year, save_dir='data'):
    """
    Downloads historical standard meteorological data from NOAA NDBC.
    """
    base_url = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
    filename = f"{station_id}h{year}.txt.gz"
    url = f"{base_url}{filename}"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {save_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    # Station 46059: Top-quality station ~300NM west of San Francisco
    STATION_ID = '46059'
    YEAR = '2023'
    download_data(STATION_ID, YEAR)
