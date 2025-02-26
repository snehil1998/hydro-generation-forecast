import datetime
import requests
import os
import logging

def download_latest_bpa_and_usgs_data():
    """Download BPA wind generation and USGS discharge data."""
    start_year = 2023
    year = datetime.date.today().year
    os.makedirs("../bpa_data", exist_ok=True)
    os.makedirs("../uscs_data", exist_ok=True)
    
    bpa_url = f"https://transmission.bpa.gov/Business/Operations/Wind/OPITabularReports/WindGenTotalLoadYTD_{year}.xlsx"
    bpa_output_path = f"../bpa_data/WindGenTotalLoadYTD_{year}.xlsx"
    _download_file(bpa_url, bpa_output_path)
    
    usgs_url = f"https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&cb_00065=on&format=rdb&site_no=14105700&legacy=1&period=&begin_date={start_year}-01-01"
    usgs_output_path = "../uscs_data/discharge_data.txt"
    _download_file(usgs_url, usgs_output_path)
    

def _download_file(url, output_path):
    """Download a file from a given URL and save it locally."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            logging.info(f"File downloaded successfully: {output_path}")
        else:
            logging.error(f"Failed to download file from {url}. Status code: {response.status_code}")
    except Exception as e:
        raise ValueError(f"Error downloading file from {url}: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
