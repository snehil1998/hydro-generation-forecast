import pandas as pd
import logging

def process_discharge_file(resampling_interval='H'):
    """Process USCS The Dalles, Oregon discharge data and save resampled results to CSV."""
    if resampling_interval not in ('H', 'D'):
        logging.error("Resampling interval does not belong to [H, D]")
        return
    
    input_file = "../uscs_data/discharge_data.txt"
    output_file = f"../training_data/uscs_dalles_discharge_data_{resampling_interval}.csv"
    try:
        with open(input_file, "r") as file:
            lines = file.readlines()
        
        discharge_data = []
        for i in range(29, len(lines)):
            columns = lines[i].split('\t')
            try:
                datetime_value = pd.to_datetime(columns[2])
                discharge_value = float(columns[-2])
                discharge_data.append([datetime_value, discharge_value])
            except (IndexError, ValueError) as e:
                continue
        
        df = pd.DataFrame(discharge_data, columns=["Datetime", "Discharge_CFS"])
        df["Discharge_CFS"] = pd.to_numeric(df["Discharge_CFS"], errors='coerce')
        df["Discharge_CFS"] = df["Discharge_CFS"].interpolate(method="linear")
        
        df.set_index("Datetime", inplace=True)
        df = df.resample(resampling_interval).mean()
        df["Discharge_CFS"] = df["Discharge_CFS"].ffill().round(0)
        
        df.to_csv(output_file, index=True)
        logging.info(f"Processed discharge data saved to {output_file}")
    
    except Exception as e:
        raise ValueError(f"Error processing discharge data: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
