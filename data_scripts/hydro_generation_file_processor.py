import glob
import os
import logging
import pandas as pd

def process_hydro_generation_file(resampling_interval='H'):
    """Function to process BPA hydro generation data from external calls and save resampled reslts to CSV."""
    if resampling_interval not in ('H', 'D'):
        logging.error(f"Resampling interval does not belong to [H, D]")
        return
        
    input_path = '../bpa_data'
    output_path = '../training_data'
    usace_file = os.path.join(output_path, "usace_columbia_projects_data.csv")
    
    usace_df = _load_usace_data(usace_file)
    file_list = sorted(glob.glob(os.path.join(input_path, "*.xlsx")))
    if not file_list:
        logging.warning("No hydro generation files found.")
        return
    
    excl_list = []
    for file in file_list:
        logging.info(f"Processing file: {file}")
        df_sampled = _process_hydro_data(file, usace_df, resampling_interval)
        if df_sampled is not None:
            excl_list.append(df_sampled)
    
    if excl_list:
        excl_merged = pd.concat(excl_list, ignore_index=False)
        output_file = os.path.join(output_path, f'columbia_hydro_generation_data_{resampling_interval}.csv')
        excl_merged.to_csv(output_file, index=True, header=True)
        logging.info(f"Processed data saved to {output_file}")
    else:
        logging.warning("No valid data processed.")
        
        
def _load_usace_data(filepath):
    """Load and preprocess USACE scaling ratio data."""
    try:
        usace_df = pd.read_csv(filepath, usecols=["Year", "Month", "Scaling_Ratio"], dtype={"Year": str, "Month": str})
        usace_df["Date"] = pd.to_datetime(usace_df["Year"] + "-" + usace_df["Month"] + "-01")
        return usace_df
    except Exception as e:
        raise ValueError(f"Error loading USACE data: {e}")
    

def _process_hydro_data(file_path, usace_df, resampling_interval):
    """Process an individual BPA hydro generation file and return a cleaned DataFrame including Columbia river generation data."""
    try:
        df = pd.read_excel(file_path, header=None, skiprows=2, usecols=[0, 6])
        df.columns = ["Datetime", "Total_Hydro_Generation_MW"]
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%m/%d/%y %H:%M")
        df["Total_Hydro_Generation_MW"] = df["Total_Hydro_Generation_MW"].round(1)
        df["Total_Hydro_Generation_MWh"] = (df["Total_Hydro_Generation_MW"] * (5 / 60)).round(1)
        
        df.set_index("Datetime", inplace=True)
        df_sampled = df.resample(resampling_interval).agg({
            "Total_Hydro_Generation_MW": "mean",  # Use mean for power (MW)
            "Total_Hydro_Generation_MWh": "sum"   # Use sum for energy (MWh)
        }).reset_index()
        
        df_sampled["Total_Hydro_Generation_MW"] = df_sampled["Total_Hydro_Generation_MW"].ffill().round(1)
        df_sampled["Date"] = df_sampled["Datetime"].dt.to_period("M").dt.to_timestamp()
        
        df_sampled = df_sampled.merge(usace_df[["Date", "Scaling_Ratio"]], on="Date", how="left")
        df_sampled["Scaling_Ratio"] = df_sampled["Scaling_Ratio"].ffill()
        
        df_sampled["Columbia_Projects_Hydro_Generation_MW"] = (df_sampled["Total_Hydro_Generation_MW"] * df_sampled["Scaling_Ratio"]).round(1)
        df_sampled["Columbia_Projects_Hydro_Generation_MWh"] = (df_sampled["Total_Hydro_Generation_MWh"] * df_sampled["Scaling_Ratio"]).round(1)
        
        df_sampled.set_index("Datetime", inplace=True)
        df_sampled.drop(columns=["Date"], inplace=True)
        
        return df_sampled
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None
        
        
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
