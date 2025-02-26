import os
import re
import pandas as pd
import datetime
import logging

def process_scaling_ratio_file():
    """Process USACE power generation data including scaling ratio for Columbia river projects and save to CSV."""
    start_year = 2023
    output_file = "../training_data/usace_columbia_projects_data.csv"
    today = datetime.date.today()
    columbia_projects = [
        "Dworshak", "Albeni Falls", "Bonneville", "John Day", "McNary",
        "The Dalles", "Chief Joseph", "Ice Harbor", "Little Goose", "Lower Monumental",
        "Libby", "Lower Granite", "Grand Coulee", "Hungry Horse"
    ]
    
    scaling_ratio = []
    while start_year <= today.year:
        for month in range(1, 13):
            if start_year > today.year or (start_year == today.year and month > today.month):
                df = pd.DataFrame(scaling_ratio, columns=["Year", "Month", "Columbia_Projects_Power_Generation", "Total_Power_Generation", "Scaling_Ratio"])
                df.to_csv(output_file, index=False)
                logging.info(f"Processed data saved to {output_file}")
                return
            
            file_path = f"../usace_data/pwr_{str(start_year)}{str(month).zfill(2)}.txt"
            if not os.path.exists(file_path):
                prev_scaling_ratio = scaling_ratio[-1][-1] if scaling_ratio else None
                scaling_ratio.append([start_year, month, None, None, prev_scaling_ratio])
                logging.warning(f"Missing data file: {file_path}, using previous scaling ratio.")
                continue
            
            try:
                with open(file_path, "r") as file:
                    lines = file.readlines()
                
                columbia_projects_generation = 0
                for row in range(5, len(lines)):
                    columns = re.split(r"\s{2,}", lines[row].strip())
                    if not columns:
                        continue
                    elif columns[0] in columbia_projects:
                        columbia_projects_generation += int(columns[3])
                    elif columns[0] == "TOTAL":
                        scaling_ratio.append([
                            start_year, month, columbia_projects_generation,
                            int(columns[3]), round(columbia_projects_generation / int(columns[3]), 4)
                        ])
                        break
                
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
        
        start_year += 1

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')