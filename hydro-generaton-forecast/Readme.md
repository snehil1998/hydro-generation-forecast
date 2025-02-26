# **Hydropower Generation Forecasting Tool**

## **Overview**
This project aims to forecast Columbia River Hydropower Generation using observed discharge at The Dalles, Oregon. Several models were tested, and **XGBoost** model was selected as the final model due to its ability to handle non-linearity, lagged features, and seasonality effectively.

---
## **Setup and Instructions**
Before running the files and the tool, install the required dependencies using:
```bash
pip install -r requirements.txt
```

Additionally, you can analyze each model using the respective notebooks in the `notebooks` folder. Instructions on running the forecasting tool and data processing scripts are available in `tutorial.ipynb`.

---
## **1. Data Sources & Preprocessing**
### **Datasets Used:**
1. **Bonneville Power Administration (BPA)** - Provides **5-minute MW generation data**, which is resampled for model use.
2. **U.S. Army Corps of Engineers (USACE)** - Provides **monthly power summaries**, used to calculate **scaling ratios** for Columbia River-specific generation.
3. **United States Geological Survey (USGS)** - Provides **15-minute discharge data** at The Dalles, which is resampled for model use.

### **Preprocessing Steps:**
- BPA generation data (5-minute) and USGS discharge data (15-minute) were resampled to hourly (1H) and daily (1D) for consistency and noise reduction.
- Scaling ratios from USACE were applied to ensure BPA data reflected Columbia River-specific hydropower generation. However, there is no exact way to determine Columbia River flow separately, and the scaling ratio approach provides only an approximation.

## **Key Considerations**
### **1️⃣ Data Granularity & Scaling Challenges**
- Discharge and generation data were originally recorded at **different intervals (5-min, 15-min)**.
- Resampled to **1-hour (1H) for XGBoost** and **1-day (1D) for statistical methods**.

### **2️⃣ Timeframe Selection**
- **1D timeframe** used for statistical models due to their simplicity.
- **1H timeframe** used for XGBoost to capture finer seasonal trends.

---
## **2. Model Selection & Experimentation**
### **1️⃣ Linear Regression**
✅ Used as a baseline model.
❌ Required future discharge values, making it impractical for real-world forecasting.

📌 Notebook: `linear_regression.ipynb`

### **2️⃣ Vector Autoregression (VAR)**
✅ Tested for its ability to capture historical dependencies.   
❌ Struggled with variability, producing smooth fitting lines that failed to capture non-linearity. 
❌ Required stationarity transformations (ADF test), making it less flexible.

📌 Notebook: `vector_autoregression.ipynb`

### **3️⃣ XGBoost (Final Model)**
✅ Performed best, handling non-linearity, lagged discharge, and seasonality.
✅ Captured fine-grained seasonal effects using hourly (1H) data.
✅ Included lagged discharge values and seasonality features (hour, day of week, month).
❌ Limited hyperparameter tuning has been done so far.

📌 Notebook: `xgboost.ipynb`

---
## **3. Feature Engineering for XGBoost**
- **Lag Features**: Created discharge lags at **1H, 6H, 12H, 24H, 168H**, etc.
- **Seasonality Features**: Extracted **hour, day of week, month**.

---
## **4. Forecasting Tool Usage**
### **Modes Available:**
1. **Model Evaluation Mode**: Trains the model on a train-test split and evaluates performance.
2. **Forecast Mode**: Uses the entire dataset for training and generates forecasts.
3. **Stored Model Loading**: Enables quick predictions without retraining.
4. **Plots & Visualizations**: Displays hourly and daily generation trends over the forecast period.

📌 Notebook: `tutorial.ipynb`

---
## **5. Potential Improvements**
### ✅ **1. Hyperparameter Optimization and Cross-validation**
- Extensive hyperparameter tuning for XGBoost is needed.
- Implement cross-validation to ensure better model generalization.

### ✅ **2. Training on More Data**
- The current analysis uses only the past two years of data. Expanding the training dataset will help the model better capture fluctuations and long-term trends.

### ✅ **3. Additional Data Sources**
- Include temperature, precipitation, snowmelt, dam operations, etc.
- Use discharge data from other Columbia River projects.

### ✅ **4. Alternative Models**
- Testing more complex algorithms, such as **LSTMs (Long Short-Term Memory networks)** and **Recurrent Neural Networks (RNNs)** for time series forecasting, although they are computationally expensive.

### ✅ **5. Noise Reduction & Data Smoothing**
- Apply smoothening techniques such as rolling averages to handle extreme fluctuations.

### ✅ **6. Platform Enhancements**
- Building an API service and an interactive UI to allow real-time predictions for stakeholders.
- Unit testing for model scripts to ensure reliability.
- Robust monitoring & error handling to detect data anomalies.
- The system currently requires manual data updates for BPA, USACE, and USGS by running scripts. A production system should automate data ingestion through API integrations or scheduled data pulls.

---
## **6. Conclusion**
This project serves as a prototype for an operational hydropower forecasting system. XGBoost outperformed statistical models, successfully capturing the complex relationship between discharge and power generation. Future improvements can focus on hyperparameter tuning, additional external factors, data automation, and advanced time series models.