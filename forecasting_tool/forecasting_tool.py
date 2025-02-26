from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ForecastingTool:
    def __init__(self, test_size=0.2, enable_evaluation=True):
        if type(test_size) != float or test_size >= 1 or test_size <= 0:
            raise ValueError(f"Invalid test size: {test_size}")
        
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.test_size = test_size
        self.model_path = "../forecasting_tool/hydropower_xgb.pkl"
        self.lags_in_hours = [1, 6, 12, 24, 168, 720, 2160, 8640] # lag intervals between 1 hour to 1 year
        self.generation_data_path = "../training_data/columbia_hydro_generation_data_H.csv"
        self.discharge_data_path = "../training_data/uscs_dalles_discharge_data_H.csv"
        self.enable_evaluation = enable_evaluation
        self.feature_columns = ['Discharge_CFS', 'Hour', 'Dayofweek', 'Month'] + [f"Discharge_Lag_{lag}" for lag in self.lags_in_hours]
        
        logging.info("Loading datasets...")
        df_generation = pd.read_csv(self.generation_data_path, parse_dates=['Datetime'], index_col='Datetime')
        df_discharge = pd.read_csv(self.discharge_data_path, parse_dates=['Datetime'], index_col='Datetime')
        self.df = df_generation.merge(df_discharge, on='Datetime', how='inner')
        

    def train(self, n_estimators=500, learning_rate=0.01, early_stopping_rounds=50):
        """Trains the model using either train-test split (for evaluation) or full data (for forecasting)."""
        if type(n_estimators) != int or type(learning_rate) != float or type(early_stopping_rounds) != int:
            raise ValueError("Invalid parameters. They can only be integers")
        
        if n_estimators <= 0 or learning_rate <= 0 or early_stopping_rounds <= 0:
            raise ValueError("Invalid parameters. They cannot be negative.")
        
        logging.info("Starting training pipeline...")
        
        self.df = self._create_features(self.df)

        X = self.df[self.feature_columns]
        y = self.df['Columbia_Projects_Hydro_Generation_MW']

        if self.enable_evaluation:
            logging.info("Splitting data for evaluation...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)
        else:
            logging.info("Training on full dataset (no test set)...")
            self.X_train, self.y_train = X, y

        logging.info("Training XGBoost model...")
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, early_stopping_rounds=early_stopping_rounds, objective='reg:squarederror')
        verbose = 100
        if not self.enable_evaluation:
            verbose = False
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_train, self.y_train)], verbose=verbose)
        
        self._save_model()

        # Evaluate Model if enabled
        if self.enable_evaluation:
            return self._evaluate_model()
    
    
    def forecast(self, days=7):
        """Loads trained model and predicts for the given number of days."""   
        logging.info(f"Forecasting {days} days into the future...")

        # Load trained model
        self.model = joblib.load(self.model_path)
        logging.info(f"Model loaded from {self.model_path}")

        last_datetime = self.df.index[-1]
        future_steps = days * 24
        future_df = pd.DataFrame(index=pd.date_range(start=last_datetime + timedelta(hours=1), periods=future_steps, freq='H'))
        df_copy = self.df.copy()
        future_df['isFuture'] = True
        df_copy['isFuture'] = False
        df_and_future = pd.concat([df_copy, future_df])
        df_and_future = self._create_features(df_and_future)
        future_with_features = df_and_future.query('isFuture').copy()
        
        future_with_features['Prediction'] = self.model.predict(future_with_features[self.feature_columns])
        logging.info("Forecasting completed.")
        
        self._plot_hourly_forecast(future_with_features)
        self._plot_daily_forecast(future_with_features)
        future_with_features['Prediction'].to_csv(f"../forecasting_tool/forecast{days}.csv", index=True)
        
    
    def _create_features(self, df):
        """Create seasonality and lagged discharge features for training"""
        logging.info("Creating features...")
        df['Hour'] = df.index.hour
        df['Dayofweek'] = df.index.dayofweek
        df['Month'] = df.index.month
        for lag in self.lags_in_hours:
            df[f"Discharge_Lag_{lag}"] = df['Discharge_CFS'].shift(lag)
        return df

    
    def _save_model(self):
        """Save the trained model only if evaluation is disabled (for forecasting)"""
        if not self.enable_evaluation:
            joblib.dump(self.model, self.model_path)
            logging.info(f"Model saved at {self.model_path}")
        else:
            logging.info("Model training completed (not saved for evaluation).")
            

    def _evaluate_model(self):
        """Evaluates the trained model on the test set."""
        logging.info("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        y_pred_df = pd.DataFrame(y_pred, index=self.y_test.index, columns=['Prediction'])
        self._plot_evaluation(y_pred_df)
        return {f"RMSE: {rmse:.2f}, MAE: {mae:.2f}"}
    
    
    def _plot_evaluation(self, y_pred_df):
        """Plots actual vs. predicted hydropower generation data"""
        _, ax = plt.subplots(figsize=(15, 6))
        self.y_test.plot(ax=ax, color="blue", linewidth=2, alpha=0.7, label="Actual Data")
        y_pred_df['Prediction'].plot(ax=ax, color="red", linewidth=2, linestyle="--", alpha=0.9, label="Predicted Data")

        ax.set_title("Actual vs. Predicted Hydropower Generation", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Hydropower Generation (MW)", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    
    def _plot_hourly_forecast(self, future_with_features):
        """Plot hourly forecast vs past 30 days of actual data"""
        _, ax = plt.subplots(figsize=(12, 6))

        self.df['Columbia_Projects_Hydro_Generation_MW'][-720:].plot(ax=ax, color="blue", linewidth=2, label="Actual Data (Last 30 Days)")
        future_with_features['Prediction'].plot(ax=ax, color="orange", linewidth=2, linestyle="--", label="Predicted Future Data")

        ax.set_title("Hydropower Generation Forecast vs. Actual Data", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Hydropower Generation (MW)", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=12)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    
    def _plot_daily_forecast(self, future_with_features):
        """Plot daily forecast"""
        daily_forecast = future_with_features['Prediction'].resample('D').mean()

        _, ax = plt.subplots(figsize=(12, 6))

        daily_forecast.plot(ax=ax, color="orange", linewidth=2, linestyle="--", marker='o', markersize=5, label="Predicted Daily Generation")

        ax.set_title("Daily Forecasted Hydropower Generation", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Hydropower Generation (MW)", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


