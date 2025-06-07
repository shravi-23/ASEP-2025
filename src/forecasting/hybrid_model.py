import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class HybridForecastingModel:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.lstm_model = self._build_lstm()
        self.arima_model = None
        self.arima_order = None
        self.last_values = None
        
    def _build_lstm(self):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _prepare_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
        
    def fit(self, data: pd.Series, arima_order: Tuple[int, int, int] = (2, 1, 2)) -> None:
        """Fit both LSTM and ARIMA models"""
        # Store last values for prediction
        self.last_values = data.values[-self.sequence_length:]
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Train LSTM
        X, y = self._prepare_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.lstm_model.fit(X, y, epochs=50, verbose=0)
        
        # Train ARIMA
        self.arima_order = arima_order
        self.arima_model = ARIMA(data, order=arima_order)
        self.arima_model = self.arima_model.fit()
        
    def predict(self, steps: int = 12) -> np.ndarray:
        """Generate hybrid predictions"""
        # LSTM prediction
        last_sequence = self.lstm_model.predict(
            np.array([self.last_values]).reshape(1, self.sequence_length, 1)
        )
        lstm_pred = last_sequence[0]
        
        # ARIMA prediction
        arima_pred = self.arima_model.forecast(steps=steps)
        
        # Combine predictions (simple average)
        hybrid_pred = (lstm_pred + arima_pred) / 2.0
        return hybrid_pred
        
    def evaluate(self, test_data: pd.Series) -> float:
        """Calculate RMSE on test data"""
        predictions = self.predict(len(test_data))
        rmse = np.sqrt(np.mean((predictions - test_data.values) ** 2))
        return rmse 