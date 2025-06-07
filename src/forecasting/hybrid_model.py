import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

# Suppress warnings
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class HybridForecastingModel:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.lstm_model = LSTMModel()
        self.arima_model = None
        self.arima_order = None
        self.last_values = None
        
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
        X = torch.FloatTensor(X.reshape((X.shape[0], X.shape[1], 1)))
        y = torch.FloatTensor(y)
        
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        criterion = nn.MSELoss()
        
        self.lstm_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.lstm_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Train ARIMA
        self.arima_order = arima_order
        self.arima_model = ARIMA(data, order=arima_order)
        self.arima_model = self.arima_model.fit()
        
    def predict(self, steps: int = 12) -> np.ndarray:
        """Generate hybrid predictions"""
        # LSTM prediction
        self.lstm_model.eval()
        with torch.no_grad():
            last_sequence = torch.FloatTensor(self.last_values).reshape(1, self.sequence_length, 1)
            lstm_pred = self.lstm_model(last_sequence)
            lstm_pred = lstm_pred.numpy()
        
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