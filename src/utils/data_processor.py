import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df, sequence_length):
        """Prepare data for LSTM model"""
        # Calculate technical indicators
        df = self.add_technical_indicators(df)
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(df)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, sequence_length)
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for time series prediction"""
        X = []
        y = []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 0])  # Predicting the close price
            
        return np.array(X), np.array(y)
