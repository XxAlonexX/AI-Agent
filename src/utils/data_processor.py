import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def add_technical_indicators(self, df):
        """Add various technical indicators to the dataframe."""
        # Price-based indicators
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
        
        # Momentum indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
        
        # Volatility indicators
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(df['Close'])
        
        # Volume indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Trend indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        return df
    
    def calculate_returns(self, df):
        """Calculate various return metrics."""
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=20).std()
        return df
    
    def prepare_data(self, df, sequence_length=60):
        """Prepare data for LSTM model."""
        # Add technical indicators
        df = self.add_technical_indicators(df)
        df = self.calculate_returns(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features for training
        feature_columns = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'ATR', 'Bollinger_Upper', 'Bollinger_Lower', 'OBV',
            'ADX', 'Daily_Return'
        ]
        
        # Scale the features
        data = df[feature_columns].values
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 0])  # Predict Close price
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into training and validation sets
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]
        
        print(f"Data shapes after preparation:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_val: {y_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original scale."""
        return self.scaler.inverse_transform(predictions.reshape(-1, 1))
    
    def generate_trading_signals(self, predictions, threshold=0.001):
        """Generate trading signals based on predictions."""
        signals = np.zeros_like(predictions)
        signals[predictions > threshold] = 1    # Buy signal
        signals[predictions < -threshold] = -1   # Sell signal
        return signals
