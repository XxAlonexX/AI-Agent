import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def add_technical_indicators(self, df):
        """Add various technical indicators to the dataframe."""
        # Extract price data from multi-index DataFrame
        df = df.droplevel('Ticker', axis=1)
        
        # Ensure we have the right data format
        print("Data types:", df.dtypes)
        print("Close price sample:", df['Close'].head())
        
        # Convert to numpy array and ensure it's float64
        close = df['Close'].astype(float).values
        high = df['High'].astype(float).values
        low = df['Low'].astype(float).values
        volume = df['Volume'].astype(float).values
        
        # Price-based indicators
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['EMA_20'] = talib.EMA(close, timeperiod=20)
        
        # Momentum indicators
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(close)
        
        # Volatility indicators
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = talib.BBANDS(close)
        
        # Volume indicators
        df['OBV'] = talib.OBV(close, volume)
        df['ADI'] = talib.AD(high, low, close, volume)
        
        # Trend indicators
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        
        return df
    
    def calculate_returns(self, df):
        """Calculate various return metrics."""
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=20).std()
        return df
    
    def prepare_data(self, df, sequence_length=60):
        """Prepare data for LSTM model."""
        print(f"Initial dataframe shape: {df.shape}")
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        df = self.calculate_returns(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        print(f"Shape after adding indicators and dropping NaN: {df.shape}")
        
        # Select features for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal',
            'ATR', 'Bollinger_Upper', 'Bollinger_Lower',
            'OBV', 'ADX', 'Daily_Return', 'Rolling_Volatility'
        ]
        print(f"Number of features: {len(feature_columns)}")
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        print(f"Scaled data shape: {scaled_data.shape}")
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, feature_columns.index('Close')])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        print(f"X shape after sequence creation: {X.shape}")
        print(f"y shape after sequence creation: {y.shape}")
        
        # Split into training and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        print(f"Final shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        
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
