import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

class LSTMTrader(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMTrader, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        out = self.fc(context_vector)
        return out

class TradingModelTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
    def train_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(X)
        loss = self.criterion(outputs, y.view(-1, 1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y.view(-1, 1))
        return loss.item()
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()
    
    def evaluate_strategy(self, predictions, actual_returns, initial_capital=100000):
        """Evaluate the trading strategy performance."""
        df = pd.DataFrame({
            'Predicted_Return': predictions.flatten(),
            'Actual_Return': actual_returns
        })
        
        # Generate trading signals
        df['Position'] = np.where(df['Predicted_Return'] > 0, 1, -1)
        
        # Calculate strategy returns
        df['Strategy_Return'] = df['Position'].shift(1) * df['Actual_Return']
        df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        df['Portfolio_Value'] = initial_capital * df['Cumulative_Return']
        
        # Calculate performance metrics
        total_return = (df['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital
        sharpe_ratio = np.sqrt(252) * (df['Strategy_Return'].mean() / df['Strategy_Return'].std())
        max_drawdown = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1).min()
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'MSE': mean_squared_error(actual_returns, predictions),
            'MAE': mean_absolute_error(actual_returns, predictions)
        }
        
        return metrics, df
