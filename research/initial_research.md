# Quantitative Trading Strategy Using LSTM Networks

## Introduction
This research explores the application of Long Short-Term Memory (LSTM) networks for predicting stock price movements and developing profitable trading strategies.

## Data Collection and Preprocessing
- Historical price data from major stock indices
- Technical indicators:
  - Moving averages (20-day, 50-day)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)

## Model Architecture
The LSTM model consists of:
- Input layer with multiple features
- 2 LSTM layers with 64 hidden units
- Dropout layers for regularization
- Dense output layer for price prediction

## Training Process
- Sequence length: 60 days
- Train/test split: 80/20
- Optimization: Adam optimizer
- Loss function: Mean Squared Error

## Initial Results
(To be updated with actual training results)

## Next Steps
1. Implement backtesting framework
2. Add more technical indicators
3. Experiment with different model architectures
4. Incorporate market sentiment analysis
