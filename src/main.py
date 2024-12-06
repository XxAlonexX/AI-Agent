import os
from pathlib import Path
import torch
from agent.quant_research_agent import QuantResearchAgent
from models.lstm_trader import LSTMTrader, TradingModelTrainer
from utils.data_processor import DataProcessor

def main():
    repo_path = Path(__file__).parent.parent
    agent = QuantResearchAgent(repo_path)
    
    research_content = """# Trading Strategy Research
    
## Overview
This document presents research on a deep learning-based trading strategy using LSTM networks.

## Methodology
1. Data Collection
2. Feature Engineering
3. Model Architecture
4. Training Process
5. Results Analysis
    """
    
    research_doc = agent.create_research_document("LSTM_Trading_Strategy", research_content)
    print(f"Created research document: {research_doc}")
    
    try:
        data = agent.fetch_market_data("AAPL", "2020-01-01", "2023-01-01")
        print("Fetched market data for AAPL")
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return
    
    processor = DataProcessor()
    try:
        X, y = processor.prepare_data(data, sequence_length=60)
        print("Processed data with technical indicators")
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Split data into train and test
        train_size = int(len(X) * 0.8)
        X_train = X_tensor[:train_size]
        y_train = y_tensor[:train_size]
        X_test = X_tensor[train_size:]
        y_test = y_tensor[train_size:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    try:
        model = LSTMTrader(
            input_size=X.shape[2],
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        print("Created LSTM model")
        
        trainer = TradingModelTrainer(model)
        print("Starting model training...")
        
        # Train the model
        n_epochs = 10
        batch_size = 32
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                loss = trainer.train_step(batch_X, batch_y)
                total_loss += loss
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
            
    except Exception as e:
        print(f"Error in model training: {e}")
        return
    
    try:
        model_path = agent.save_model(model, "lstm_trader_v1")
        if model_path:
            print(f"Saved model to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    try:
        agent.commit_changes("Initial commit: Added LSTM trading model and research")
        print("Committed changes to repository")
    except Exception as e:
        print(f"Error committing changes: {e}")

if __name__ == "__main__":
    main()
