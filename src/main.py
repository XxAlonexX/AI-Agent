import os
from pathlib import Path
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
    
    processor = DataProcessor()
    try:
        X, y = processor.prepare_data(data, sequence_length=60)
        print("Processed data with technical indicators")
    except Exception as e:
        print(f"Error processing data: {e}")
    
    try:
        model = LSTMTrader(
            input_size=X.shape[2],
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        print("Created LSTM model")
    except Exception as e:
        print(f"Error creating LSTM model: {e}")
    
    trainer = TradingModelTrainer(model)
    
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
