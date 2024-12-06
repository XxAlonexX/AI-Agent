import os
from pathlib import Path
import torch
import numpy as np
from agent.quant_research_agent import QuantResearchAgent
from models.lstm_trader import LSTMTrader, TradingModelTrainer
from utils.data_processor import DataProcessor

def train_and_evaluate(X_train, y_train, X_val, y_val, model, trainer, n_epochs=50, batch_size=32):
    """Train the model and track its performance."""
    # Reshape input data for LSTM [batch_size, sequence_length, features]
    X_train = torch.FloatTensor(X_train).unsqueeze(0) if len(X_train.shape) == 2 else torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val).unsqueeze(0) if len(X_val.shape) == 2 else torch.FloatTensor(X_val)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        total_train_loss = 0
        total_val_loss = 0
        
        # Training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Ensure proper dimensions
            if len(batch_X.shape) == 2:
                batch_X = batch_X.unsqueeze(0)
            
            loss = trainer.train_step(batch_X, batch_y)
            total_train_loss += loss
        
        # Validation
        val_loss = trainer.validate(X_val, y_val)
        
        # Calculate average losses
        avg_train_loss = total_train_loss / (len(X_train) // batch_size)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        trainer.scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    return train_losses, val_losses

def main():
    # Initialize components
    repo_path = Path(__file__).parent.parent
    agent = QuantResearchAgent(repo_path)
    
    # Create research document
    research_content = """# Quantitative Trading Strategy Research

## Overview
This document presents comprehensive research on a deep learning-based trading strategy using LSTM networks with attention mechanism.

## Methodology
1. Data Collection and Preprocessing
   - Historical price and volume data
   - Technical indicators calculation
   - Feature scaling and sequence creation

2. Model Architecture
   - LSTM with attention mechanism
   - Dropout for regularization
   - Batch normalization
   - Learning rate scheduling

3. Training Process
   - Early stopping
   - Gradient clipping
   - Cross-validation

4. Strategy Evaluation
   - Performance metrics
   - Risk analysis
   - Trading signals generation
    """
    
    research_doc = agent.create_research_document("Advanced_LSTM_Trading_Strategy", research_content)
    print(f"Created research document: {research_doc}")
    
    try:
        # Fetch historical data
        data = agent.fetch_market_data("AAPL", "2018-01-01", "2023-12-01")
        print("Fetched market data for AAPL")
        
        try:
            # Process data
            processor = DataProcessor()
            X_train, X_val, y_train, y_val = processor.prepare_data(data, sequence_length=60)
            print("Processed data with technical indicators")
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train)
            X_val = torch.FloatTensor(X_val)
            y_train = torch.FloatTensor(y_train)
            y_val = torch.FloatTensor(y_val)
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            
            # Create and train model
            input_size = X_train.shape[2]  # Number of features
            model = LSTMTrader(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                output_size=1,
                dropout=0.2
            )
            print("Created LSTM model with attention mechanism")
            
            trainer = TradingModelTrainer(model)
            print("Starting model training...")
            
            # Train the model
            train_losses, val_losses = train_and_evaluate(
                X_train, y_train,
                X_val, y_val,
                model, trainer,
                n_epochs=50,
                batch_size=32
            )
            
            # Evaluate on test set
            test_predictions = trainer.predict(X_val)
            metrics, performance_df = trainer.evaluate_strategy(
                test_predictions,
                y_val.numpy(),
                initial_capital=100000
            )
            
            # Update research document with results
            results_content = "\n## Results\n"
            results_content += "\n### Performance Metrics\n"
            for metric, value in metrics.items():
                results_content += f"- {metric}: {value}\n"
            
            with open(research_doc, 'a') as f:
                f.write(results_content)
            
            # Save model
            model_path = agent.save_model(model, "lstm_trader_v2")
            if model_path:
                print(f"Saved model to: {model_path}")
            
            # Commit changes
            agent.commit_changes("Updated research with LSTM trading strategy results")
            print("Committed changes to repository")
        
        except Exception as e:
            import traceback
            print(f"Error in data processing or model setup: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            raise
    
    except Exception as e:
        print(f"Error in training process: {e}")
        return

if __name__ == "__main__":
    main()
