import os
import git
import markdown
from datetime import datetime
import torch
import yfinance as yf
import pandas as pd
from pathlib import Path

class QuantResearchAgent:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.research_path = self.repo_path / "research"
        self.models_path = self.repo_path / "models"
        self.data_path = self.repo_path / "data"
        
        for path in [self.research_path, self.models_path, self.data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.repo = git.Repo.init(repo_path)

    def create_research_document(self, title, content):
        filename = f"{datetime.now().strftime('%Y%m%d')}_{title.lower().replace(' ', '_')}.md"
        filepath = self.research_path / filename
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath

    def save_model(self, model, model_name):
        """Save the trained model."""
        try:
            # Create models directory if it doesn't exist
            models_dir = self.repo_path / "models"
            if not models_dir.exists():
                models_dir.mkdir(parents=True)
            
            # Save just the model state dict
            model_path = models_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved successfully to {model_path}")
            return str(model_path)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return None

    def commit_changes(self, message):
        try:
            self.repo.index.add('*')
            self.repo.index.commit(message)
            
            origin = self.repo.remote(name='origin')
            current = self.repo.active_branch
            origin.push(current.name)
            print(f"Changes pushed to remote repository: {origin.url}")
        except Exception as e:
            print(f"Warning: Error during git operations: {str(e)}")

    def fetch_market_data(self, symbol, start_date, end_date):
        """Fetch market data for a given symbol."""
        import yfinance as yf
        import time
        from datetime import datetime, timedelta
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Fetch data with longer timeout
                data = yf.download(symbol, 
                                 start=start_date, 
                                 end=end_date,
                                 progress=False,
                                 timeout=20)
                
                if data.empty:
                    print(f"No data received for {symbol}. Trying with extended date range...")
                    # Try with a longer date range
                    extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
                    data = yf.download(symbol, 
                                     start=extended_start, 
                                     end=end_date,
                                     progress=False,
                                     timeout=20)
                
                # Clean the data
                data = data.dropna()
                
                # Handle multi-index columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                data = data.astype(float)
                
                if len(data) == 0:
                    raise ValueError(f"No valid data found for {symbol}")
                    
                print(f"Successfully fetched {len(data)} rows of market data")
                print("Columns:", data.columns.tolist())
                print("Sample data:\n", data.head())
                
                return data
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")

    def analyze_strategy(self, data, strategy_name):
        pass

    def train_model(self, data, model_type):
        pass
