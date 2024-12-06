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
        try:
            model_path = self.models_path / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            
            config_path = self.models_path / f"{model_name}_config.txt"
            config = {
                'input_size': model.lstm.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'output_size': model.fc.out_features
            }
            with open(config_path, 'w') as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
            
            return model_path
        except Exception as e:
            print(f"Warning: Could not save model due to: {str(e)}")
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
        data = yf.download(symbol, start=start_date, end=end_date)
        return data

    def analyze_strategy(self, data, strategy_name):
        pass

    def train_model(self, data, model_type):
        pass
