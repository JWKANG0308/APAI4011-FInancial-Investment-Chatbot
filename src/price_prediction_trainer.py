import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.price_prediction import (
    LSTMModel,
    aggregate_sentiment_by_date,
    create_sequences,
    get_sentiment_score,
    get_stock_data,
    merge_stock_sentiment,
    normalize_data,
    setup_sentiment_model,
)


class PricePredictionTrainer:
    def __init__(
        self,
        data_dir="data/raw",
        model_output_path="models/lstm_stock_prediction_model.pth",
        scaler_output_path="models/price_scalers.pkl",
        outputs_dir="outputs",
        lookback=8,
        batch_size=32,
        lr=0.001,
        epochs=100,
        test_size=0.2,
        random_state=42,
    ):
        self.data_dir = data_dir
        self.model_output_path = model_output_path
        self.scaler_output_path = scaler_output_path
        self.outputs_dir = outputs_dir
        self.lookback = lookback
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.test_size = test_size
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.sentiment_model = None
        self.sentiment_device = None

        self.model = None
        self.history = {"train_losses": []}
        self.scalers = {}
        self.per_ticker_data = {}

        os.makedirs("models", exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    def load_sentiment_model(self):
        self.tokenizer, self.sentiment_model, self.sentiment_device = setup_sentiment_model()

    def txt_to_df(self, filepath):
        parsed_data = []

        with open(filepath, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                parts = [p for p in line.split("\t") if p]
                if len(parts) >= 7:
                    headline = parts[3].strip()
                    date_str = parts[6].strip()
                    transformed_date_str = date_str.replace("/", "-")

                    if (transformed_date_str[:2] == "12") or ((transformed_date_str[:2] == "11") and (i > 100)):
                        parsed_data.append((headline, "2024-" + transformed_date_str))
                    else:
                        parsed_data.append((headline, "2025-" + transformed_date_str))

        df = pd.DataFrame(parsed_data, columns=["headline", "date"])
        return df

    def load_news_data(self, ticker):
        filepath = os.path.join(self.data_dir, f"{ticker}.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"News file not found: {filepath}")
        return self.txt_to_df(filepath)

    def add_sentiment_scores(self, df_news):
        df_news = df_news.copy()
        df_news["sentiment"] = df_news["headline"].apply(
            lambda x: get_sentiment_score(x, self.tokenizer, self.sentiment_model, self.sentiment_device)
        )
        return df_news

    def build_dataset_for_ticker(self, ticker, start_date, end_date):
        print(f"\nProcessing {ticker}...")

        df_news = self.load_news_data(ticker)
        df_news = self.add_sentiment_scores(df_news)
        df_daily_sentiment = aggregate_sentiment_by_date(df_news)

        df_stock = get_stock_data(ticker, start_date, end_date)
        df_merged = merge_stock_sentiment(df_stock, df_daily_sentiment)

        df_normalized, scaler = normalize_data(df_merged)
        X, y = create_sequences(df_normalized, lookback=self.lookback)

        self.scalers[ticker] = scaler
        self.per_ticker_data[ticker] = {
            "df_news": df_news,
            "df_daily_sentiment": df_daily_sentiment,
            "df_stock": df_stock,
            "df_merged": df_merged,
            "df_normalized": df_normalized,
            "X": X,
            "y": y,
        }

        print(f"  News rows: {len(df_news)}")
        print(f"  Stock rows: {len(df_stock)}")
        print(f"  Sequence shape: X={X.shape}, y={y.shape}")

        return X, y

    def combine_datasets(self, ticker_configs):
        X_list = []
        y_list = []

        for ticker, start_date, end_date in ticker_configs:
            X, y = self.build_dataset_for_ticker(ticker, start_date, end_date)
            X_list.append(X)
            y_list.append(y)

        X_combined = np.concatenate(X_list, axis=0)
        y_combined = np.concatenate(y_list, axis=0)

        print("\nCombined dataset")
        print(f"  X_combined shape: {X_combined.shape}")
        print(f"  y_combined shape: {y_combined.shape}")

        return X_combined, y_combined

    def prepare_dataloaders(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

    def train_model(self, train_loader):
        self.model = LSTMModel().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.history["train_losses"] = []

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            self.history["train_losses"].append(avg_train_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{self.epochs} | Train Loss: {avg_train_loss:.6f}")

        print("✓ Training completed")

    def evaluate_model(self, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_tensor).cpu().numpy()
            test_pred = self.model(X_test_tensor).cpu().numpy()
            y_train_actual = y_train_tensor.cpu().numpy()
            y_test_actual = y_test_tensor.cpu().numpy()

        train_mse = mean_squared_error(y_train_actual, train_pred)
        train_mae = mean_absolute_error(y_train_actual, train_pred)
        train_r2 = r2_score(y_train_actual, train_pred)

        test_mse = mean_squared_error(y_test_actual, test_pred)
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        test_r2 = r2_score(y_test_actual, test_pred)
        test_mape = np.mean(np.abs((y_test_actual - test_pred) / (y_test_actual + 1e-8))) * 100

        metrics = {
            "train_mse": float(train_mse),
            "train_mae": float(train_mae),
            "train_r2": float(train_r2),
            "test_mse": float(test_mse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
            "test_mape": float(test_mape),
            "test_accuracy_r2_percent": float(test_r2 * 100),
            "test_accuracy_1_minus_mape_percent": float(100 - test_mape),
        }

        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print("\nTRAINING:")
        print(f"  MSE:      {metrics['train_mse']:.6f}")
        print(f"  MAE:      {metrics['train_mae']:.6f}")
        print(f"  R²:       {metrics['train_r2']:.6f}")
        print(f"  Accuracy: {metrics['train_r2'] * 100:.2f}%")

        print("\nTEST:")
        print(f"  MSE:      {metrics['test_mse']:.6f}")
        print(f"  MAE:      {metrics['test_mae']:.6f}")
        print(f"  R²:       {metrics['test_r2']:.6f}")
        print(f"  Accuracy (R²): {metrics['test_accuracy_r2_percent']:.2f}%")

        print("\nAlternative:")
        print(f"  MAPE:     {metrics['test_mape']:.2f}%")
        print(f"  Accuracy (1-MAPE): {metrics['test_accuracy_1_minus_mape_percent']:.2f}%")

        if metrics["test_accuracy_r2_percent"] >= 80:
            print(f"\n✓✓ EXCELLENT - {metrics['test_accuracy_r2_percent']:.1f}% variance explained")
        elif metrics["test_accuracy_r2_percent"] >= 60:
            print(f"\n✓ GOOD - {metrics['test_accuracy_r2_percent']:.1f}% variance explained")
        else:
            print(f"\n~ FAIR - {metrics['test_accuracy_r2_percent']:.1f}% variance explained")

        return metrics, train_pred, test_pred, y_train_actual, y_test_actual

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_output_path)
        print(f"✓ Model saved to: {self.model_output_path}")

        with open(self.scaler_output_path, "wb") as f:
            pickle.dump(self.scalers, f)
        print(f"✓ Scalers saved to: {self.scaler_output_path}")

    def save_metrics(self, metrics):
        metrics_path = os.path.join(self.outputs_dir, "training_metrics.csv")
        df_metrics = pd.DataFrame([metrics])
        df_metrics["timestamp"] = datetime.now().isoformat()
        df_metrics.to_csv(metrics_path, index=False)
        print(f"✓ Metrics saved to: {metrics_path}")

    def plot_training_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_losses"])
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(self.outputs_dir, "training_loss.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved plot: {save_path}")

    def plot_test_results(self, y_test_actual, test_pred):
        plt.figure(figsize=(14, 6))
        plt.plot(y_test_actual, label="Actual (Test Set)", alpha=0.6)
        plt.plot(test_pred, label="Predicted (Test Set)", alpha=0.6)
        plt.title("LSTM Model: Actual vs Predicted (Test Set - Shuffled)")
        plt.xlabel("Sample Index")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(self.outputs_dir, "test_actual_vs_predicted.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved plot: {save_path}")

    def plot_continuous_tracking(self, ticker="WMT"):
        if ticker not in self.per_ticker_data:
            print(f"Skipping continuous plot: {ticker} not found")
            return

        X_ticker = self.per_ticker_data[ticker]["X"]
        y_ticker = self.per_ticker_data[ticker]["y"]

        X_ticker_tensor = torch.FloatTensor(X_ticker).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_ticker = self.model(X_ticker_tensor).cpu().numpy()

        plt.figure(figsize=(14, 6))
        plt.plot(y_ticker, label=f"Actual {ticker} Price (Normalized)")
        plt.plot(pred_ticker, label=f"Predicted {ticker} Price (Normalized)", linestyle="--")
        plt.title(f"Time-Series Tracking: {ticker} - Continuous Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Close Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(self.outputs_dir, f"{ticker.lower()}_tracking.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved plot: {save_path}")

    def run(self, ticker_configs):
        print("=" * 80)
        print("PRICE PREDICTION TRAINER")
        print("=" * 80)
        print(f"Using device: {self.device}")

        self.load_sentiment_model()

        X, y = self.combine_datasets(ticker_configs)
        train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = self.prepare_dataloaders(X, y)

        self.train_model(train_loader)

        metrics, train_pred, test_pred, y_train_actual, y_test_actual = self.evaluate_model(
            X_train_tensor,
            y_train_tensor,
            X_test_tensor,
            y_test_tensor,
        )

        self.save_model()
        self.save_metrics(metrics)
        self.plot_training_loss()
        self.plot_test_results(y_test_actual, test_pred)
        self.plot_continuous_tracking("WMT")

        return metrics
