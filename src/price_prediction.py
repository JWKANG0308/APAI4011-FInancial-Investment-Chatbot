from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.squeeze()


def setup_sentiment_model():
    """Load FinBERT model for sentiment analysis"""
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def get_sentiment_score(text, tokenizer, model, device):
    """Calculate sentiment score for a given text (pos - neg)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        neu, pos, neg = probs[0].cpu().numpy()
        sentiment_score = pos - neg

    return sentiment_score


def get_stock_data_with_news(ticker, lookback_days=8):
    """Fetch stock data and news headlines from yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 10)

    stock = yf.Ticker(ticker)
    df_stock = stock.history(start=start_date, end=end_date)

    if df_stock.empty:
        raise ValueError(f"No stock data found for ticker: {ticker}")

    df_stock = df_stock.reset_index()
    df_stock = df_stock[["Date", "Close", "Volume"]]
    df_stock.columns = ["date", "close", "volume"]
    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date
    df_stock = df_stock.tail(lookback_days).reset_index(drop=True)

    news = stock.get_news(count=250, tab="news")

    news_data = []
    for item in news:
        try:
            title = item["content"]["title"]
            pub_date = item["content"]["pubDate"]
            pub_datetime = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            news_data.append({"headline": title, "date": pub_datetime.date()})
        except Exception:
            continue

    df_news = pd.DataFrame(news_data)
    if not df_news.empty:
        df_news = df_news.sort_values("date", ascending=False).reset_index(drop=True)

    return df_stock, df_news


def process_sentiment(df_news, tokenizer, sentiment_model, device):
    """Process sentiment for news headlines"""
    if df_news.empty:
        return pd.DataFrame(columns=["date", "sentiment"])

    df_news = df_news.copy()
    df_news["sentiment"] = df_news["headline"].apply(
        lambda x: get_sentiment_score(x, tokenizer, sentiment_model, device)
    )

    df_sentiment = df_news.groupby("date")[["sentiment"]].mean().reset_index()
    return df_sentiment


def merge_stock_sentiment(df_stock, df_sentiment):
    """Merge stock data with sentiment scores"""
    df_stock = df_stock.copy()
    df_sentiment = df_sentiment.copy()

    if df_sentiment.empty:
        df_stock["sentiment"] = 0.0
        return df_stock

    df_stock["date"] = pd.to_datetime(df_stock["date"])
    df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])

    df_stock = df_stock.sort_values("date").reset_index(drop=True)
    df_sentiment = df_sentiment.sort_values("date").reset_index(drop=True)

    sentiment_dict = dict(zip(df_sentiment["date"], df_sentiment["sentiment"]))
    stock_dates_set = set(df_stock["date"])

    df_stock["sentiment"] = 0.0
    accumulated_sentiments = []

    for sentiment_date in df_sentiment["date"]:
        sentiment_value = sentiment_dict[sentiment_date]

        if sentiment_date in stock_dates_set:
            accumulated_sentiments.append(sentiment_value)
            avg_sentiment = np.mean(accumulated_sentiments)
            df_stock.loc[df_stock["date"] == sentiment_date, "sentiment"] = avg_sentiment
            accumulated_sentiments = []
        else:
            accumulated_sentiments.append(sentiment_value)

    return df_stock


def normalize_data(df):
    """Normalize close and volume using MinMaxScaler"""
    df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[["close", "volume"]] = scaler.fit_transform(df[["close", "volume"]])
    return df, scaler


def predict_stock_price(ticker, model_path="models/lstm_stock_prediction_model.pth", lookback_days=8):
    """Main inference function - predicts next day close price"""
    try:
        tokenizer, sentiment_model, device = setup_sentiment_model()
        df_stock, df_news = get_stock_data_with_news(ticker, lookback_days)
        df_sentiment = process_sentiment(df_news, tokenizer, sentiment_model, device)
        df_merged = merge_stock_sentiment(df_stock, df_sentiment)

        original_close = df_merged["close"].values.copy()
        df_normalized, scaler = normalize_data(df_merged)

        if len(df_normalized) < lookback_days:
            raise ValueError(f"Not enough data. Need {lookback_days} days, got {len(df_normalized)}")

        features = df_normalized[["close", "volume", "sentiment"]].values[-lookback_days:]
        X_input = features.reshape(1, lookback_days, 3)

        lstm_model = LSTMModel(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2,
        ).to(device)

        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        lstm_model.eval()

        X_tensor = torch.FloatTensor(X_input).to(device)

        with torch.no_grad():
            prediction = lstm_model(X_tensor).cpu().numpy()

        pred_value = float(prediction) if np.ndim(prediction) == 0 else prediction[0]

        dummy_array = np.array([[pred_value, 0.0]])
        unnormalized = scaler.inverse_transform(dummy_array)
        predicted_actual_price = unnormalized[0, 0]

        last_price = original_close[-1]
        price_change = predicted_actual_price - last_price
        pct_change = ((predicted_actual_price / last_price) - 1) * 100

        result = f"📊 Price Prediction for {ticker}\n"
        result += f"Current Price: ${last_price:.2f}\n"
        result += f"Predicted Price (Next Day): ${predicted_actual_price:.2f}\n"
        result += f"Expected Change: ${price_change:+.2f} ({pct_change:+.2f}%)\n"

        return result

    except Exception as e:
        return f"Error predicting price for {ticker}: {str(e)}"
