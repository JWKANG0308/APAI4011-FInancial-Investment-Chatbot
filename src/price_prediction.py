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
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def get_sentiment_score(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        neu, pos, neg = probs[0].cpu().numpy()
        sentiment_score = pos - neg

    return float(sentiment_score)


def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    data = data.reset_index()
    data = data[["Date", "Close", "Volume"]]
    data.columns = ["date", "close", "volume"]
    return data


def aggregate_sentiment_by_date(df):
    avg_score = df.groupby("date")[["sentiment"]].mean().reset_index()
    return avg_score


def merge_stock_sentiment(df_stock, df_sentiment):
    df_stock = (
        df_stock.reset_index()
        if df_stock.index.name == "date" or (df_stock.index.name is None and not isinstance(df_stock.index, pd.RangeIndex))
        else df_stock.copy()
    )
    df_sentiment = (
        df_sentiment.reset_index()
        if df_sentiment.index.name == "date" or (df_sentiment.index.name is None and not isinstance(df_sentiment.index, pd.RangeIndex))
        else df_sentiment.copy()
    )

    df_stock["date"] = pd.to_datetime(df_stock["date"])
    df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])

    df_stock = df_stock.sort_values("date").reset_index(drop=True)
    df_sentiment = df_sentiment.sort_values("date").reset_index(drop=True)

    sentiment_dict = dict(zip(df_sentiment["date"], df_sentiment["sentiment"]))

    result = df_stock.copy()
    result["sentiment"] = 0.0

    stock_dates_set = set(df_stock["date"])
    accumulated_sentiments = []

    for sentiment_date in df_sentiment["date"]:
        sentiment_value = sentiment_dict[sentiment_date]

        if sentiment_date in stock_dates_set:
            accumulated_sentiments.append(sentiment_value)
            avg_sentiment = np.mean(accumulated_sentiments)
            result.loc[result["date"] == sentiment_date, "sentiment"] = avg_sentiment
            accumulated_sentiments = []
        else:
            accumulated_sentiments.append(sentiment_value)

    return result


def normalize_data(df):
    df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[["close", "volume"]] = scaler.fit_transform(df[["close", "volume"]])
    return df, scaler


def create_sequences(df, lookback=8):
    features = df[["close", "volume", "sentiment"]].values

    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(features[i:i + lookback])
        y.append(df.iloc[i + lookback]["close"])

    return np.array(X), np.array(y)


def inverse_close_from_scaler(normalized_close_value, scaler):
    dummy_array = np.array([[normalized_close_value, 0.0]])
    unnormalized = scaler.inverse_transform(dummy_array)
    return float(unnormalized[0, 0])


def predict_stock_price(ticker, model_path="models/lstm_stock_prediction_model.pth", lookback_days=8):
    try:
        tokenizer, sentiment_model, device = setup_sentiment_model()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 20)

        stock = yf.Ticker(ticker)
        df_stock = stock.history(start=start_date, end=end_date).reset_index()
        if df_stock.empty:
            raise ValueError(f"No stock data found for ticker: {ticker}")

        df_stock = df_stock[["Date", "Close", "Volume"]]
        df_stock.columns = ["date", "close", "volume"]
        df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date

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

        if df_news.empty:
            df_sentiment = pd.DataFrame(columns=["date", "sentiment"])
        else:
            df_news["sentiment"] = df_news["headline"].apply(
                lambda x: get_sentiment_score(x, tokenizer, sentiment_model, device)
            )
            df_sentiment = aggregate_sentiment_by_date(df_news)

        df_merged = merge_stock_sentiment(df_stock, df_sentiment)
        original_close = df_merged["close"].values.copy()

        df_normalized, scaler = normalize_data(df_merged)

        if len(df_normalized) < lookback_days:
            raise ValueError(f"Not enough data. Need {lookback_days} days, got {len(df_normalized)}")

        features = df_normalized[["close", "volume", "sentiment"]].values[-lookback_days:]
        X_input = features.reshape(1, lookback_days, 3)

        model = LSTMModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        X_tensor = torch.FloatTensor(X_input).to(device)

        with torch.no_grad():
            prediction = model(X_tensor).cpu().numpy()

        pred_value = float(prediction) if np.ndim(prediction) == 0 else float(prediction[0])
        predicted_actual_price = inverse_close_from_scaler(pred_value, scaler)

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
