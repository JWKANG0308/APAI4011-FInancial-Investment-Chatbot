import numpy as np
import pandas as pd
import torch
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SectorRecommendationModel:
    """Simplified sector recommendation - first 10 headlines per stock"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

        self.sectors = {
            "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
            "Telecommunications": ["T", "VZ", "TMUS", "CMCSA", "CHTR"],
            "Healthcare": ["UNH", "JNJ", "LLY", "ABBV", "MRK"],
            "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "SPG"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
            "Consumer Staples": ["WMT", "PG", "KO", "PEP", "COST"],
            "Industrials": ["UPS", "BA", "HON", "UNP", "CAT"],
            "Basic Materials": ["LIN", "APD", "SHW", "ECL", "NEM"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
        }

    def get_finbert_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()

        return float(probs[0] - probs[1])

    def get_first_10_headlines(self, ticker_symbol):
        try:
            stock = yf.Ticker(ticker_symbol)
            news = stock.get_news()

            headlines = []
            for item in news[:10]:
                try:
                    headlines.append(item["content"]["title"])
                except Exception:
                    continue
            return headlines
        except Exception:
            return []

    def analyze_stock_sentiment(self, ticker):
        headlines = self.get_first_10_headlines(ticker)

        if not headlines:
            return {"ticker": ticker, "sentiment_score": 0.0, "num_headlines": 0}

        scores = [self.get_finbert_sentiment(h) for h in headlines]

        return {
            "ticker": ticker,
            "sentiment_score": np.mean(scores),
            "num_headlines": len(scores),
        }

    def analyze_sector(self, sector_name, tickers):
        stock_results = []
        all_scores = []

        for ticker in tickers:
            result = self.analyze_stock_sentiment(ticker)
            stock_results.append(result)

            if result["num_headlines"] > 0:
                all_scores.append(result["sentiment_score"])

        sector_avg = np.mean(all_scores) if all_scores else 0.0

        return {
            "sector": sector_name,
            "sentiment_score": sector_avg,
            "total_headlines": sum([r["num_headlines"] for r in stock_results]),
        }

    def get_top_3_recommendations(self):
        results = []
        for sector, tickers in self.sectors.items():
            result = self.analyze_sector(sector, tickers)
            results.append(result)

        df = pd.DataFrame(
            [
                {
                    "Rank": 0,
                    "Sector": r["sector"],
                    "Sentiment Score": r["sentiment_score"],
                    "Total Headlines": r["total_headlines"],
                }
                for r in results
            ]
        )

        df = df.sort_values("Sentiment Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        df = df[["Rank", "Sector", "Sentiment Score", "Total Headlines"]]

        output = "🏆 Top 3 Recommended Sectors:\n\n"
        for _, row in df.head(3).iterrows():
            output += f"{int(row['Rank'])}. {row['Sector']} (Sentiment: {row['Sentiment Score']:+.4f})\n"

        return output
