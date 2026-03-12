import numpy as np
import pandas as pd
import torch
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TickerComparisonSystem:
    """
    Compare multiple tickers with:
    - News sentiment analysis
    - Financial metrics comparison
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None

    def initialize_models(self):
        if self.tokenizer is None:
            print("Loading FinBERT sentiment model...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()
            print("✓ Model loaded")

    def get_finbert_sentiment(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()

        return float(probs[0] - probs[1])

    def get_first_20_headlines(self, ticker_symbol: str) -> list:
        try:
            stock = yf.Ticker(ticker_symbol)
            news = stock.get_news(count=20, tab="news")

            headlines = []
            for item in news[:20]:
                try:
                    if "title" in item:
                        headlines.append(item["title"])
                    elif "content" in item and "title" in item["content"]:
                        headlines.append(item["content"]["title"])
                except Exception:
                    continue
            return headlines
        except Exception:
            return []

    def analyze_stock_sentiment(self, ticker: str) -> dict:
        headlines = self.get_first_20_headlines(ticker)

        if not headlines:
            return {"ticker": ticker, "sentiment_score": 0.0, "num_headlines": 0}

        scores = [self.get_finbert_sentiment(h) for h in headlines]

        return {
            "ticker": ticker,
            "sentiment_score": np.mean(scores),
            "num_headlines": len(scores),
        }

    def get_financial_metrics(self, ticker_symbol: str) -> dict:
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            return {
                "ticker": ticker_symbol,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "current_price": info.get("currentPrice", np.nan),
                "market_cap": info.get("marketCap", np.nan),
                "pe_ratio": info.get("trailingPE", np.nan),
                "forward_pe": info.get("forwardPE", np.nan),
                "price_to_book": info.get("priceToBook", np.nan),
                "profit_margin": info.get("profitMargins", np.nan) * 100 if info.get("profitMargins") else np.nan,
                "revenue_growth": info.get("revenueGrowth", np.nan) * 100 if info.get("revenueGrowth") else np.nan,
                "earnings_growth": info.get("earningsGrowth", np.nan) * 100 if info.get("earningsGrowth") else np.nan,
                "dividend_yield": info.get("dividendYield", np.nan) * 100 if info.get("dividendYield") else np.nan,
                "beta": info.get("beta", np.nan),
                "roe": info.get("returnOnEquity", np.nan) * 100 if info.get("returnOnEquity") else np.nan,
                "debt_to_equity": info.get("debtToEquity", np.nan),
                "target_price": info.get("targetMeanPrice", np.nan),
                "recommendation": info.get("recommendationKey", "N/A"),
            }

        except Exception as e:
            print(f"  Error fetching data for {ticker_symbol}: {e}")
            return {"ticker": ticker_symbol, "company_name": "Error"}

    def compare_tickers(self, ticker_symbols: list) -> tuple:
        print("=" * 80)
        print("TICKER COMPARISON ANALYSIS")
        print("=" * 80)
        print(f"\nComparing: {', '.join(ticker_symbols)}")

        self.initialize_models()

        print("\n" + "=" * 80)
        print("SENTIMENT ANALYSIS (20 Headlines per Ticker)")
        print("=" * 80)

        sentiment_results = []
        for ticker in ticker_symbols:
            print(f"\nAnalyzing {ticker}...")
            result = self.analyze_stock_sentiment(ticker)
            sentiment_results.append(result)
            print(f"  Sentiment Score: {result['sentiment_score']:+.4f}")
            print(f"  Headlines: {result['num_headlines']}")

        print("\n" + "=" * 80)
        print("GATHERING FINANCIAL METRICS")
        print("=" * 80)

        financial_results = []
        for ticker in ticker_symbols:
            print(f"  Fetching data for {ticker}...")
            result = self.get_financial_metrics(ticker)
            financial_results.append(result)

        sentiment_df = pd.DataFrame(sentiment_results)
        financial_df = pd.DataFrame(financial_results)

        return sentiment_df, financial_df

    def display_sentiment_comparison(self, sentiment_df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("SENTIMENT COMPARISON")
        print("=" * 80)

        sentiment_df = sentiment_df.sort_values("sentiment_score", ascending=False)
        print("\n" + sentiment_df.to_string(index=False))

        if len(sentiment_df) > 0:
            best = sentiment_df.iloc[0]
            worst = sentiment_df.iloc[-1]

            print("\n" + "-" * 80)
            print(f"✓ Most Positive: {best['ticker']} (Score: {best['sentiment_score']:+.4f})")
            print(f"✗ Most Negative: {worst['ticker']} (Score: {worst['sentiment_score']:+.4f})")

    def display_financial_comparison(self, financial_df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("FINANCIAL METRICS COMPARISON")
        print("=" * 80)

        print("\n" + "-" * 80)
        print("VALUATION METRICS")
        print("-" * 80)
        valuation_df = financial_df[
            ["ticker", "current_price", "market_cap", "pe_ratio", "forward_pe", "price_to_book"]
        ].copy()
        valuation_df.columns = ["Ticker", "Price ($)", "Market Cap", "P/E", "Fwd P/E", "P/B"]
        print(valuation_df.to_string(index=False))

        print("\n" + "-" * 80)
        print("PROFITABILITY & GROWTH")
        print("-" * 80)
        profitability_df = financial_df[
            ["ticker", "profit_margin", "roe", "revenue_growth", "earnings_growth"]
        ].copy()
        profitability_df.columns = ["Ticker", "Profit Margin %", "ROE %", "Revenue Growth %", "Earnings Growth %"]
        print(profitability_df.to_string(index=False))

        print("\n" + "-" * 80)
        print("RISK & DIVIDEND")
        print("-" * 80)
        risk_df = financial_df[["ticker", "beta", "debt_to_equity", "dividend_yield"]].copy()
        risk_df.columns = ["Ticker", "Beta", "Debt/Equity", "Dividend Yield %"]
        print(risk_df.to_string(index=False))

        print("\n" + "-" * 80)
        print("ANALYST RECOMMENDATIONS")
        print("-" * 80)
        analyst_df = financial_df[["ticker", "current_price", "target_price", "recommendation"]].copy()
        analyst_df["upside"] = (
            (analyst_df["target_price"] - analyst_df["current_price"]) / analyst_df["current_price"] * 100
        )
        analyst_df.columns = ["Ticker", "Current Price", "Target Price", "Recommendation", "Upside %"]
        print(analyst_df.to_string(index=False))

    def generate_summary(self, sentiment_df: pd.DataFrame, financial_df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY & RECOMMENDATIONS")
        print("=" * 80)

        merged = pd.merge(sentiment_df, financial_df, on="ticker")

        best_sentiment = merged.loc[merged["sentiment_score"].idxmax()]
        print(f"\n✓ Best News Sentiment: {best_sentiment['ticker']}")
        print(f"  Score: {best_sentiment['sentiment_score']:+.4f}")

        profitable = merged[merged["pe_ratio"] > 0]
        if len(profitable) > 0:
            best_valuation = profitable.loc[profitable["pe_ratio"].idxmin()]
            print(f"\n✓ Best Valuation (Lowest P/E): {best_valuation['ticker']}")
            print(f"  P/E Ratio: {best_valuation['pe_ratio']:.2f}")

        growth_stocks = merged[merged["earnings_growth"] > 0]
        if len(growth_stocks) > 0:
            best_growth = growth_stocks.loc[growth_stocks["earnings_growth"].idxmax()]
            print(f"\n✓ Highest Growth: {best_growth['ticker']}")
            print(f"  Earnings Growth: {best_growth['earnings_growth']:.2f}%")

        dividend_stocks = merged[merged["dividend_yield"] > 0]
        if len(dividend_stocks) > 0:
            best_dividend = dividend_stocks.loc[dividend_stocks["dividend_yield"].idxmax()]
            print(f"\n✓ Highest Dividend: {best_dividend['ticker']}")
            print(f"  Dividend Yield: {best_dividend['dividend_yield']:.2f}%")

        return merged
