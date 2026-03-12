import pandas as pd
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline


class DetailClassifier:
    """Classify user requests for specific ticker information details"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        self.model = MultinomialNB(alpha=1.0)
        self.classes = []

    def create_training_data(self):
        data = []

        price_queries = [
            "show me price", "price information", "tell me the price", "current price",
            "stock price", "what's the price", "price details", "show price info",
            "give me price", "price data", "see price", "view price",
            "display price", "I want price", "show prices", "pricing information",
            "market price", "trading price", "share price", "stock value",
            "how much is it", "what does it cost", "price now", "current value",
            "today's price", "latest price", "price today", "current stock price",
        ]
        for q in price_queries:
            data.append({"query": q, "label": "price_information"})

        valuation_queries = [
            "valuation", "valuation metrics", "show valuation", "PE ratio",
            "price to earnings", "valuation info", "metrics", "ratios",
            "financial ratios", "valuation data", "P/E", "price to book",
            "show me valuation", "valuation details", "company valuation",
            "stock valuation", "value metrics", "give me valuation",
            "display valuation", "I want valuation", "see valuation",
            "profit margin", "profitability", "margins", "ROE",
            "return on equity", "valuation ratios", "multiples",
        ]
        for q in valuation_queries:
            data.append({"query": q, "label": "valuation_information"})

        dividend_queries = [
            "dividend", "dividend information", "show dividend", "dividends",
            "dividend yield", "payout", "dividend data", "div yield",
            "dividend rate", "show me dividend", "dividend details",
            "give me dividend", "I want dividend", "see dividend",
            "display dividend", "dividend info", "does it pay dividend",
            "dividend payment", "dividend history", "div info",
            "payout ratio", "dividend policy", "yield information",
        ]
        for q in dividend_queries:
            data.append({"query": q, "label": "dividend_information"})

        analyst_queries = [
            "analyst", "analyst information", "recommendations", "target price",
            "analyst rating", "what do analysts say", "analyst opinion",
            "show analyst", "analyst data", "ratings", "price target",
            "analyst recommendation", "analyst details", "analyst info",
            "give me analyst", "I want analyst", "see analyst",
            "display analyst", "analyst views", "analyst forecast",
            "analyst estimates", "consensus", "analyst consensus",
            "buy or sell", "recommendation rating",
        ]
        for q in analyst_queries:
            data.append({"query": q, "label": "analyst_information"})

        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def train(self):
        df = self.create_training_data()

        X = df["query"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        self.classes = self.model.classes_.tolist()

        X_test_tfidf = self.vectorizer.transform(X_test)
        accuracy = self.model.score(X_test_tfidf, y_test)

        return accuracy

    def predict(self, text):
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)

        return {
            "detail_type": prediction,
            "confidence": float(confidence),
        }


class TickerInformationSystem:
    """Complete ticker information system"""

    def __init__(self):
        self.summarizer = None
        self.detail_classifier = DetailClassifier()
        self.detail_classifier.train()

    def initialize_summarizer(self):
        if self.summarizer is None:
            self.summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    def get_summary(self, text):
        self.initialize_summarizer()
        try:
            summary = self.summarizer(text, max_new_tokens=100, min_length=30, do_sample=False)
            return summary[0]["summary_text"]
        except Exception:
            return text[:200] + "..."

    def get_basic_info(self, ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            output = f"\nℹ️ Company Info for {ticker_symbol}:\n"
            output += f"Company: {info.get('longName', 'N/A')}\n"
            output += f"Sector: {info.get('sector', 'N/A')}\n"
            output += f"Industry: {info.get('industry', 'N/A')}\n"
            output += f"Country: {info.get('country', 'N/A')}\n"
            output += f"Website: {info.get('website', 'N/A')}\n"

            business_summary = info.get("longBusinessSummary", "No summary available")
            if business_summary != "No summary available":
                summary = self.get_summary(business_summary)
                output += f"\nSummary: {summary}\n"

            return output, info
        except Exception as e:
            return f"Error: {e}", None

    def get_price_info(self, ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            output = f"\n💰 Price Info for {ticker_symbol}:\n"
            output += f"Current Price: ${info.get('currentPrice', 'N/A')}\n"
            output += f"Previous Close: ${info.get('previousClose', 'N/A')}\n"
            output += f"Day High: ${info.get('dayHigh', 'N/A')}\n"
            output += f"Day Low: ${info.get('dayLow', 'N/A')}\n"
            output += f"52W High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            output += f"52W Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"

            market_cap = info.get("marketCap", "N/A")
            if market_cap != "N/A":
                output += f"Market Cap: ${market_cap:,.0f}\n"

            volume = info.get("volume", "N/A")
            if volume != "N/A":
                output += f"Volume: {volume:,}\n"

            return output
        except Exception:
            return f"Error fetching price for {ticker_symbol}"

    def get_valuation_info(self, ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            output = f"\n📈 Valuation Metrics for {ticker_symbol}:\n"
            output += f"P/E Ratio (Trailing): {info.get('trailingPE', 'N/A')}\n"
            output += f"P/E Ratio (Forward): {info.get('forwardPE', 'N/A')}\n"
            output += f"Price to Book: {info.get('priceToBook', 'N/A')}\n"
            output += f"Price to Sales: {info.get('priceToSalesTrailing12Months', 'N/A')}\n"

            profit_margin = info.get("profitMargins", "N/A")
            if profit_margin != "N/A":
                output += f"Profit Margin: {profit_margin * 100:.2f}%\n"

            roe = info.get("returnOnEquity", "N/A")
            if roe != "N/A":
                output += f"Return on Equity: {roe * 100:.2f}%\n"

            return output
        except Exception:
            return f"Error fetching valuation for {ticker_symbol}"

    def get_dividend_info(self, ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            output = f"\n💵 Dividend Info for {ticker_symbol}:\n"
            output += f"Dividend Rate: {info.get('dividendRate', 'N/A')}\n"

            div_yield = info.get("dividendYield", "N/A")
            if div_yield != "N/A":
                output += f"Dividend Yield: {div_yield * 100:.2f}%\n"
            else:
                output += "Dividend Yield: N/A\n"

            payout_ratio = info.get("payoutRatio", "N/A")
            if payout_ratio != "N/A":
                output += f"Payout Ratio: {payout_ratio * 100:.2f}%\n"
            else:
                output += "Payout Ratio: N/A\n"

            return output
        except Exception:
            return f"Error fetching dividend info for {ticker_symbol}"

    def get_analyst_info(self, ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            output = f"\n🎯 Analyst Info for {ticker_symbol}:\n"
            output += f"Target Mean Price: ${info.get('targetMeanPrice', 'N/A')}\n"
            output += f"Target High Price: ${info.get('targetHighPrice', 'N/A')}\n"
            output += f"Target Low Price: ${info.get('targetLowPrice', 'N/A')}\n"

            rec = info.get("recommendationKey", "N/A")
            output += f"Recommendation: {rec.upper() if rec != 'N/A' else 'N/A'}\n"
            output += f"Number of Analysts: {info.get('numberOfAnalystOpinions', 'N/A')}\n"

            return output
        except Exception:
            return f"Error fetching analyst info for {ticker_symbol}"
