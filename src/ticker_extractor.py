import re

import numpy as np
import pandas as pd


class TickerExtractor:
    """Extract company ticker, name, and sector from user queries"""

    def __init__(self, file_path="data/raw/nasdaq_list.csv"):
        if file_path is None:
            raise ValueError("file_path must be provided")

        if file_path.endswith(".csv"):
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.read_excel(file_path)

        required_cols = ["ticker", "company_name", "sector"]
        if not all(col in self.df.columns for col in required_cols):
            column_mapping = {}
            for col in self.df.columns:
                col_lower = col.lower()
                if "ticker" in col_lower or "symbol" in col_lower:
                    column_mapping[col] = "ticker"
                elif "company" in col_lower or "name" in col_lower or "security" in col_lower:
                    column_mapping[col] = "company_name"
                elif "sector" in col_lower or "industry" in col_lower:
                    column_mapping[col] = "sector"

            if len(column_mapping) >= 3:
                self.df = self.df.rename(columns=column_mapping)

        self.df["ticker"] = self.df["ticker"].astype(str).str.upper().str.strip()
        self.df["company_name"] = self.df["company_name"].astype(str).str.strip()
        self.df["sector"] = self.df["sector"].fillna("Unknown").astype(str).str.strip()
        self.df = self.df.drop_duplicates(subset=["ticker"])

        self.ticker_to_info = {}
        self.company_to_ticker = {}

        for _, row in self.df.iterrows():
            ticker = row["ticker"]
            company = row["company_name"]
            sector = row["sector"]

            self.ticker_to_info[ticker] = {
                "ticker": ticker,
                "company_name": company,
                "sector": sector,
            }

            company_lower = company.lower()
            self.company_to_ticker[company_lower] = ticker

            base_name = " ".join(company_lower.split()).rstrip(".,;:")

            if base_name and len(base_name) > 2 and base_name != company_lower:
                self.company_to_ticker[base_name] = ticker

            first_word = base_name.split()[0] if base_name.split() else ""
            if len(first_word) > 3:
                self.company_to_ticker[first_word] = ticker

        print(f"✓ Loaded {len(self.df)} companies")
        print(f"✓ Created {len(self.company_to_ticker)} name variations")

    def extract_ticker_pattern(self, text):
        ticker_pattern = r"\b[A-Z]{1,5}\b"
        potential_tickers = re.findall(ticker_pattern, text)
        valid_tickers = [t for t in potential_tickers if t in self.ticker_to_info]
        return list(set(valid_tickers))

    def extract_company_name(self, text):
        text_lower = text.lower()
        found_tickers = set()

        sorted_companies = sorted(
            self.company_to_ticker.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for company, ticker in sorted_companies:
            if company in text_lower:
                found_tickers.add(ticker)
                text_lower = text_lower.replace(company, " " * len(company))

        return list(found_tickers)

    def extract(self, query):
        tickers = set()
        tickers.update(self.extract_ticker_pattern(query))
        tickers.update(self.extract_company_name(query))

        results = []
        for ticker in tickers:
            if ticker in self.ticker_to_info:
                results.append(self.ticker_to_info[ticker])

        if not results:
            return [
                {
                    "ticker": np.nan,
                    "company_name": np.nan,
                    "sector": np.nan,
                }
            ]

        return results
