# APAI4011-FInancial-Investment-Chatbot
# Financial Investment Chatbot

A finance-focused investment chatbot built in Python that supports stock-related question answering through task classification, ticker extraction, sentiment-based analysis, stock price forecasting, and financial information retrieval.

This project was originally developed as a semester project in Google Colab and later reorganized into a modular GitHub repository.

## Report
[Final Project Report](report/From Intent to Insight A Multi Task Conversational Agent for Financial Decision Support.pdf)

---

## Features

### 1. Financial Query Classification
The chatbot classifies user queries into finance-related task categories using a TF-IDF + Naive Bayes classifier.

Supported task types:
- `prediction`
- `sector_recommendation`
- `comparison`
- `ticker_price`
- `ticker_valuation`
- `ticker_dividend`
- `ticker_analyst`
- `ticker_info`

### 2. Ticker Extraction
The system identifies company names or ticker symbols from user queries using a custom mapping table.

### 3. Stock Price Prediction
A sentiment-enhanced LSTM model predicts next-day stock price movement using:
- historical close price
- trading volume
- financial news sentiment

### 4. Sector Recommendation
The chatbot analyzes recent financial news sentiment across representative companies in different sectors and returns the top recommended sectors.

### 5. Multi-Ticker Comparison
Users can compare multiple stocks based on:
- recent news sentiment
- valuation metrics
- growth / profitability
- dividend information
- analyst targets

### 6. Company Information Retrieval
The chatbot can retrieve:
- general company information
- current price information
- valuation metrics
- dividend details
- analyst recommendations

### 7. LSTM Price Prediction Training Pipeline
This repository also includes a training pipeline for the stock price prediction model.

Training pipeline includes:
- historical stock data collection from Yahoo Finance
- financial headline sentiment scoring with FinBERT
- sentiment aggregation by date
- stock + sentiment merging
- LSTM sequence generation
- training / evaluation / model saving
- visualization of training and prediction results

---

## Project Structure

```text
finance-investment-chatbot/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_chatbot.py
├── train_price_model.py
│
├── src/
│   ├── __init__.py
│   ├── task_classifier.py
│   ├── ticker_extractor.py
│   ├── price_prediction.py
│   ├── price_prediction_trainer.py
│   ├── sector_recommendation.py
│   ├── ticker_comparison.py
│   └── ticker_information.py
│
├── data/
│   ├── raw/
│   │   ├── nasdaq_list.csv
│   │   ├── task_classification_dataset_8categories.csv
│   │   ├── KO.txt
│   │   ├── AAPL.txt
│   │   ├── JNJ.txt
│   │   ├── UNH.txt
│   │   └── WMT.txt
│
│
├── models/
    ├── task_classifier.pkl
    ├── lstm_stock_prediction_model.pth
