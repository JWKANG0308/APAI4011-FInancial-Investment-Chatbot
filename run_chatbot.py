import os
import warnings

from src.price_prediction import predict_stock_price
from src.sector_recommendation import SectorRecommendationModel
from src.task_classifier import TaskClassifierTrainer
from src.ticker_comparison import TickerComparisonSystem
from src.ticker_extractor import TickerExtractor
from src.ticker_information import TickerInformationSystem

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def print_other_tasks():
    """Print recommended tasks"""
    print("\n💡 Other things I can help with:")
    print("  • Stock Price Prediction")
    print("  • Sector Recommendations")
    print("  • Ticker Comparison")
    print("  • Price Information")
    print("  • Valuation Metrics")
    print("  • Dividend Information")
    print("  • Analyst Recommendations")
    print("  • General Company Information\n")


def main():
    """Main chatbot loop"""
    print("=" * 80)
    print("💼 FINANCIAL INVESTMENT CHATBOT")
    print("=" * 80)
    print("Ask me about stock predictions, sector recommendations, company info, and more!")
    print("Type 'quit' or 'exit' to end the session.\n")

    print("🤖 Initializing chatbot...")

    classifier = TaskClassifierTrainer()
    classifier.load_model("models/task_classifier.pkl")

    extractor = TickerExtractor("data/raw/nasdaq_list.csv")
    sector_model = SectorRecommendationModel()
    comp_system = TickerComparisonSystem()
    ticker_info_system = TickerInformationSystem()

    print("✅ Chatbot ready!\n")
    print("-" * 80)

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in ["quit", "exit", "bye"]:
            print("\n👋 Thank you for using the Financial Chatbot. Goodbye!")
            break

        if not query:
            continue

        result = classifier.predict(query)
        task = result["predicted_label"]
        confidence = result["confidence"]

        print(f"task class {task}")

        if confidence < 0.15:
            print("\n🤖 Chatbot:")
            print("I apologize, but I cannot find the information you are asking for. Please rephrase your query.")
            print_other_tasks()
            continue

        print("\n🤖 Chatbot:")

        if task == "sector_recommendation":
            output = sector_model.get_top_3_recommendations()
            print(output)
            print_other_tasks()
            continue

        extracted = extractor.extract(query)
        tickers = [
            item["ticker"]
            for item in extracted
            if str(item["ticker"]) != "nan" and item["ticker"] is not None
        ]

        if not tickers:
            print("I couldn't identify a stock ticker in your query. Please mention a company name or ticker symbol.")
            print_other_tasks()
            continue

        main_ticker = tickers[0]
        print(f"tickers are extracted {main_ticker}")

        try:
            if task == "prediction":
                output = predict_stock_price(main_ticker, model_path="models/lstm_stock_prediction_model.pth")
                print(output)

            elif task == "comparison":
                if len(tickers) < 2:
                    print(f"For comparison, I need at least 2 tickers. I found: {tickers}")
                    print("Please include multiple company names or ticker symbols.")
                else:
                    sentiment_df, financial_df = comp_system.compare_tickers(tickers)
                    comp_system.display_sentiment_comparison(sentiment_df)
                    comp_system.display_financial_comparison(financial_df)
                    comp_system.generate_summary(sentiment_df, financial_df)

            elif task == "ticker_price":
                output = ticker_info_system.get_price_info(main_ticker)
                print(output)

            elif task == "ticker_valuation":
                output = ticker_info_system.get_valuation_info(main_ticker)
                print(output)

            elif task == "ticker_dividend":
                output = ticker_info_system.get_dividend_info(main_ticker)
                print(output)

            elif task == "ticker_analyst":
                output = ticker_info_system.get_analyst_info(main_ticker)
                print(output)

            elif task == "ticker_info":
                output, _ = ticker_info_system.get_basic_info(main_ticker)
                print(output)

            else:
                print(f"Task '{task}' is recognized but not yet implemented.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

        print_other_tasks()
        print("-" * 80)


if __name__ == "__main__":
    main()
