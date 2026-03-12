from src.price_prediction_trainer import PricePredictionTrainer


def main():
    trainer = PricePredictionTrainer(
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
    )

    ticker_configs = [
        ("KO", "2024-11-07", "2025-11-06"),
        ("AAPL", "2024-11-06", "2025-11-05"),
        ("JNJ", "2024-11-07", "2025-11-06"),
        ("UNH", "2024-11-06", "2025-11-05"),
        ("WMT", "2024-11-06", "2025-11-05"),
    ]

    trainer.run(ticker_configs)


if __name__ == "__main__":
    main()
