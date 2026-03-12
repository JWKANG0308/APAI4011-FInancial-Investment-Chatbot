import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB


class TaskClassifierTrainer:
    """
    Train Naive Bayes model for 8-category financial task classification

    Categories:
    1. prediction - Tomorrow stock forecasts
    2. sector_recommendation - Sector investment advice
    3. comparison - Compare multiple companies
    4. ticker_price - Current stock prices
    5. ticker_valuation - P/E, margins, ROE metrics
    6. ticker_dividend - Dividend information
    7. ticker_analyst - Analyst ratings & targets
    8. ticker_info - General company information
    """

    def __init__(self, max_features=500, ngram_range=(1, 2), alpha=1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words="english",
        )
        self.model = MultinomialNB(alpha=alpha)
        self.classes = []
        self.training_history = {}

    def load_data_from_csv(self, csv_path="data/raw/task_classification_dataset_8categories.csv"):
        """Load training data from CSV file"""
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)

        if "query" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'query' and 'label' columns")

        print(f"✓ Loaded {len(df)} samples from CSV")
        return df

    def train(self, csv_path="data/raw/task_classification_dataset_8categories.csv", test_size=0.2, random_state=42):
        """Train the model from CSV data"""
        print("=" * 80)
        print("TRAINING REFINED TASK CLASSIFIER - 8 CATEGORIES")
        print("=" * 80)

        df = self.load_data_from_csv(csv_path)
        X = df["query"]
        y = df["label"]

        print(f"\nTotal samples: {len(df)}")
        print("\nClass distribution:")
        for label, count in y.value_counts().sort_index().items():
            print(f"  {label:25s}: {count}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        print("\nExtracting TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"Feature matrix shape: {X_train_tfidf.shape}")

        print("\nTraining Naive Bayes classifier...")
        self.model.fit(X_train_tfidf, y_train)
        self.classes = self.model.classes_.tolist()

        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_tfidf, y_train, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=self.classes, columns=self.classes)
        print(cm_df)

        self.training_history = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": float(accuracy),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "classes": self.classes,
            "class_distribution": y.value_counts().to_dict(),
        }

        return {
            "accuracy": accuracy,
            "cv_scores": cv_scores,
            "classes": self.classes,
        }

    def predict(self, text):
        """Predict task category for a single query"""
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)

        all_probs = {cls: float(prob) for cls, prob in zip(self.classes, probabilities)}

        return {
            "text": text,
            "predicted_label": prediction,
            "confidence": float(confidence),
            "all_probabilities": all_probs,
        }

    def predict_batch(self, texts):
        """Predict task categories for multiple queries"""
        return [self.predict(text) for text in texts]

    def save_model(
        self,
        filepath="models/task_classifier.pkl",
        history_filepath="models/task_classifier_history.json",
    ):
        """Save trained model and training history"""
        model_data = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "classes": self.classes,
            "training_history": self.training_history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to: {filepath}")

        with open(history_filepath, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2)
        print(f"✓ Training history saved to: {history_filepath}")

    def load_model(self, filepath="models/task_classifier.pkl"):
        """Load trained model"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data["vectorizer"]
        self.model = model_data["model"]
        self.classes = model_data["classes"]
        self.training_history = model_data.get("training_history", {})

        print(f"✓ Model loaded from: {filepath}")
        print(f"  Classes: {', '.join(self.classes)}")

    def get_feature_importance(self, top_n=10):
        """Get top features for each class"""
        feature_names = self.vectorizer.get_feature_names_out()

        print("\n" + "=" * 80)
        print("TOP FEATURES PER CLASS")
        print("=" * 80)

        for idx, class_name in enumerate(self.classes):
            log_probs = self.model.feature_log_prob_[idx]
            top_indices = np.argsort(log_probs)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]

            print(f"\n{class_name}:")
            print(f"  {', '.join(top_features)}")
