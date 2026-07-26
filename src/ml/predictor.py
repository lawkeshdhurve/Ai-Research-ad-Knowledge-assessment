import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pickle
import numpy as np
from typing import Dict, Any, List
import tensorflow as tf
from src.ml.dataset_prep import CATEGORIES
from src.ml.train_classifier import build_and_train_classifier
from config.settings import settings

class DomainClassifierPredictor:
    """Predictor wrapper that loads saved TensorFlow model and classifies documents."""

    def __init__(self):
        self.model_path = str(settings.TF_MODEL_PATH)
        self.tokenizer_path = str(settings.TOKENIZER_PATH)
        self.model = None
        self.categories = CATEGORIES
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Loads model or auto-trains if not present."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.tokenizer_path):
            print("TensorFlow model file not found. Auto-training classifier...")
            self.model, _ = build_and_train_classifier(self.model_path, self.tokenizer_path)
        else:
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                with open(self.tokenizer_path, "rb") as f:
                    meta = pickle.load(f)
                    self.categories = meta.get("categories", CATEGORIES)
            except Exception as e:
                print(f"Error loading TF model: {e}. Re-training...")
                self.model, _ = build_and_train_classifier(self.model_path, self.tokenizer_path)

    def predict_category(self, text: str) -> Dict[str, Any]:
        """
        Predicts domain category for a text snippet.
        Returns predicted category name, confidence score, and category probabilities.
        """
        if not text or not text.strip():
            return {
                "category": "General Tech",
                "confidence": 1.0,
                "probabilities": {cat: 0.2 for cat in self.categories}
            }

        self._ensure_model_loaded()

        try:
            # Format text input for model inference
            input_tensor = tf.constant([text], dtype=tf.string)
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            confidence = float(predictions[top_idx])
            predicted_category = self.categories[top_idx]

            probabilities = {
                cat: round(float(prob), 4)
                for cat, prob in zip(self.categories, predictions)
            }

            return {
                "category": predicted_category,
                "confidence": round(confidence, 4),
                "probabilities": probabilities
            }
        except Exception as e:
            print(f"Inference error in TF classifier: {e}")
            return {
                "category": "General Tech",
                "confidence": 0.5,
                "probabilities": {cat: 0.2 for cat in self.categories}
            }
