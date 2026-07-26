import os
import pickle
import numpy as np
from typing import Tuple, Any

# Ensure TensorFlow doesn't output excessive warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
from src.ml.dataset_prep import generate_sample_dataset, CATEGORIES
from config.settings import settings

def build_and_train_classifier(
    model_path: str = None, 
    tokenizer_path: str = None
) -> Tuple[Any, Any]:
    """
    Builds, trains, evaluates, and saves a TensorFlow document domain classifier model.
    Saves .h5 model file and tokenizer/vectorizer pickle artifact.
    """
    if model_path is None:
        model_path = str(settings.TF_MODEL_PATH)
    if tokenizer_path is None:
        tokenizer_path = str(settings.TOKENIZER_PATH)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    texts, labels = generate_sample_dataset()
    labels = np.array(labels)
    num_classes = len(CATEGORIES)

    vocab_size = 5000
    max_len = 150

    text_tensor = tf.constant(texts, dtype=tf.string)

    # 1. Text Vectorization Layer
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len
    )
    vectorize_layer.adapt(text_tensor)

    # 2. Sequential Neural Network Architecture
    model = models.Sequential([
        vectorize_layer,
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    # 3. Compile Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Train Model
    model.fit(
        text_tensor, 
        labels, 
        epochs=15, 
        batch_size=16, 
        verbose=0
    )

    # 5. Persist Model and Artifacts
    model.save(model_path)

    # Store metadata / category mapping artifact
    metadata = {
        "categories": CATEGORIES,
        "max_len": max_len,
        "vocab_size": vocab_size
    }
    with open(tokenizer_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"TensorFlow classification model saved to: {model_path}")
    return model, metadata

if __name__ == "__main__":
    build_and_train_classifier()
