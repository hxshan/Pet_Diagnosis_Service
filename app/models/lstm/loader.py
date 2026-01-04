import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

# In-memory registry
_LSTM_MODELS: dict[str, tf.keras.Model] = {}

def load_lstm_model(version: str) -> tf.keras.Model:
    """
    Load and cache LSTM model by version (Keras 3 compatible).
    """
    # Return cached model if already loaded
    if version in _LSTM_MODELS:
        return _LSTM_MODELS[version]

    # Construct model path
    model_path = Path("artifacts") / f"{version}.keras"
    print(f"Loading LSTM model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found: {model_path}")

    model = load_model(model_path)

    # Cache it for future use
    _LSTM_MODELS[version] = model

    return model
