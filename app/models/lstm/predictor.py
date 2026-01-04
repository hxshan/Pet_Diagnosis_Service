import numpy as np
from app.models.lstm.loader import load_lstm_model


def run_lstm_inference(
    sequence: list[float],
    metadata: list[float] | None,
    model_version: str
) -> tuple[int, float]:
    """
    Run LSTM inference and return (prediction, confidence).
    """

    model = load_lstm_model(model_version)

    # Convert input to model format
    x = np.array(sequence, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, timesteps)

    # Example: metadata ignored for now
    preds = model.predict(x, verbose=0)[0]

    prediction = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return prediction, confidence
