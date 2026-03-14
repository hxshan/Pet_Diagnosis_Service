from fastapi import APIRouter, UploadFile, File
from typing import List
import os
import traceback

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

router = APIRouter()

# Path to your CNN model in artifacts. Change if your file has a different name.
# MODEL_PATH = "artifacts/dog_skin_cnn_mobilenet2.keras"
MODEL_PATH = "artifacts/dog_skin_cnn_mobilenet3.keras"

# Example class names used in your Colab snippet. Update to match your model's classes if needed.
CLASS_NAMES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "Invalid",
    "demodicosis",
    "ringworm"
]


# CLASS_NAMES = [
#     "Dermatitis",
#     "Fungal_infections",
#     "Healthy",
#     "Hypersensitivity",
#     "demodicosis",
#     "ringworm",
# ]


def load_model():
    # Load Keras model from artifacts
    return keras.models.load_model(MODEL_PATH)


cnn_model = None
last_load_error: str | None = None


def try_load_model() -> bool:
    """Attempt to load the CNN model and record any error. Returns True on success."""
    global cnn_model, last_load_error
    try:
        cnn_model = load_model()
        last_load_error = None
        return True
    except Exception:
        import traceback as _tb

        last_load_error = _tb.format_exc()
        cnn_model = None
        return False


# Try loading at import/startup (non-blocking apps should still call reload endpoint if needed)
try_load_model()


@router.get("/predict-cnn/status")
def predict_cnn_status():
    exists = os.path.exists(MODEL_PATH)
    return {"model_loaded": cnn_model is not None, "model_path": MODEL_PATH, "model_file_exists": exists, "last_error": last_load_error}


@router.post("/predict-cnn/reload")
def predict_cnn_reload():
    ok = try_load_model()
    return {"model_loaded": ok, "model_path": MODEL_PATH, "error": last_load_error}


def _preprocess_image_bytes(data: bytes, img_size=(224, 224)) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.fit(img, img_size, method=Image.BILINEAR)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


from io import BytesIO


@router.post("/predict-cnn")
def predict_cnn(file: UploadFile = File(...), top_k: int = 3):
    """Upload an image file and get CNN classification results.

    Returns predicted class, confidence, and top_k list.
    """
    try:
        if cnn_model is None:
            return {"detail": f"CNN model not loaded or missing file: {MODEL_PATH}"}

        data = file.file.read()
        img_size = (224, 224)
        x = _preprocess_image_bytes(data, img_size=img_size)

        probs = cnn_model.predict(x, verbose=0)[0]
        idx = np.argsort(probs)[::-1][:top_k]

        return {
            "predicted_class": CLASS_NAMES[int(idx[0])],
            "confidence": float(probs[int(idx[0])]),
            "top_k": [{"class": CLASS_NAMES[int(i)], "prob": float(probs[int(i)])} for i in idx],
        }

    except Exception as e:
        traceback.print_exc()
        return {"detail": "CNN inference failed", "error": str(e)}
