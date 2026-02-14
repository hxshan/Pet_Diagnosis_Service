from fastapi import APIRouter
import os
from pydantic import BaseModel
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import traceback

router = APIRouter()


MODEL_PATH = "artifacts/xgb_pet_diagnosis_model.json"

CLASS_NAMES = [
    "FleasTicks",
    "Worms",
    "MildRespiratory",
    "AllergicDermatitis",
    "Gastroenteritis"
]

FEATURE_COLUMNS = [
    "species",
    "breed",
    "sex",
    "neutered",
    "age_years",
    "weight_kg",
    "num_previous_visits",
    "prev_diagnosis_class",
    "days_since_last_visit",
    "chronic_flag",
    "itching",
    "vomiting",
    "diarrhea",
    "coughing",
    "sneezing",
    "fleas_seen",
    "worms_seen",
    "red_skin",
    "lethargy",
    "loss_appetite"
]


class XGBPetRequest(BaseModel):
    species: int
    breed: int
    sex: int
    neutered: int
    age_years: float
    weight_kg: float
    num_previous_visits: int
    prev_diagnosis_class: int
    days_since_last_visit: int
    chronic_flag: int
    itching: int
    vomiting: int
    diarrhea: int
    coughing: int
    sneezing: int
    fleas_seen: int
    worms_seen: int
    red_skin: int
    lethargy: int
    loss_appetite: int


def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


xgb_model = None
last_load_error: str | None = None


def try_load_model() -> bool:
    """Attempt to load the XGBoost model and record any error. Returns True on success."""
    global xgb_model, last_load_error
    try:
        xgb_model = load_model()
        last_load_error = None
        return True
    except Exception as e:
        import traceback as _tb
        last_load_error = _tb.format_exc()
        xgb_model = None
        return False


# Try loading at import/startup
try_load_model()


@router.get('/predict-xgb/status')
def predict_xgb_status():
    """Return status of XGBoost model load and whether the model file exists."""
    exists = os.path.exists(MODEL_PATH)
    return {
        "model_loaded": xgb_model is not None,
        "model_path": MODEL_PATH,
        "model_file_exists": exists
    }


@router.post('/predict-xgb/reload')
def predict_xgb_reload():
    """Attempt to reload the XGBoost model on demand. Returns load success and any error."""
    ok = try_load_model()
    return {"model_loaded": ok, "model_path": MODEL_PATH, "error": last_load_error}


@router.post("/predict-xgb")
def predict_xgb(req: XGBPetRequest, confidence_threshold: float = 0.4):
    try:
        if xgb_model is None:
            return {"detail": "XGBoost model not loaded or missing file: %s" % MODEL_PATH}

        input_data = req.dict()
        df_input = pd.DataFrame([input_data])
        # Ensure correct column order
        df_input = df_input[FEATURE_COLUMNS]

        probs = xgb_model.predict_proba(df_input)[0]
        pred_class = int(np.argmax(probs))
        max_conf = float(np.max(probs))

        if max_conf < confidence_threshold:
            return {
                "status": "uncertain",
                "confidence": max_conf,
                "message": "Low confidence prediction. Further evaluation recommended."
            }

        return {
            "status": "predicted",
            "predicted_class_index": pred_class,
            "predicted_class_name": CLASS_NAMES[pred_class],
            "confidence": max_conf,
            "all_class_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        }

    except Exception as e:
        traceback.print_exc()
        return {"detail": "XGBoost inference failed", "error": str(e)}
