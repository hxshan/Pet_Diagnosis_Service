from fastapi import APIRouter
import os
from pydantic import BaseModel
from typing import List, Optional
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

# Map common symptom IDs (from UI) to model feature column names. Keep identity mapping by default.
SYMPTOM_ID_MAP = {
    'itching': 'itching',
    'vomiting': 'vomiting',
    'diarrhea': 'diarrhea',
    'coughing': 'coughing',
    'sneezing': 'sneezing',
    'fleas_seen': 'fleas_seen',
    'worms_seen': 'worms_seen',
    'red_skin': 'red_skin',
    'lethargy': 'lethargy',
    'loss_appetite': 'loss_appetite',
}


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


class SymptomCheckerRequest(BaseModel):
    # Minimal, friendly request used by the mobile/web symptom checker UI.
    # Fields are optional where reasonable; defaults will be used when missing.
    species: Optional[int] = 0
    breed: Optional[int] = 0
    sex: Optional[int] = 0
    neutered: Optional[int] = 0
    age_years: Optional[float] = 1.0
    weight_kg: Optional[float] = 1.0
    num_previous_visits: Optional[int] = 0
    prev_diagnosis_class: Optional[int] = 0
    days_since_last_visit: Optional[int] = 9999
    chronic_flag: Optional[int] = 0
    # symptoms is a list of symptom keys (e.g., 'itching', 'vomiting', 'diarrhea')
    symptoms: Optional[List[str]] = []
    # optional metadata collected in the UI
    severity: Optional[str] = None
    duration_days: Optional[int] = 0


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


@router.post('/predict-xgb/symptom-checker')
def predict_xgb_symptom_checker(req: SymptomCheckerRequest, top_k: int = 3, confidence_threshold: float = 0.35):
    """Endpoint tailored for the mobile/web symptom checker UI.

    Accepts a compact `SymptomCheckerRequest` (list of symptom keys plus a few optional pet metadata fields),
    maps the symptoms into the XGBoost feature vector expected by the model, runs inference and returns the
    top-k predictions along with the full probability map. The route uses a conservative default confidence threshold
    to mark uncertain predictions.
    """
    try:
        if xgb_model is None:
            return {"detail": "XGBoost model not loaded or missing file: %s" % MODEL_PATH}

        # Start with default values for the feature vector
        feature = {c: 0 for c in FEATURE_COLUMNS}

        # Copy numeric fields where provided
        feature.update({
            "species": int(req.species or 0),
            "breed": int(req.breed or 0),
            "sex": int(req.sex or 0),
            "neutered": int(req.neutered or 0),
            "age_years": float(req.age_years or 0.0),
            "weight_kg": float(req.weight_kg or 0.0),
            "num_previous_visits": int(req.num_previous_visits or 0),
            "prev_diagnosis_class": int(req.prev_diagnosis_class or 0),
            "days_since_last_visit": int(req.days_since_last_visit or 0),
            "chronic_flag": int(req.chronic_flag or 0),
        })

        # Map symptom keys (strings) to binary features if present in FEATURE_COLUMNS
        # Accept symptoms like 'itching', 'vomiting', etc.
        for s in (req.symptoms or []):
            raw = str(s).strip()
            # Try direct symptom id map first
            mapped = SYMPTOM_ID_MAP.get(raw)
            if mapped and mapped in feature:
                feature[mapped] = 1
                continue

            # Try normalized key (lowercase, underscores)
            norm = raw.lower().replace(' ', '_')
            mapped = SYMPTOM_ID_MAP.get(norm) or norm
            if mapped in feature:
                feature[mapped] = 1
                continue

            # If not recognized, ignore the symptom but log to stderr for debugging
            # (do not raise; tolerate unknown symptom keys from UI)
            # print(f"Unmapped symptom key: {raw}")

        # Build dataframe with correct column order
        df_input = pd.DataFrame([feature])[FEATURE_COLUMNS]

        probs = xgb_model.predict_proba(df_input)[0]
        # sort classes by probability descending
        ranked = sorted([(i, float(probs[i]), CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))], key=lambda x: x[1], reverse=True)

        top = []
        for idx, p, name in ranked[:top_k]:
            top.append({"class_index": int(idx), "class_name": name, "confidence": p})

        max_conf = top[0]["confidence"] if top else 0.0

        if max_conf < confidence_threshold:
            return {
                "status": "uncertain",
                "confidence": max_conf,
                "message": "Low confidence prediction. Consider collecting more details or uploading a photo.",
                "predictions": top,
                "all_class_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
            }

        return {
            "status": "predicted",
            "predictions": top,
            "confidence": max_conf,
            "all_class_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        }

    except Exception:
        traceback.print_exc()
        return {"detail": "XGBoost symptom-checker inference failed"}
