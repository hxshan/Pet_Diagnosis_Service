from fastapi import APIRouter
import os
import pickle
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from xgboost import XGBClassifier

router = APIRouter()

# MODEL_PATH = "artifacts/xgb_vet_verified_3000.json"
# META_PATH  = "artifacts/xgb_vet_verified_3000_meta.pkl"

# MODEL_PATH = "artifacts/xgb_vet_verified_3000_v2.json"
# META_PATH  = "artifacts/xgb_vet_verified_3000_v2_meta.pkl"

MODEL_PATH = "artifacts/xgb_vet_verified_3000_v3.json"
META_PATH  = "artifacts/xgb_vet_verified_3000_v3_meta.pkl"

# ── Runtime state ─────────────────────────────────────────────────────────────
xgb_model       = None
feature_columns = None
label_encoders  = None
diag_classes    = None
last_load_error: str | None = None


def try_load_model() -> bool:
    """Load the XGBoost model and meta (label encoders, feature columns, class names)."""
    global xgb_model, feature_columns, label_encoders, diag_classes, last_load_error
    try:
        model = XGBClassifier()
        model.load_model(MODEL_PATH)

        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)

        xgb_model       = model
        feature_columns = meta["feature_columns"]
        label_encoders  = meta["label_encoders"]
        diag_classes    = meta["diag_classes"]
        last_load_error = None
        return True
    except Exception:
        last_load_error = traceback.format_exc()
        xgb_model = None
        return False


# Try at import / startup
try_load_model()


# ── Request schema ─────────────────────────────────────────────────────────────
# Categorical fields accept readable strings (e.g. "dog", "cat", "male", "yes").
# Numeric fields are ints or floats as in training.
class XGBVetRequest(BaseModel):
    # Categorical (will be label-encoded by stored LabelEncoder)
    species:  str
    breed:    str
    sex:      str
    neutered: str

    # Numeric — core
    age_years:  float
    weight_kg:  float
    vaccinated: int = 0

    # Visit history
    num_previous_visits:   int   = 0
    prev_diagnosis_class:  int   = -1
    days_since_last_visit: int   = 0
    chronic_flag:          int   = 0

    # Gastro / general
    vomiting:     int = 0
    diarrhea:     int = 0
    dehydration:  int = 0
    loss_appetite:int = 0
    fever:        int = 0
    lethargy:     int = 0

    # Skin
    itching:      int = 0
    red_skin:     int = 0
    hair_loss:    int = 0
    skin_lesions: int = 0
    wounds:       int = 0

    # Tick fever indicators
    dark_urine:    int = 0
    pale_gums:     int = 0
    pale_eyelids:  int = 0
    tick_exposure: int = 0

    # UTI
    pain_urinating:     int = 0
    frequent_urination: int = 0
    blood_in_urine:     int = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_case(raw_dict: dict) -> pd.DataFrame:
    """Apply stored LabelEncoders to categorical columns, return ordered DataFrame."""
    d = raw_dict.copy()
    for col, le in label_encoders.items():
        if col in d:
            try:
                d[col] = int(le.transform([d[col]])[0])
            except ValueError:
                # Unknown label — use 0 as fallback and note it
                d[col] = 0
    df = pd.DataFrame([d])
    df = df[feature_columns]
    return df


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/predict-xgb-v2/status")
def predict_xgb_v2_status():
    """Return load status for the v2 XGBoost model and meta files."""
    return {
        "model_loaded":      xgb_model is not None,
        "model_path":        MODEL_PATH,
        "meta_path":         META_PATH,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "meta_file_exists":  os.path.exists(META_PATH),
        "feature_columns":   feature_columns,
        "diag_classes":      diag_classes,
        "last_error":        last_load_error,
    }


@router.post("/predict-xgb-v2/reload")
def predict_xgb_v2_reload():
    """Reload the v2 model and meta at runtime (no server restart needed)."""
    ok = try_load_model()
    return {"model_loaded": ok, "model_path": MODEL_PATH, "error": last_load_error}


@router.post("/predict-xgb-v2")
def predict_xgb_v2(req: XGBVetRequest, confidence_threshold: float = 0.35):
    """Run inference using the new vet-verified XGBoost model.

    Categorical fields (species, breed, sex, neutered) should be human-readable
    strings matching the training data (e.g. 'dog', 'cat', 'male', 'female', 'yes', 'no').

    Returns the top predicted class, full probability map and top-3 differentials.
    """
    try:
        if xgb_model is None:
            ok = try_load_model()
            if not ok:
                return {"detail": f"Model not loaded. Error: {last_load_error}"}

        raw = req.dict()
        df_input = encode_case(raw)

        probs      = xgb_model.predict_proba(df_input)[0]
        pred_idx   = int(np.argmax(probs))
        max_conf   = float(probs[pred_idx])

        # Build probability map
        prob_map = {diag_classes[i]: float(probs[i]) for i in range(len(diag_classes))}

        # Top-3 differentials sorted by probability descending
        sorted_idx  = np.argsort(probs)[::-1]
        top_3 = [
            {"class": diag_classes[int(i)], "probability": float(probs[int(i)])}
            for i in sorted_idx[:3]
        ]

        if max_conf < confidence_threshold:
            return {
                "status":      "uncertain",
                "confidence":  max_conf,
                "top_3":       top_3,
                "probability_map": prob_map,
                "message":     "Low confidence — further veterinary evaluation recommended.",
            }

        return {
            "status":           "predicted",
            "predicted_class":  diag_classes[pred_idx],
            "confidence":       max_conf,
            "top_3":            top_3,
            "probability_map":  prob_map,
        }

    except Exception as e:
        traceback.print_exc()
        return {"detail": "XGBoost v2 inference failed", "error": str(e)}
