from app.api.v1.response_formatter import format_chat_response
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import traceback

from app.api.v1.nlp_utils import build_model_input_from_text
from app.models.lstm.loader import _LSTM_MODELS, load_lstm_model  # use your loader
SEQ_LEN = 8
DIAG_CLASSES = [ "FleasTicks", "Worms", "MildRespiratory", "AllergicDermatitis", "Gastroenteritis" ] 
LSTM_MODEL_NAMES = ["model_1_baseline", "model_lstm64", "model_dropout", "model_dense", "model_final"]

router = APIRouter()


# -------------------------------
# Request body model
# -------------------------------
class PetDiagnosisRequest(BaseModel):
    user_text: str
    species: int
    breed: int
    sex: int
    neutered: int
    age_years: float
    weight_kg: float
    season: str  # "rainy" | "summer" | "winter"


# -------------------------------
# Prediction endpoint
# -------------------------------
@router.post("/predict")
def predict_pet_diagnosis(req: PetDiagnosisRequest):
    try:
        # Get model from cached registry
        model = load_lstm_model("model_1_baseline")

        # Build static info dict
        pet_static_info = {
            "species": req.species,
            "breed": req.breed,
            "sex": req.sex,
            "neutered": req.neutered,
            "age_years": req.age_years,
            "weight_kg": req.weight_kg
        }

        # Build model input
        X_input = build_model_input_from_text(
            user_text=req.user_text,
            pet_static_info=pet_static_info,
            season=req.season,
            seq_len=SEQ_LEN
        )

        # Debugging
        print("X_input.shape:", X_input.shape)
        print("X_input dtype:", X_input.dtype)

        # Run prediction
        probs = model.predict(X_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(probs))

        # Return structured response
        # return {
        #     "predicted_class": DIAG_CLASSES[pred_class_idx],
        #     "confidence": {cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)}
        # }

        confidence_map = {
            cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)
        }

        predicted_class = DIAG_CLASSES[pred_class_idx]

        chat_message = format_chat_response(
            predicted_class=predicted_class,
            confidence_map=confidence_map
        )

        return {
            "diagnosis": predicted_class,
            "confidence": confidence_map,
            "message": chat_message
        }

    except Exception as e:
        traceback.print_exc()
        return {"detail": "Inference failed", "error": str(e)}


@router.post("/predict-lstm64")
def predict_pet_diagnosis_lstm64(req: PetDiagnosisRequest):
    try:
        # Get model from cached registry
        model = load_lstm_model("model_lstm64")

        # Build static info dict
        pet_static_info = {
            "species": req.species,
            "breed": req.breed,
            "sex": req.sex,
            "neutered": req.neutered,
            "age_years": req.age_years,
            "weight_kg": req.weight_kg
        }

        # Build model input
        X_input = build_model_input_from_text(
            user_text=req.user_text,
            pet_static_info=pet_static_info,
            season=req.season,
            seq_len=SEQ_LEN
        )

        # Debugging
        print("X_input.shape:", X_input.shape)
        print("X_input dtype:", X_input.dtype)

        # Run prediction
        probs = model.predict(X_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(probs))

        confidence_map = {
            cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)
        }

        predicted_class = DIAG_CLASSES[pred_class_idx]

        chat_message = format_chat_response(
            predicted_class=predicted_class,
            confidence_map=confidence_map
        )

        return {
            "diagnosis": predicted_class,
            "confidence": confidence_map,
            "message": chat_message
        }

    except Exception as e:
        traceback.print_exc()
        return {"detail": "Inference failed", "error": str(e)}
    

@router.post("/predict-model_dropout")
def predict_pet_diagnosis_dropout(req: PetDiagnosisRequest):
    try:
        # Get model from cached registry
        model = load_lstm_model("model_dropout")
        # Build static info dict
        pet_static_info = {
            "species": req.species,
            "breed": req.breed,
            "sex": req.sex,
            "neutered": req.neutered,
            "age_years": req.age_years,
            "weight_kg": req.weight_kg
        }
        # Build model input
        X_input = build_model_input_from_text(
            user_text=req.user_text,
            pet_static_info=pet_static_info,
            season=req.season,
            seq_len=SEQ_LEN
        )
        # Debugging
        print("X_input.shape:", X_input.shape)
        print("X_input dtype:", X_input.dtype)
        # Run prediction
        probs = model.predict(X_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(probs))
        confidence_map = {
            cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)
        }
        predicted_class = DIAG_CLASSES[pred_class_idx]
        chat_message = format_chat_response(
            predicted_class=predicted_class,
            confidence_map=confidence_map
        )
        return {
            "diagnosis": predicted_class,
            "confidence": confidence_map,
            "message": chat_message
        }
    except Exception as e:
        traceback.print_exc()
        return {"detail": "Inference failed", "error": str(e)}
    
@router.post("/predict-model_dense")
def predict_pet_diagnosis_dense(req: PetDiagnosisRequest):
    try:
        # Get model from cached registry
        model = load_lstm_model("model_dense")
        # Build static info dict
        pet_static_info = {
            "species": req.species,
            "breed": req.breed,
            "sex": req.sex,
            "neutered": req.neutered,
            "age_years": req.age_years,
            "weight_kg": req.weight_kg
        }
        # Build model input
        X_input = build_model_input_from_text(
            user_text=req.user_text,
            pet_static_info=pet_static_info,
            season=req.season,
            seq_len=SEQ_LEN
        )
        # Debugging
        print("X_input.shape:", X_input.shape)
        print("X_input dtype:", X_input.dtype)
        # Run prediction
        probs = model.predict(X_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(probs))
        confidence_map = {
            cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)
        }
        predicted_class = DIAG_CLASSES[pred_class_idx]
        chat_message = format_chat_response(
            predicted_class=predicted_class,
            confidence_map=confidence_map
        )
        return {
            "diagnosis": predicted_class,
            "confidence": confidence_map,
            "message": chat_message
        }
    except Exception as e:
        traceback.print_exc()
        return {"detail": "Inference failed", "error": str(e)}
    
@router.post("/predict-model_final_lr_sched")
def predict_pet_diagnosis_final(req: PetDiagnosisRequest):
    try:
        # Get model from cached registry
        model = load_lstm_model("model_final")
        # Build static info dict
        pet_static_info = {
            "species": req.species,
            "breed": req.breed,
            "sex": req.sex,
            "neutered": req.neutered,
            "age_years": req.age_years,
            "weight_kg": req.weight_kg
        }
        # Build model input
        X_input = build_model_input_from_text(
            user_text=req.user_text,
            pet_static_info=pet_static_info,
            season=req.season,
            seq_len=SEQ_LEN
        )
        # Debugging
        print("X_input.shape:", X_input.shape)
        print("X_input dtype:", X_input.dtype)
        # Run prediction
        probs = model.predict(X_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(probs))
        confidence_map = {
            cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)
        }
        predicted_class = DIAG_CLASSES[pred_class_idx]
        chat_message = format_chat_response(
            predicted_class=predicted_class,
            confidence_map=confidence_map
        )
        return {
            "diagnosis": predicted_class,
            "confidence": confidence_map,
            "message": chat_message
        }
    except Exception as e:
        traceback.print_exc()
        return {"detail": "Inference failed", "error": str(e)}
    
@router.post("/predict-ensemble")
def predict_pet_diagnosis_ensemble(req: PetDiagnosisRequest):
    try:
        # --- Build static info dict ---
        pet_static_info = {
            "species": req.species,
            "breed": req.breed,
            "sex": req.sex,
            "neutered": req.neutered,
            "age_years": req.age_years,
            "weight_kg": req.weight_kg
        }

        # --- Build LSTM input ---
        X_input = build_model_input_from_text(
            user_text=req.user_text,
            pet_static_info=pet_static_info,
            season=req.season,
            seq_len=SEQ_LEN
        )

        # --- Run all models and collect probabilities ---
        probs_list = []
        for name in LSTM_MODEL_NAMES:
            model = load_lstm_model(name)
            probs = model.predict(X_input, verbose=0)[0]
            probs_list.append(probs)

        # --- Average the probabilities ---
        avg_probs = np.mean(probs_list, axis=0)
        pred_class_idx = int(np.argmax(avg_probs))
        predicted_class = DIAG_CLASSES[pred_class_idx]

        # --- Format confidence map ---
        confidence_map = {cls: float(p) for cls, p in zip(DIAG_CLASSES, avg_probs)}

        # --- Format chat response ---
        chat_message = format_chat_response(
            predicted_class=predicted_class,
            confidence_map=confidence_map
        )

        return {
            "diagnosis": predicted_class,
            "confidence": confidence_map,
            "message": chat_message
        }

    except Exception as e:
        traceback.print_exc()
        return {"detail": "Inference failed", "error": str(e)}