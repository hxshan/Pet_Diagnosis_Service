from fastapi import APIRouter
import os
from app.core.config import settings
import json
import re
import traceback
import requests

import numpy as np
import pandas as pd

from app.api.v1.nlp_utils import extract_time_features_from_text
from app.api.v1.routes.predict_xgb import (
    xgb_model as _xgb_model,
    try_load_model as _try_load_xgb,
    CLASS_NAMES as XGB_CLASS_NAMES,
    FEATURE_COLUMNS as XGB_FEATURE_COLUMNS,
)

from app.api.v1.routes.predict import PetDiagnosisRequest
from app.api.v1.nlp_utils import build_model_input_from_text
from app.api.v1.response_formatter import format_chat_response
from app.models.lstm.loader import load_lstm_model

router = APIRouter()


def _extract_text_from_choice(jresp):
    try:
        # common OpenRouter / OpenAI-like shape: choices -> message -> content (list/dict)
        c = jresp['choices'][0]['message']['content']
        if isinstance(c, list) and len(c) > 0:
            first = c[0]
            if isinstance(first, dict) and 'text' in first:
                return first['text']
            if isinstance(first, str):
                return first
        if isinstance(c, str):
            return c
    except Exception:
        pass
    try:
        # older/simple shape: choices[0]['text']
        return jresp['choices'][0]['text']
    except Exception:
        return None

def _extract_json_from_text(text: str):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        return None


@router.post("/predict-openrouter")
def predict_with_openrouter(req: PetDiagnosisRequest):
    """
    Route that uses OpenRouter to classify intent and generate chat.
    If OpenRouter classifies the message as a diagnosis request, run the
    existing LSTM pipeline and return the diagnosis. Otherwise return
    the OpenRouter chat reply.
    """
    try:
        # Read OpenRouter key (prefer Settings)
        or_key = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not or_key:
            return {"detail": "OPENROUTER_API_KEY not configured in environment or Settings"}

        user_text = req.user_text

        system_prompt = (
            "You are a strict classifier. Answer with a JSON object only.\n"
            "Given the user's message, return: {\"is_diagnosis\": true|false, \"explanation\": \"short reason\"}.\n"
            "A diagnosis request is when the user asks for a medical/health diagnosis for their pet (e.g., symptoms, possible causes, what it might be).\n"
            "Do not include any other text outside the JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        # Call OpenRouter to classify intent
        model_name = settings.openrouter_model or "gpt-3.5-turbo"
        headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": messages, "temperature": 0.0, "max_tokens": 150}
        # OpenRouter endpoint per docs (note: openrouter.ai, not api.openrouter.ai)
        openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        r = requests.post(openrouter_url, headers=headers, json=payload, timeout=10)
        jr = r.json()

        # Parse response content robustly across a few shapes OpenRouter may return
        def _extract_text_from_choice(jresp):
            try:
                # common OpenRouter / OpenAI-like shape: choices -> message -> content (list/dict)
                c = jresp['choices'][0]['message']['content']
                if isinstance(c, list) and len(c) > 0:
                    first = c[0]
                    if isinstance(first, dict) and 'text' in first:
                        return first['text']
                    if isinstance(first, str):
                        return first
                if isinstance(c, str):
                    return c
            except Exception:
                pass
            try:
                # older/simple shape: choices[0]['text']
                return jresp['choices'][0]['text']
            except Exception:
                return None

        content = _extract_text_from_choice(jr) or str(jr)

        parsed = _extract_json_from_text(content)

        if not parsed or not isinstance(parsed, dict):
            # treat as chat content
            return {"type": "chat", "score": None, "result": content}

        is_diag = bool(parsed.get("is_diagnosis"))
        explanation = parsed.get("explanation", "")

        if is_diag:
            pet_static_info = {
                "species": req.species,
                "breed": req.breed,
                "sex": req.sex,
                "neutered": req.neutered,
                "age_years": req.age_years,
                "weight_kg": req.weight_kg
            }

            X_input = build_model_input_from_text(
                user_text=req.user_text,
                pet_static_info=pet_static_info,
                season=req.season,
                seq_len=8
            )

            model = load_lstm_model("model_1_baseline")
            probs = model.predict(X_input, verbose=0)[0]
            DIAG_CLASSES = ["FleasTicks", "Worms", "MildRespiratory", "AllergicDermatitis", "Gastroenteritis"]
            pred_class_idx = int(float(probs.argmax()))
            confidence_map = {cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)}
            predicted_class = DIAG_CLASSES[pred_class_idx]

            chat_message = format_chat_response(predicted_class=predicted_class, confidence_map=confidence_map)

            return {
                "diagnosis": predicted_class,
                "confidence": confidence_map,
                "message": chat_message,
                "openrouter_explanation": explanation
            }

        # Otherwise generate a short chat reply using OpenRouter
        chat_messages = [{"role": "system", "content": "You are a helpful vet-assistant. Keep answers short and friendly."}, {"role": "user", "content": user_text}]
        payload = {"model": model_name, "messages": chat_messages, "temperature": 0.6, "max_tokens": 200}
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=10)
        jr = r.json()
        chat_content = _extract_text_from_choice(jr) or str(jr)

        return {"type": "chat", "score": None, "result": chat_content, "openrouter_explanation": explanation}

    except Exception as e:
        traceback.print_exc()
        return {"detail": "OpenRouter integration failed", "error": str(e)}


@router.post("/predict-openrouter-xgb")
def predict_openrouter_xgb(req: PetDiagnosisRequest, confidence_threshold: float = 0.4):
    """Classify intent with OpenRouter; if diagnosis, run XGBoost and ask OpenRouter to
    generate a short differential-diagnosis style message that includes probabilities.
    """
    try:
        or_key = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not or_key:
            return {"detail": "OPENROUTER_API_KEY not configured in environment or Settings"}

        user_text = req.user_text

        # 1) classify intent (strict JSON response expected)
        system_prompt = (
            "You are a strict classifier. Answer with a JSON object only.\n"
            "Given the user's message, return: {\"is_diagnosis\": true|false, \"explanation\": \"short reason\"}.\n"
            "A diagnosis request is when the user asks for a medical/health diagnosis for their pet (e.g., symptoms, possible causes, what it might be).\n"
            "Do not include any other text outside the JSON."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]

        model_name = settings.openrouter_model or "gpt-3.5-turbo"
        headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": messages, "temperature": 0.0, "max_tokens": 150}
        openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        r = requests.post(openrouter_url, headers=headers, json=payload, timeout=10)
        jr = r.json()
        content = _extract_text_from_choice(jr) or str(jr)
        parsed = _extract_json_from_text(content)

        # If not clearly a diagnosis request, return a normal chat reply
        if not parsed or not isinstance(parsed, dict) or not bool(parsed.get("is_diagnosis")):
            chat_messages = [{"role": "system", "content": "You are a helpful vet-assistant. Keep answers short and friendly."}, {"role": "user", "content": user_text}]
            payload = {"model": model_name, "messages": chat_messages, "temperature": 0.6, "max_tokens": 200}
            r = requests.post(openrouter_url, headers=headers, json=payload, timeout=10)
            jr = r.json()
            chat_content = _extract_text_from_choice(jr) or str(jr)
            return {"type": "chat", "score": None, "result": chat_content, "openrouter_explanation": parsed.get("explanation") if isinstance(parsed, dict) else None}

        # It's a diagnosis request -> prepare XGBoost input
        time_feats = extract_time_features_from_text(user_text)

        input_data = {
            "species": req.species,
            "breed": req.breed,
            "sex": req.sex,
            "neutered": req.neutered,
            "age_years": req.age_years,
            "weight_kg": req.weight_kg,
            "num_previous_visits": 0,
            "prev_diagnosis_class": 0,
            "days_since_last_visit": time_feats.get("last_vet_visit_days_ago", 0),
            "chronic_flag": 0,
            "itching": int(time_feats.get("itching", 0)),
            "vomiting": int(time_feats.get("vomiting", 0)),
            "diarrhea": int(time_feats.get("diarrhea", 0)),
            "coughing": int(time_feats.get("coughing", 0)),
            "sneezing": int(time_feats.get("sneezing", 0)),
            "fleas_seen": int(time_feats.get("fleas_seen", 0)),
            "worms_seen": int(time_feats.get("worms_seen", 0)),
            "red_skin": int(time_feats.get("red_skin", 0)),
            "lethargy": int(time_feats.get("lethargy", 0)),
            "loss_appetite": int(time_feats.get("loss_appetite", 0)),
        }

        # Ensure model loaded
        if _xgb_model is None:
            ok = _try_load_xgb()
            if not ok:
                return {"detail": "XGBoost model not loaded", "error": "Model not available"}

        df_input = pd.DataFrame([input_data])
        # reorder columns to expected features
        df_input = df_input[XGB_FEATURE_COLUMNS]

        probs = _xgb_model.predict_proba(df_input)[0]
        pred_idx = int(np.argmax(probs))
        confidence_map = {XGB_CLASS_NAMES[i]: float(probs[i]) for i in range(len(XGB_CLASS_NAMES))}

        # Build a short prompt for LLM to generate the human-facing differential diagnosis
        prob_list_text = ", ".join([f"{name}: {confidence_map[name]*100:.1f}%" for name in XGB_CLASS_NAMES])
        # Instruct the LLM to be professional and clinical — avoid apologetic language, state intent, give top diagnosis and other close possibilities
        llm_user_msg = (
            f"User said: {user_text}\n\n"
            f"Model probabilities: {prob_list_text}\n\n"
            "Produce a professional, clinical assessment for a pet owner. Requirements:\n"
            "1) Begin with a one-line 'Assessment:' that names the most likely diagnosis and its percentage.\n"
            "2) Next, include 'Top differentials:' and list the two next most likely causes with percentages and one-line rationale each.\n"
            "3) Finish with a single 'Recommendation:' line advising veterinary evaluation.\n"
            "Do NOT use apologetic wording (e.g., 'I'm sorry'). Keep language concise, factual, and neutral. Limit output to 4-6 short lines."
        )

        chat_messages = [{"role": "system", "content": "You are a professional veterinary assistant. Use concise, clinical language. Do not include apologies or overly empathetic phrasing."}, {"role": "user", "content": llm_user_msg}]
        payload = {"model": model_name, "messages": chat_messages, "temperature": 0.6, "max_tokens": 250}
        r = requests.post(openrouter_url, headers=headers, json=payload, timeout=10)
        jr = r.json()
        assistant_text = _extract_text_from_choice(jr) or str(jr)

        return {
            "diagnosis_suggested": XGB_CLASS_NAMES[pred_idx],
            "confidence_map": confidence_map,
            "message": assistant_text,
            "openrouter_explanation": parsed.get("explanation") if isinstance(parsed, dict) else None
        }

    except Exception as e:
        traceback.print_exc()
        return {"detail": "OpenRouter+XGB integration failed", "error": str(e)}
