from fastapi import APIRouter
import os
from app.core.config import settings
import json
import re
import traceback

try:
    # new OpenAI >=1.0 interface
    from openai import OpenAI
    _OPENAI_CLIENT_CLASS = OpenAI
    _USE_NEW_OPENAI = True
except Exception:
    import openai
    _OPENAI_CLIENT_CLASS = None
    _USE_NEW_OPENAI = False

from app.api.v1.routes.predict import PetDiagnosisRequest
from app.api.v1.nlp_utils import build_model_input_from_text
from app.api.v1.response_formatter import format_chat_response
from app.models.lstm.loader import load_lstm_model

router = APIRouter()


def _extract_json_from_text(text: str):
    # Try to find a JSON object in the model text response
    try:
        # simple regex to find {...}
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        return None


@router.post("/predict-openai")
def predict_with_openai(req: PetDiagnosisRequest):
    """
    New experimental route that uses OpenAI to classify whether the
    user wants a diagnosis. If OpenAI determines it's a diagnosis request
    we run the existing LSTM pipeline and return the same structure as
    the original /predict endpoints. Otherwise we return the LLM chat reply.
    """
    try:
        # Prefer the project Settings (reads .env via pydantic); fall back to os.getenv
        key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return {"detail": "OPENAI_API_KEY not configured in environment or Settings"}


        user_text = req.user_text

        # First: ask the model to classify intent and return JSON only
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

        # Create client and call the appropriate API depending on installed openai package
        if _USE_NEW_OPENAI and _OPENAI_CLIENT_CLASS is not None:
            client = _OPENAI_CLIENT_CLASS(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.0,
            )

            # extract text safely from new-style response
            try:
                content = resp.choices[0].message.content[0].text
            except Exception:
                try:
                    content = resp['choices'][0]['message']['content'][0]['text']
                except Exception:
                    content = str(resp)

        else:
            openai.api_key = key
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.0
            )
            content = resp["choices"][0]["message"]["content"].strip()

        parsed = _extract_json_from_text(content)

        # If parsing failed, be conservative: treat as non-diagnosis and return model's content
        if not parsed or not isinstance(parsed, dict):
            # Return the raw LLM reply as chat
            # Optionally: ask the model to generate a conversational reply
            return {"type": "chat", "score": None, "result": content}

        is_diag = bool(parsed.get("is_diagnosis"))
        explanation = parsed.get("explanation", "")

        if is_diag:
            # Build model input and run existing LSTM pipeline (same as /predict)
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
            DIAG_CLASSES = [ "FleasTicks", "Worms", "MildRespiratory", "AllergicDermatitis", "Gastroenteritis" ]
            pred_class_idx = int(float(probs.argmax()))
            confidence_map = {cls: float(p) for cls, p in zip(DIAG_CLASSES, probs)}
            predicted_class = DIAG_CLASSES[pred_class_idx]

            chat_message = format_chat_response(
                predicted_class=predicted_class,
                confidence_map=confidence_map
            )

            return {
                "diagnosis": predicted_class,
                "confidence": confidence_map,
                "message": chat_message,
                "openai_explanation": explanation
            }
        else:
            # Non-diagnosis: generate a friendly reply via the LLM
            # We'll ask the model to produce a helpful short reply
            chat_messages = [
                {"role": "system", "content": "You are a helpful vet-assistant. Keep answers short and friendly."},
                {"role": "user", "content": user_text}
            ]

            if _USE_NEW_OPENAI and _OPENAI_CLIENT_CLASS is not None:
                client = _OPENAI_CLIENT_CLASS(api_key=key)
                chat_resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=chat_messages,
                    max_tokens=200,
                    temperature=0.6,
                )
                try:
                    chat_content = chat_resp.choices[0].message.content[0].text
                except Exception:
                    try:
                        chat_content = chat_resp['choices'][0]['message']['content'][0]['text']
                    except Exception:
                        chat_content = str(chat_resp)
            else:
                chat_resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=chat_messages,
                    max_tokens=200,
                    temperature=0.6
                )
                chat_content = chat_resp["choices"][0]["message"]["content"].strip()
            return {"type": "chat", "score": None, "result": chat_content, "openai_explanation": explanation}

    except Exception as e:
        traceback.print_exc()
        return {"detail": "OpenAI integration failed", "error": str(e)}
