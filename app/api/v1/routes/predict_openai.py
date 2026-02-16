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
import requests

# optional local Llama support (llama-cpp-python)
_HAS_LLAMA = False
try:
    from llama_cpp import Llama
    _HAS_LLAMA = True
except Exception:
    _HAS_LLAMA = False

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
    the original /predict endpoints. Otherwise we return the LLM chat reply
    """
    try:
        # Determine which LLM source to use: local Llama (fast, if enabled),
        # then OpenRouter (hosted, via key), then OpenAI as fallback.

        user_text = req.user_text

        # Build classifier prompt (we'll reuse for whichever backend we call)
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

        content = None

        # 1) Local Llama (fast, low-latency) if enabled and available
        if settings.local_llm_enabled and _HAS_LLAMA and settings.local_llm_model_path:
            try:
                llama = Llama(model_path=settings.local_llm_model_path)
                # simple prompt: system + user
                prompt = system_prompt + "\nUSER: " + user_text
                resp = llama.create(prompt=prompt, max_tokens=150, temperature=0.0)
                # llama-cpp-python returns choices with 'text' in many versions
                try:
                    content = resp['choices'][0]['text']
                except Exception:
                    try:
                        content = resp.choices[0].text
                    except Exception:
                        content = str(resp)
            except Exception as e:
                # if local Llama fails, fall back to hosted options
                content = None

        # 2) OpenRouter hosted (if key present)
        if content is None and (settings.openrouter_api_key or os.getenv('OPENROUTER_API_KEY')):
            or_key = settings.openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
            model_name = settings.openrouter_model or "gpt-3.5-turbo"
            try:
                headers = {
                    "Authorization": f"Bearer {or_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 150
                }
                r = requests.post("https://api.openrouter.ai/v1/chat/completions", headers=headers, json=payload, timeout=10)
                jr = r.json()
                try:
                    # attempt OpenAI-like shape
                    content = jr['choices'][0]['message']['content'][0]['text']
                except Exception:
                    try:
                        content = jr['choices'][0]['message']['content']
                    except Exception:
                        content = str(jr)
            except Exception:
                content = None

        # 3) Fallback: OpenAI (existing handling)
        if content is None:
            key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                return {"detail": "No LLM configured: enable local LLM or set OPENROUTER_API_KEY/OPENAI_API_KEY"}

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

            # Try local Llama chat if available and enabled
            chat_content = None
            if settings.local_llm_enabled and _HAS_LLAMA and settings.local_llm_model_path:
                try:
                    llama = Llama(model_path=settings.local_llm_model_path)
                    prompt = "You are a helpful vet-assistant. Keep answers short and friendly.\nUSER: " + user_text
                    cresp = llama.create(prompt=prompt, max_tokens=200, temperature=0.6)
                    try:
                        chat_content = cresp['choices'][0]['text']
                    except Exception:
                        try:
                            chat_content = cresp.choices[0].text
                        except Exception:
                            chat_content = str(cresp)
                except Exception:
                    chat_content = None

            # If no local chat result, try OpenRouter
            if chat_content is None and (settings.openrouter_api_key or os.getenv('OPENROUTER_API_KEY')):
                or_key = settings.openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
                model_name = settings.openrouter_model or "gpt-3.5-turbo"
                try:
                    headers = {
                        "Authorization": f"Bearer {or_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model_name,
                        "messages": chat_messages,
                        "temperature": 0.6,
                        "max_tokens": 200
                    }
                    r = requests.post("https://api.openrouter.ai/v1/chat/completions", headers=headers, json=payload, timeout=10)
                    jr = r.json()
                    try:
                        chat_content = jr['choices'][0]['message']['content'][0]['text']
                    except Exception:
                        try:
                            chat_content = jr['choices'][0]['message']['content']
                        except Exception:
                            chat_content = str(jr)
                except Exception:
                    chat_content = None

            # Fallback to OpenAI if still no chat content
            if chat_content is None:
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
        return {"detail": "LLM integration failed", "error": str(e)}
