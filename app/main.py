from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.router import api_router
from app.models.lstm.loader import load_lstm_model
import threading
import traceback

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(
    api_router,
    prefix="/api/v1"
)

@app.on_event("startup")
def startup_event():
    """
    Kick off model loading in a background thread so startup doesn't block or cause worker churn.
    Any exceptions during model load are caught and logged; the model loader can still be invoked
    on-demand later (e.g., via reload endpoints).
    """

    def _load():
        try:
            load_lstm_model("model_1_baseline")
        except Exception:
            traceback.print_exc()

    t = threading.Thread(target=_load, daemon=True)
    t.start()

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "environment": settings.app_env
    }
