from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.router import api_router
from app.models.lstm.loader import load_lstm_model

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
    Load ML models once at startup.
    """
    load_lstm_model("model_1_baseline")

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "environment": settings.app_env
    }
