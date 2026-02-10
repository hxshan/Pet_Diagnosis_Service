from fastapi import APIRouter
import app.api.v1.routes.predict as predict
import app.api.v1.routes.predict_openai as predict_openai

api_router = APIRouter()

api_router.include_router(
    predict.router,
    prefix="",
)

api_router.include_router(
    predict_openai.router,
    prefix="",
)
