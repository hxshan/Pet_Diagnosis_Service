from fastapi import APIRouter
import app.api.v1.routes.predict as predict
import app.api.v1.routes.predict_openrouter as predict_openrouter
import app.api.v1.routes.predict_xgb as predict_xgb

api_router = APIRouter()

api_router.include_router(
    predict.router,
    prefix="",
)

api_router.include_router(
    predict_openrouter.router,
    prefix="",
)

api_router.include_router(
    predict_xgb.router,
    prefix="",
)
