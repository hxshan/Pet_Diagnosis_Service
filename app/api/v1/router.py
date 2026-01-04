from fastapi import APIRouter
import app.api.v1.routes.predict as predict

api_router = APIRouter()

api_router.include_router(
    predict.router,
    prefix="",
)
