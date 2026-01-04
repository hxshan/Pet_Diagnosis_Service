from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    user_text: str
    species: int
    breed: int
    sex: int
    neutered: int
    age_years: float
    weight_kg: float
    season: str  # "rainy" | "summer" | "winter"
        
class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str