from pydantic import BaseModel, ConfigDict
from typing import Optional
import numpy as np

class PredictionRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    apparent_temperature: float
    humidity: float
    wind_speed: float
    wind_bearing: float
    visibility: float
    loud_cover: float
    pressure: float
    year: int
    month: int
    date: int

class PredictionResponse(BaseModel):
    prediction: float

class TrainingStatus(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    status: str
    message: str
    evals_data: Optional[object] = None  # Allows None, defaults to None
    weights: Optional[np.ndarray] = None  # Allows None, defaults to None
    bias: Optional[float] = None          # Allows None, defaults to None
    model: Optional[object] = None  