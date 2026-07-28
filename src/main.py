

import sys
import pandas as pd 

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.models import PredictionRequest, PredictionResponse
from src.exception import CustomException
from src.logger import logging
from src.models import TrainingStatus
from src.state_store import load_training_state

training_status = TrainingStatus(
    status="idle",
    message="Model not started yet",
    evals_data=None,
    weights=None,
    bias=None,
    model=None,
)
load_training_state(training_status)

from src.services import _run_training, predict_

app = FastAPI(
    title="Linear Regression",
    description="Linear Regression via gradient descent built from stratch on weather data to predict the weather",
)

templates = Jinja2Templates(directory="templates")
 


@app.get("/", response_class=HTMLResponse)
def home(req: Request):
    return templates.TemplateResponse(name="index.html", request=req)


@app.api_route("/train", methods=["GET", "POST"], response_class=HTMLResponse)
def train_model(req: Request, background_tasks: BackgroundTasks):
    if req.method == "POST" and training_status.status != "training":
        if training_status.status == "idle":
            training_status.status = "Model is Training..."
            background_tasks.add_task(_run_training)

    if req.method == "GET" and training_status.status == "idle":
        training_status.message = "Model not started yet"

    evals_data = training_status.evals_data or {}
    train_metrics = evals_data.get("train-metrics", {}) if isinstance(evals_data, dict) else {}
    test_metrics = evals_data.get("test-metrics", {}) if isinstance(evals_data, dict) else {}

    return templates.TemplateResponse(
        name="train.html",
        request=req,
        context={
            "status": training_status.status,
            "message": training_status.message,
            "evals_data": evals_data,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
    )


@app.get("/train-status") 
def train_status():
    return {
        "status": training_status.status,
        "message": training_status.message,
        "evals_data": training_status.evals_data,
    }


@app.get("/predict", response_class=HTMLResponse)
def predict_page(req: Request):
    return templates.TemplateResponse(name="predict.html", request=req)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        df = pd.DataFrame([payload.model_dump()])
        result = predict_(df)
        logging.info("The prediction completed with value: %s", result["prediction"])
        return PredictionResponse(prediction=result["prediction"])
    except Exception as e:
        raise CustomException(e, sys)

    