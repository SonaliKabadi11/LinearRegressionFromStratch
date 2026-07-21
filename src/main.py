

import threading

from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.services import _run_training, predict_, training_state

app = FastAPI(
    title="Linear Regression",
    description="Linear Regression via gradient descent built from stratch on weather data to predict the weather",
)

templates = Jinja2Templates(directory="templates")






@app.get("/", response_class=HTMLResponse)
def home(req: Request):
    return templates.TemplateResponse(name="index.html", request=req)


@app.get("/train")
def train_model(req: Request):
    if training_state["status"] != "training":
        if training_state["status"] == "idle":
            thread = threading.Thread(target=_run_training, daemon=True)
            thread.start()
        elif training_state["status"] == "completed":
            training_state.update(
                {
                    "status": "completed",
                    "message": "Training complete.",
                    "evals_data": training_state["evals_data"],
                    "weights": training_state["weights"],
                    "bias": training_state["bias"]
                }
            )

    return templates.TemplateResponse(
        request=req,
        name="train.html",
        context={
            "status": training_state["status"],
            "message": training_state["message"],
            "evals_data": training_state["evals_data"],
            
        },
    )


@app.get("/train-status")
def train_status():
    return {
        "status": training_state["status"],
        "message": training_state["message"],
        "evals_data": training_state["evals_data"],
    }


@app.post("/predict")
def predict(payload: dict = Body(...)):
    payload={
  "Apparent Temperature (C)": 7.5,
  "Humidity": 0.85,
  "Wind Speed (km/h)": 12.0,
  "Wind Bearing (degrees)": 250,
  "Visibility (km)": 15,
  "Pressure (millibars)": 1015,
  "year": 2006,
  "month": 3,
  "date": 31
}
    return predict_(payload)

    