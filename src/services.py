from fastapi import HTTPException
import pandas as pd

from src.pipeline import LinearRegressionScratch, read_file, split_data
from src.logger import logging
from src.main import training_status
from src.state_store import save_training_state


# training_state = {
#     "status": "idle",
#     "message": "Model not started yet.",
#     "evals_data": None,
#     "weights": None,
#     "bias": None,
#     "model": None,
# }


def _run_training():
    try:
        training_status.status = "training"
        training_status.message = "Model is training..."
        training_status.evals_data = None
        training_status.weights = None
        training_status.bias = None
        training_status.model = None
        # training_state.update(
        #     {
        #         "status": "training",
        #         "message": "Model is training...",
        #         "evals_data": None,
        #         "weights": None,
        #         "bias": None,
        #         "model": None,
        #     }
        # )
        df = read_file()
        x_train, y_train, x_test, y_test = split_data(df)
        model = LinearRegressionScratch(x_train, y_train, x_test, y_test)
        weights, bias = model.train()
       
        eval_data = model.evaluation(weights, bias)
        training_status.status = "completed"
        training_status.message = "Training complete."
        training_status.evals_data = eval_data
        training_status.weights = weights
        training_status.bias = bias
        training_status.model = model
        save_training_state(training_status)
        # training_state.update(
        #     {
        #         "status": "completed",
        #         "message": "Training complete.",
        #         "evals_data": eval_data,
        #         "weights": weights,
        #         "bias": bias,
        #         "model": model,
        #     }
        # )
        logging.info("Training completed")
    except Exception as e:
        logging.exception("Training failed")
        training_status.status = "failed"
        training_status.message = f"Training failed: {e}"
        training_status.evals_data = None
        training_status.weights = None
        training_status.bias = None
        training_status.model = None


def _normalize_payload(payload):
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def predict_(payload):
    model = training_status.model
    weights = training_status.weights
    bias = training_status.bias

    if model is None or weights is None or bias is None:
        logging.warning("Model has not been trained yet.")
        raise HTTPException(status_code=400, detail="Model has not been trained yet.")

    # payload_dict = _normalize_payload(payload)
    df = read_file()
    feature_columns = [column for column in df.columns if column != "Temperature (C)"]
    column_map = {
        "apparent_temperature": "Apparent Temperature (C)",
        "humidity": "Humidity",
        "wind_speed": "Wind Speed (km/h)",
        "wind_bearing": "Wind Bearing (degrees)",
        "visibility": "Visibility (km)",
        "pressure": "Pressure (millibars)",
        "loud_cover": "Loud Cover",
        "year": "year",
        "month": "month",
        "date": "date",
    }
    
    # FIX: Apply the map so the columns match your training data layout
    payload = payload.rename(columns=column_map)
    
    prediction = model.predict(weights, bias, payload)
    return {"prediction": float(prediction)}
