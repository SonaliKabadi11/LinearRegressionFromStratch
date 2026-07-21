from fastapi import HTTPException
import pandas as pd

from src.pipeline import LinearRegressionScratch, read_file, split_data

training_state = {
    "status": "idle",
    "message": "Model not started yet.",
    "evals_data": None,
    "weights": None,
    "bias": None,
    "model": None,
}


def _run_training():
    global training_state
    try:
        training_state.update(
            {
                "status": "training",
                "message": "Model is training...",
                "evals_data": None,
                "weights": None,
                "bias": None,
                "model": None,
            }
        )
        df = read_file()
        x_train, y_train, x_test, y_test = split_data(df)
        model = LinearRegressionScratch(x_train, y_train, x_test, y_test)
        weights, bias = model.train()
        eval_data = model.evaluation(weights, bias)
        training_state.update(
            {
                "status": "completed",
                "message": "Training complete.",
                "evals_data": eval_data,
                "weights": weights,
                "bias": bias,
                "model": model,
            }
        )
    except Exception as e:
        training_state.update(
            {
                "status": "failed",
                "message": f"Training failed: {e}",
                "evals_data": None,
                "weights": None,
                "bias": None,
                "model": None,
            }
        )


def predict_(payload: dict):
    model = training_state.get("model")
    weights = training_state.get("weights")
    bias = training_state.get("bias")

    if model is None or weights is None or bias is None:
        raise HTTPException(status_code=400, detail="Model has not been trained yet.")

    df = read_file()
    feature_columns = [column for column in df.columns if column != "Temperature (C)"]

    missing_columns = [column for column in feature_columns if column not in payload]
    if missing_columns:
        default_values = {
            "Loud Cover": 0.0,
            "Apparent Temperature (C)": 0.0,
            "Humidity": 0.0,
            "Wind Speed (km/h)": 0.0,
            "Wind Bearing (degrees)": 0.0,
            "Visibility (km)": 0.0,
            "Pressure (millibars)": 0.0,
            "year": 2006,
            "month": 1,
            "date": 1,
        }
        for column in missing_columns:
            payload[column] = payload.get(column, default_values.get(column, 0))

    input_df = pd.DataFrame([payload], columns=feature_columns)
    prediction = model.predict(weights, bias, input_df)
    return {"prediction": float(prediction[0])}
