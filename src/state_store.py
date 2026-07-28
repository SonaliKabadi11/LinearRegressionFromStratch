import pickle
from pathlib import Path


STATE_FILE = Path(__file__).resolve().parent.parent / "model_state.pkl"


def save_training_state(training_status):
    state = {
        "status": training_status.status,
        "message": training_status.message,
        "evals_data": training_status.evals_data,
        "weights": training_status.weights,
        "bias": training_status.bias,
        "model": training_status.model,
    }
    with STATE_FILE.open("wb") as handle:
        pickle.dump(state, handle)


def load_training_state(training_status):
    if not STATE_FILE.exists():
        return False

    try:
        with STATE_FILE.open("rb") as handle:
            state = pickle.load(handle)

        training_status.status = state.get("status", "idle")
        training_status.message = state.get("message", "Model not started yet")
        training_status.evals_data = state.get("evals_data")
        training_status.weights = state.get("weights")
        training_status.bias = state.get("bias")
        training_status.model = state.get("model")
        return True
    except Exception as exc:
        print(f"Unable to load training state: {exc}")
        return False
