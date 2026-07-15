import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import model_train


def test_model_train_returns_scalar_error():
    x_train = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    y_train = pd.Series([1, 2, 3])
    x_test = pd.DataFrame({"a": [4], "b": [8]})
    y_test = pd.Series([4])

    result = model_train(x_train, y_train, x_test, y_test)

    assert isinstance(result, float)
    assert result == result
