import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


def read_file():
    """
    Reads the cleaned csv file and returns the comma separated data

    Returns:
    data : DataFrame object = cleaned data for the model
    """
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
       
        file_path = os.path.join(cwd, "data", "cleaned-data.csv")
        
        data = pd.read_csv(file_path)
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])
        logging.info("Data is read from the csv file successfully as a DataFrame")
        return data
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

def split_data(df):
    """
    Splits the dataset into train data and test data

    Parameters:
    df = Dataset for the model

    returns:
    x_train = dependent features train values (2 dimensional)
    y_train = independent feature train values (1 dimensional)
    x_test = dependent features test values (2 dimensional)
    y_test = independent feature test values (1 dimensional)
    """
    try:
        columns = [col for col in df.columns if col != "Temperature (C)"]
        X = df[columns]
        Y = df["Temperature (C)"]
        n = int(X.shape[0] * 0.7)
        x_train = X[:][:n]
        y_train = Y[:n]
        x_test = X[:][n:]
        y_test = Y[n:]
        logging.info(f"The dataset is split into train and test with 7:3 ratio respectively")
        print((f"The dataset is split into train and test with 7:3 ratio respectively"))
        print("X train", x_train.shape)
        print("Y train", y_train.shape)
        print("X test", x_test.shape)
        print("Y test", y_test.shape)

        return x_train, y_train, x_test, y_test
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

    
    

class LinearRegressionScratch :
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _fit_scaler(self, data):
        """Fit the feature scaler using the training data."""
        try:
            self.mean = data.mean()
            std = data.std()
            self.std = std.replace(0, 1)
            return (data - self.mean) / self.std
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
    

    def _transform_data(self, data):
        """Apply the fitted scaler to new data."""
        try:
            return (data - self.mean) / self.std
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


    def train(self,):
        """
        Trains the model using gradient descent

        Parameters:
        x_train = dependent features train values (2 dimensional)
        y_train = independent feature train values (1 dimensional)
        x_test = dependent features test values (2 dimensional)
        y_test = independent feature test values (1 dimensional)

        Returns:
        tuple: trained weights and bias
        """
        try:
            x_train_arr = self._fit_scaler(self.x_train).to_numpy(dtype=float)
            y_train_arr = self.y_train.to_numpy(dtype=float)
            learning_rate = 1e-4
            max_iterations = 1000000

            weights = np.zeros(x_train_arr.shape[1], dtype=float)
            bias = 0.0
            error_history = []
            prev_error = None

            for epoch in range(max_iterations):
                predictions = x_train_arr @ weights + bias
                residual = y_train_arr - predictions
                error = float(np.mean(residual ** 2))
                error_history.append(error)

                if epoch % 50 == 0 or epoch == max_iterations - 1:
                    logging.info(f"Epoch {epoch + 1}: Error {error:.6f}")
                    print(f"Epoch {epoch + 1}: Error {error:.6f}")

                if prev_error is not None and abs(prev_error - error) < 1e-8:
                    break

                prev_error = error

                gradient_weight = (-2 / x_train_arr.shape[0]) * (x_train_arr.T @ residual)
                gradient_bias = (-2 / x_train_arr.shape[0]) * np.sum(residual)

                weights -= learning_rate * gradient_weight
                bias -= learning_rate * gradient_bias

            error_history_array = np.array(error_history, dtype=float)
            plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, "error_history.png")
            plt.figure(figsize=(8, 4))
            plt.plot(error_history_array, color="steelblue")
            plt.xlabel("Iteration")
            plt.ylabel("Mean Squared Error")
            plt.title("Training Error History")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Plot saved to {plot_path}")
            self.train_predictions = predictions
            return weights, bias
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


    def predict(self, weights, bias, data):
        try:
            transformed_data = self._transform_data(data)
            return transformed_data.to_numpy(dtype=float) @ weights + bias
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def evaluation(self, weights, bias):
        try:
            train_predictions = self.predict(weights, bias, self.x_train)
            test_predictions = self.predict(weights, bias, self.x_test)

            y_train_true = self.y_train.to_numpy(dtype=float)
            y_test_true = self.y_test.to_numpy(dtype=float)

            mse_train = float(np.mean((train_predictions - y_train_true) ** 2))
            mae_train = float(np.mean(np.abs(train_predictions - y_train_true)))
            rmse_train = float(np.sqrt(mse_train))

            ss_res = np.sum((y_train_true - train_predictions) ** 2)
            ss_tot = np.sum((y_train_true - np.mean(y_train_true)) ** 2)
            r2_train = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

            mse_test = float(np.mean((test_predictions - y_test_true) ** 2))
            mae_test = float(np.mean(np.abs(test_predictions - y_test_true)))
            rmse_test = float(np.sqrt(mse_test))

            ss_res_test = np.sum((y_test_true - test_predictions) ** 2)
            ss_tot_test = np.sum((y_test_true - np.mean(y_test_true)) ** 2)
            r2_test = float(1 - (ss_res_test / ss_tot_test)) if ss_tot_test != 0 else 0.0

            logging.info("Training Metrics")
            logging.info(f"MSE:{ mse_train}")
            logging.info(f"MAE: {mae_train}")
            logging.info(f"RMSE: {rmse_train}")
            logging.info(f"R2: {r2_train}")
            print("Training Metrics")
            print("MSE:", mse_train)
            print("MAE:", mae_train)
            print("RMSE:", rmse_train)
            print("R2:", r2_train)
            print(f"That means the model explains about {r2_train}% of the variance on both train sets.")


            logging.info("Test Metrics")
            logging.info(f"MSE:{ mse_test}")
            logging.info(f"MAE: {mae_test}")
            logging.info(f"RMSE: {rmse_test}")
            logging.info(f"R2: {r2_test}")
            print("Test Metrics")
            print("MSE:", mse_test)
            print("MAE:", mae_test)
            print("RMSE:", rmse_test)
            print("R2:", r2_test)
            print(f"That means the model explains about {r2_test}% of the variance on both test sets.")
            plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
            os.makedirs(plot_dir, exist_ok=True)

            plt.figure(figsize=(6, 6))
            plt.scatter(y_train_true, train_predictions, alpha=0.3)
            plt.plot([y_train_true.min(), y_train_true.max()], [y_train_true.min(), y_train_true.max()], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted (Train)")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "actual_vs_predicted_train.png"))
            plt.close()
            logging.info(f"Saved the Actual vs Predicted (Train) plot")
            residuals = y_train_true - train_predictions
            plt.figure(figsize=(6, 4))
            plt.scatter(train_predictions, residuals, alpha=0.3)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predicted")
            plt.ylabel("Residual")
            plt.title("Residual Plot")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "residual_plot.png"))

            plt.close()
            logging.info(f"Saved the residual plot")
            return {
                "train-metrics" : {
                    "MSE" : mse_train,
                    "MAE" : mae_train,
                    "RMSE" : rmse_train,
                    "R2" : r2_train
                },
                "test-metrics": {
                    "MSE" : mse_test,
                    "MAE" : mae_test,
                    "RMSE" : rmse_test,
                    "R2" : r2_test
                }
            }
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    df = read_file()
    x_train, y_train, x_test, y_test = split_data(df)
    print(np.array(x_train[1:2][:]))
    model = LinearRegressionScratch(x_train, y_train, x_test, y_test)
    weights, bias = model.train()
    model.evaluation(weights, bias)


