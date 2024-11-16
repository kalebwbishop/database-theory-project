from system_logging import system_logging
import logging
import numpy as np
import threading
from systemds.context import SystemDSContext
from systemds.operator.algorithm import lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_systemds_lr():
    with SystemDSContext() as sds:
        starttime = time.time()
        try:
            # Load data into numpy arrays
            X = np.genfromtxt("laptop_features.csv", delimiter=",", skip_header=1)
            y = np.genfromtxt("laptop_target.csv", delimiter=",", skip_header=1)

            # Perform the train-test split (80/20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Convert to SystemDS matrices
            X_train_sds = sds.from_numpy(X_train)
            y_train_sds = sds.from_numpy(y_train.reshape(-1, 1))
            X_test_sds = sds.from_numpy(X_test)
            # y_test_sds = sds.from_numpy(y_test.reshape(-1, 1))

            # Train the linear regression model
            model = lm(X_train_sds, y_train_sds)

            # Make predictions on the test set
            predictions = (X_test_sds @ model).compute()

            # Calculate metrics
            errors = predictions - y_test
            mse = np.mean(errors ** 2)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            system_logging("systemds", f"Mean Squared Error (MSE): {mse:.4f}")
            system_logging("systemds", f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            system_logging("systemds", f"[Mean Absolute Error (MAE): {mae:.4f}")
            system_logging("systemds", f"R² (Coefficient of Determination): {r2:.4f}")

        except Exception as e:
            system_logging("systemds", f"Error: {str(e)}")
        
        finally:
            exec_time = time.time() - starttime
            system_logging("systemds", f"Execution Time: {exec_time:.2f} seconds")


if __name__ == "__main__":
    logging_thread = threading.Thread(target=system_logging, args=("systemds",), daemon=True)
    logging_thread.start()

    run_systemds_lr()
