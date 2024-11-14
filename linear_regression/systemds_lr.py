import logging
import numpy as np
from systemds.context import SystemDSContext
from systemds.operator.algorithm import lm
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)

# Start SystemDS context
with SystemDSContext() as sds:
    # Load data into numpy arrays first
    X = np.genfromtxt("covid_features.csv", delimiter=",", skip_header=1)
    y = np.genfromtxt("covid_target.csv", delimiter=",", skip_header=1)

    # Perform the train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to SystemDS matrices
    X_train_sds = sds.from_numpy(X_train)
    y_train_sds = sds.from_numpy(y_train.reshape(-1, 1))  # Reshape for compatibility
    X_test_sds = sds.from_numpy(X_test)
    y_test_sds = sds.from_numpy(y_test.reshape(-1, 1))

    # Train the linear regression model
    model = lm(X_train_sds, y_train_sds)

    # Make predictions on the test set using matrix multiplication
    predictions = X_test_sds @ model

    # Calculate Mean Squared Error
    errors = predictions - y_test_sds
    mse = (errors * errors).mean().compute()  # Calculate MSE
    rmse = np.sqrt(mse)  # Calculate RMSE

    # Log the Mean Squared Error
    logging.info("Mean Squared Error: %s", mse)
    logging.info("Root Mean Squared Error: %s", rmse)

