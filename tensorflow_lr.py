import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import threading
import time
from system_logging import system_logging

def run_tensorflow_lr():
    starttime = time.time()

    try:
        # Load the data
        X = pd.read_csv("laptop_features.csv")
        y = pd.read_csv("laptop_target.csv")

        # Ensure y is in the correct shape for TensorFlow
        y = y.values.flatten()  # Convert to a 1D array if necessary

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define and compile the model
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[X_train.shape[1]])])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])

        # Train the model
        model.fit(X_train, y_train, epochs=100, verbose=1)

        # Evaluate the model on test data
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        rmse = tf.math.sqrt(loss).numpy()
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        system_logging("tensorflow", f"Mean Squared Error (MSE): {loss}")
        system_logging("tensorflow", f"Mean Absolute Error (MAE): {mae}")
        system_logging("tensorflow", f"Root Mean Squared Error (RMSE): {rmse}")
        system_logging("tensorflow", f"R² (Coefficient of Determination): {r2}")

    except Exception as e:
        system_logging("tensorflow", f"Error: {str(e)}")
    finally:
        exec_time = time.time() - starttime
        system_logging("tensorflow", f"Execution Time: {exec_time}")


if __name__ == "__main__":
    logging_thread = threading.Thread(target=system_logging, args=("tensorflow",), daemon=True)
    logging_thread.start()

    run_tensorflow_lr()
