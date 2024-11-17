import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import time
import shutil

from system_logging import SystemLogging

def run_tensorflow_lr(system_logging):
    starttime = time.time()
    endtime = 0


    try:
        # Load the data
        X = pd.read_csv("laptop_features.csv")
        y = pd.read_csv("laptop_target.csv").values.flatten() 

        loop_start = time.time()
        while loop_start + 15 > time.time():
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define and compile the model
            model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[X_train.shape[1]])])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        loss='mean_squared_error',
                        metrics=['mean_absolute_error'])

            # Train the model
            model.fit(X_train, y_train, epochs=100, verbose=0)

            # Evaluate the model on test data
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            rmse = tf.math.sqrt(loss).numpy()
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            if endtime == 0:
                endtime = time.time()

        system_logging.write_output(f"Mean Squared Error (MSE): {loss}")
        system_logging.write_output(f"Mean Absolute Error (MAE): {mae}")
        system_logging.write_output(f"Root Mean Squared Error (RMSE): {rmse}")
        system_logging.write_output(f"R² (Coefficient of Determination): {r2}")

    except Exception as e:
        system_logging.write_output(f"Error: {str(e)}")
    finally:
        exec_time = endtime - starttime
        system_logging.write_output(f"Execution Time: {exec_time}")


if __name__ == "__main__":
    ml_system = "tensorflow"

    tensorflow_logging = SystemLogging(ml_system)
    tensorflow_logging.start_logging()
    run_tensorflow_lr(tensorflow_logging)
    tensorflow_logging.stop_logging()

    source = f'/tmp/local_logs/{ml_system}.log'
    destination = f'./local_logs/{ml_system}.log'

    shutil.copy(source, destination)

