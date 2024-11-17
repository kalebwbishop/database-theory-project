import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time
import shutil

from system_logging import SystemLogging

def run_pytorch_lr(system_logging):
    starttime = time.time()
    endtime = 0

    try:
        # Load data
        X = pd.read_csv("laptop_features.csv").values  # Convert to NumPy array
        y = pd.read_csv("laptop_target.csv").values.flatten()  # Flatten to 1D array

        loop_start = time.time()
        while loop_start + 15 > time.time():
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            # Create DataLoader for training and testing
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Define the model
            model = nn.Linear(X_train.shape[1], 1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Training loop
            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)
                train_loss /= len(train_loader.dataset)

            # Evaluation
            model.eval()
            with torch.no_grad():
                y_pred_tensor = model(X_test_tensor)
                y_pred = y_pred_tensor.numpy().flatten()
                y_test = y_test_tensor.numpy().flatten()

            mse = criterion(torch.tensor(y_pred).view(-1, 1), torch.tensor(y_test).view(-1, 1)).item()
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            if endtime == 0:
                endtime = time.time()


        system_logging.write_output(f"Mean Squared Error (MSE): {mse}")
        system_logging.write_output(f"Mean Absolute Error (MAE): {mae}")
        system_logging.write_output(f"Root Mean Squared Error (RMSE): {rmse}")
        system_logging.write_output(f"R² (Coefficient of Determination): {r2}")

    except Exception as e:
        system_logging.write_output(f"Error: {str(e)}")
    finally:
        exec_time = endtime - starttime
        system_logging.write_output(f"Execution Time: {exec_time}")


if __name__ == "__main__":
    ml_system = "pytorch"

    pytorch_logging = SystemLogging(ml_system)
    pytorch_logging.start_logging()
    run_pytorch_lr(pytorch_logging)
    pytorch_logging.stop_logging()

    source = f'/tmp/local_logs/{ml_system}.log'
    destination = f'./local_logs/{ml_system}.log'

    shutil.copy(source, destination)
