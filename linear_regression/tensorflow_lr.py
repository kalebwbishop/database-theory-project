import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
X = pd.read_csv("covid_data_features.csv")
y = pd.read_csv("covid_data_target.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and compile the model
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[X_train.shape[1]])])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Mean Squared Error:", loss)
