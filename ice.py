# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("regional_climate_data.csv")  # Replace with your file path

# Display the first few rows of the dataset
print(data.head())

# Features and target
features = ['CO2', 'CH4', 'N2O', 'temperature_anomaly']
target = 'ice_concentration'

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])

# Convert to a DataFrame for easy indexing
scaled_df = pd.DataFrame(scaled_data, columns=features + [target])


# Define a function to create sequences
def create_sequences(data, target_column, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])
        y.append(data[i + sequence_length, target_column])
    return np.array(X), np.array(y)

# Define sequence length (e.g., 12 months)
sequence_length = 12

# Create input sequences and corresponding targets
X, y = create_sequences(scaled_df.values, target_column=-1, sequence_length=sequence_length)

# Split into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)  # Output layer predicting ice concentration
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
model.summary()


# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Predict on the test set
y_pred = model.predict(X_test)

# Rescale predictions back to original scale
y_test_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), len(features))), y_test.reshape(-1, 1)], axis=1))[:, -1]
y_pred_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred), len(features))), y_pred], axis=1))[:, -1]

# Plot actual vs predicted ice concentration
plt.plot(y_test_rescaled, label='Actual Ice Concentration')
plt.plot(y_pred_rescaled, label='Predicted Ice Concentration')
plt.legend()
plt.title('Actual vs Predicted Ice Concentration')
plt.show()

# Predict future ice concentration
future_steps = 12
last_sequence = X_test[-1]  # Start with the last known sequence
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_sequence[np.newaxis, :, :])
    future_predictions.append(prediction[0, 0])
    # Update sequence with the new prediction
    last_sequence = np.append(last_sequence[1:], [[0, 0, 0, 0, prediction[0, 0]]], axis=0)

# Rescale future predictions back to the original scale
future_predictions_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((len(future_predictions), len(features))), np.array(future_predictions).reshape(-1, 1)], axis=1))[:, -1]

# Display future predictions
print("Future Predictions:", future_predictions_rescaled)
