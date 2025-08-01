import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import re

# Load the dataset
df = pd.read_excel('basic.xlsx')

# Extract performance metrics and target variables (temperatures of each core)
metrics = ['Core 1 Performance', 'Core 2 Performance', 'Core 3 Performance', 'Core 4 Performance', 'Core 5 Performance', 'Core 6 Performance']
cores = ['Core 1 Temperature', 'Core 2 Temperature', 'Core 3 Temperature', 'Core 4 Temperature', 'Core 5 Temperature', 'Core 6 Temperature']

# Preprocess the dataset to remove non-numeric characters
print("Before cleaning:", df[metrics].iloc[0])
df[metrics] = df[metrics].replace(r'[^\d.-]', '', regex=True)
df[cores] = df[cores].replace(r'[^\d.-]', '', regex=True)      # Raw string
print("After cleaning:", df[metrics].iloc[0])

# Convert the values to float
df[metrics] = df[metrics].astype(float)
df[cores] = df[cores].astype(float)

X = df[metrics].values
y = df[cores].values

# Normalize the feature data and target variables
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape the input data to fit LSTM input shape (samples, timesteps, features)
timesteps = 1
X_train = np.reshape(X_train, (X_train.shape[0], timesteps, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], timesteps, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=256, input_shape=(timesteps, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.3))
model.add(Dense(units=len(cores)))  # Output layer with neurons equal to the number of cores
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Save the trained model weights
model.save_weights('trained_model_weights.h5')

# Make predictions on the test set
y_pred_scaled = model.predict(X_test)

# Inverse scaling to get the actual temperature values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

# Calculate the accuracy using mean squared error (MSE)
mse = mean_squared_error(y_actual, y_pred)
accuracy = 100 - mse  # Higher accuracy corresponds to lower MSE
print(f"Accuracy: {accuracy:.2f}%")
