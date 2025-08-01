import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import wmi

# Function to retrieve current CPU performance metrics
def get_cpu_performance():
    c = wmi.WMI(namespace="root/OpenHardwareMonitor")

    core_performance = []

    try:
        # Retrieve performance information
        for sensor in c.Sensor():
            if sensor.SensorType == 'Load' and sensor.Name.startswith('CPU Core'):
                core_performance.append(sensor.Value)

    except Exception as e:
        print(f"Error accessing performance sensors: {e}")

    return core_performance

# Load the dataset
df = pd.read_excel('basic.xlsx')

# Extract performance metrics (input features)
metrics = ['Core 1 Performance', 'Core 2 Performance', 'Core 3 Performance', 'Core 4 Performance', 'Core 5 Performance', 'Core 6 Performance']
cores = ['Core 1 Temperature', 'Core 2 Temperature', 'Core 3 Temperature', 'Core 4 Temperature', 'Core 5 Temperature', 'Core 6 Temperature']

# Preprocess the dataset to remove non-numeric characters
df[metrics] = df[metrics].replace('[^\d.-]', '', regex=True)
df[cores] = df[cores].replace('[^\d.-]', '', regex=True)

# Convert the values to float
df[metrics] = df[metrics].astype(float)
df[cores] = df[cores].astype(float)

X = df[metrics].values

# Normalize the feature data
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

# Calculate the number of timesteps for 1 minute prediction
time_interval = pd.to_timedelta(1, unit='hour')

# Set the frequency of the index
df.index = pd.date_range(start=df.index[0], periods=len(df), freq=time_interval)

# Calculate the number of timesteps
timesteps = int((time_interval / pd.Timedelta(seconds=1)) / (df.index[1] - df.index[0]).total_seconds())

# Reshape the input data to fit LSTM input shape (samples, timesteps, features)
X_input = np.reshape(X_scaled[-timesteps:], (1, timesteps, X_scaled.shape[1]))

# Determine the number of features (neurons) in the output layer
num_output_neurons = X_scaled.shape[1]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=256, input_shape=(timesteps, X_input.shape[2]), return_sequences=True))
model.add(LSTM(units=128))
model.add(Dense(units=num_output_neurons))  # Output layer with neurons equal to the number of features
model.compile(loss='mean_squared_error', optimizer='adam')

# Load the trained model weights
model.load_weights('trained_model_weights.h5', by_name=True)  # Add 'by_name=True' to handle mismatch

# Get current CPU performance metrics
core_performance = get_cpu_performance()

# Prepare the input data for prediction
input_data = np.array(core_performance).reshape(1, len(core_performance))

# Normalize the input data
input_data_scaled = scaler_X.transform(input_data[:, :X_scaled.shape[1]])

# Reshape the input data to fit LSTM input shape (samples, timesteps, features)
input_data_reshaped = np.reshape(input_data_scaled, (1, timesteps, input_data_scaled.shape[1]))

# Make predictions on the input data
y_pred_scaled = model.predict(input_data_reshaped)

# Inverse scaling to get the actual temperature values
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(df[cores].values)  # Fit the scaler on the entire temperature values

y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Print the predicted temperature values for each core
for i, core in enumerate(cores):
    print(f"Predicted temperature for {core}: {y_pred[0][i]}")

# Define the threshold temperature for hotspot
threshold_temperature = 50  # Adjust the value as per your requirement

# Check if any core's temperature exceeds the threshold to determine hotspot
hotspot_detected = False
hotspot_cores = []
for i, core in enumerate(cores):
    if y_pred[0][i] > threshold_temperature:
        hotspot_detected = True
        hotspot_cores.append(core)

# Print hotspot detection result
if hotspot_detected:
    print("Hotspot detected in the following core(s):")
    

    # Determine which core is hotter
    hottest_core_index = np.argmax(y_pred)
    hottest_core = cores[hottest_core_index]
    print(f"{hottest_core} will be the hotspot ")
else:
    print("No hotspot detected.")
