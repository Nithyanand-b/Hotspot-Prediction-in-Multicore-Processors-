import pandas as pd
import time
import wmi
from openpyxl import Workbook
import os

# Function to retrieve CPU metrics, core temperatures, and core performance metrics
def get_cpu_metrics():
    # c = wmi.WMI(namespace="root/HWiNFO64")
    c = wmi.WMI(namespace="root/OpenHardwareMonitor")
    core_temperatures = []
    core_performance = []

    try:
        # Retrieve temperature and performance information
        for sensor in c.Sensor():
            if sensor.SensorType == 'Temperature' and sensor.Name.startswith('CPU Core'):
                core_temperatures.append(f"{sensor.Value} °C")  # Append unit to temperature value
            elif sensor.SensorType == 'Load' and sensor.Name.startswith('CPU Core'):
                core_performance.append(f"{sensor.Value:.2f} %")  # Append unit and format performance value
    
    except Exception as e:
        print(f"Error accessing sensors: {e}")

    return core_temperatures, core_performance


# Create an empty list to store the data
data_list = []

# Duration in seconds (10 minutes = 600 seconds)
duration = 10

# Capture data for the specified duration
start_time = time.time()
while (time.time() - start_time) < duration:
    # Get core temperatures and performance metrics
    core_temperatures, core_performance = get_cpu_metrics()

    # Create a timestamp
    timestamp = pd.Timestamp.now()

    # Create a dictionary with the data
    data = {
        'Timestamp': timestamp
    }

    # Add core temperatures to the dictionary
    for i, temperature in enumerate(core_temperatures):
        core_name = f'Core {i + 1} Temperature'
        data[core_name] = temperature

    # Add core performance metrics to the dictionary
    for i, performance in enumerate(core_performance):
        core_name = f'Core {i + 1} Performance'
        data[core_name] = performance

    # Append the data to the list
    data_list.append(data)

    # Wait for 1 second before capturing the next data point
    time.sleep(1)

# Create the dataframe from the list of dictionaries
df = pd.DataFrame(data_list)

# Save the dataframe to an Excel file
df.to_excel('basic.xlsx', index=False)

# Print the current working directory
print(f"Data saved to: {os.getcwd()}/basic.xlsx")