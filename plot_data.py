import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import import_dataset

# Load dataset and scaler
scaler, train_df, val_df, test_df = import_dataset()

# Load the trained LSTM model
model = load_model('lstm_model.keras')

# Define parameters
timesteps = 10

# Create the test sequence generator
test_generator = TimeseriesGenerator(
    test_df.drop(columns=['time[s]']).values,
    test_df['serving_cell_rssi_1'].values,
    length=timesteps, batch_size=32
)

# Make predictions
predictions = model.predict(test_generator)
predicted_rssi = predictions.flatten()

# Get the true RSSI values
true_rssi = test_df['serving_cell_rssi_1'].values[timesteps:]

# Descale the predicted and true values
predicted_rssi_descaled = scaler.inverse_transform(
    np.hstack((predicted_rssi.reshape(-1, 1), np.zeros((predicted_rssi.shape[0], train_df.shape[1] - 2)))))[:, 0]

true_rssi_descaled = scaler.inverse_transform(
    np.hstack((true_rssi.reshape(-1, 1), np.zeros((true_rssi.shape[0], train_df.shape[1] - 2)))))[:, 0]

# Get corresponding timestamps and positions
timestamps = test_df['time[s]'].values[timesteps:]
positions_x = test_df['position_x'].values[timesteps:]
positions_y = test_df['position_y'].values[timesteps:]

# Create results_df
results_df = pd.DataFrame({
    'time[s]': timestamps,
    'serving_cell_rssi_1': true_rssi_descaled,
    'predicted_rssi': predicted_rssi_descaled,
    'position_x': positions_x,
    'position_y': positions_y
})

# Calculate error metrics
mae = mean_absolute_error(true_rssi_descaled, predicted_rssi_descaled)
mse = mean_squared_error(true_rssi_descaled, predicted_rssi_descaled)
rmse = np.sqrt(mse)
r2 = r2_score(true_rssi_descaled, predicted_rssi_descaled)

# Print error metrics
print("\nPerformance Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Visualization

# RSSI Over Time
plt.figure(figsize=(12, 6))
plt.plot(results_df['time[s]'], results_df['serving_cell_rssi_1'], label="Actual RSSI", color='blue', alpha=0.7)
plt.plot(results_df['time[s]'], results_df['predicted_rssi'], label="Predicted RSSI", color='orange', alpha=0.7)
plt.title("Actual vs Predicted RSSI Over Time")
plt.xlabel("Time [s]")
plt.ylabel("RSSI")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Scatter Plot of Actual vs. Predicted RSSI
plt.figure(figsize=(8, 8))
sns.scatterplot(x=results_df['serving_cell_rssi_1'], y=results_df['predicted_rssi'], alpha=0.6)
plt.plot(
    [results_df['serving_cell_rssi_1'].min(), results_df['serving_cell_rssi_1'].max()],
    [results_df['serving_cell_rssi_1'].min(), results_df['serving_cell_rssi_1'].max()],
    color='red', linestyle='--'
)
plt.title("Actual vs Predicted RSSI")
plt.xlabel("Actual RSSI")
plt.ylabel("Predicted RSSI")
plt.grid()
plt.tight_layout()
plt.show()

# Error Distribution (Histogram)
errors = results_df['serving_cell_rssi_1'] - results_df['predicted_rssi']
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True, color='purple', alpha=0.7)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (Actual RSSI - Predicted RSSI)")
plt.ylabel("Frequency")
plt.grid()
plt.tight_layout()
plt.show()

# Spatial RSSI Prediction Plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(results_df['position_x'], results_df['position_y'], c=results_df['predicted_rssi'], cmap='viridis', s=50, alpha=0.8)
plt.colorbar(sc, label="Predicted RSSI")
plt.title("Predicted RSSI in Spatial Context")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.grid()
plt.tight_layout()
plt.show()
