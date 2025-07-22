import os
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import load_and_preprocess_data
from src.train import create_sequences
from src.model import StockLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = load_and_preprocess_data('data/sp500.csv')  # already includes features
features = df[['Close', 'SMA_10', 'SMA_50', 'Volume_change', 'RSI_14']].copy()

features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(inplace=True) 

print("Any infs:", np.isinf(features).sum())
print("Any NaNs:", features.isna().sum())

# Load the same scaler used in training
scaler = joblib.load('notebooks/scaler.save')
scaled_data = scaler.transform(features)


# Create sequences
X, y = create_sequences(scaled_data, 5)
split_idx = int(len(X) * 0.8)
X_test, y_test = X[split_idx:], y[split_idx:]

# Convert to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Load model
model = StockLSTM(input_size=5).to(device)
model.load_state_dict(torch.load('outputs/saved_models/lstm_stock_model.pth', map_location=device))
model.eval()
print("Model loaded successfully.")

# Predict
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).squeeze().cpu().numpy()
    y_true_scaled = y_test_tensor.squeeze().cpu().numpy()

# Reconstruct full feature array for inverse_transform
dummy_pred = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
dummy_pred[:, 0] = y_pred_scaled

dummy_true = np.zeros((len(y_true_scaled), scaled_data.shape[1]))
dummy_true[:, 0] = y_true_scaled

# Inverse transform only the Close column
y_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]
y_true_rescaled = scaler.inverse_transform(dummy_true)[:, 0]

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(y_true_rescaled, label='Actual Close')
plt.plot(y_pred_rescaled, label='Predicted Close')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Predicted vs Actual Close Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/actual_vs_predicted.png")
plt.show()

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")