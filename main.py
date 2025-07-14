import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import StockLSTM
from data_loader import load_and_preprocess_data

# Ensure correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_and_preprocess_data('data/sp500.csv')
df = df.apply(pd.to_numeric, errors='coerce')
# Replace inf, -inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_model = df[['Close', 'SMA_10', 'SMA_50', 'Volume_change', 'RSI_14']].copy()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_model)
# ===== Prepare sequences =====
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # Close price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, 60)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


# Initialize the same model structure
model = StockLSTM(input_size=5).to(device)
model_path = os.path.join('outputs', 'saved_models', 'lstm_stock_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model loaded successfully.")


print("Model loaded successfully.")

# ===== Predict =====
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

# ===== Inverse scaling for plotting =====
dummy = np.zeros((len(y_pred), scaled_data.shape[1]))
dummy[:, 0] = y_pred.squeeze()
y_pred_rescaled = scaler.inverse_transform(dummy)[:, 0]

dummy[:, 0] = y_true.squeeze()
y_true_rescaled = scaler.inverse_transform(dummy)[:, 0]


print("Any NaNs in predictions?", np.isnan(y_pred_rescaled).any())
print("Where are NaNs in predictions?", np.where(np.isnan(y_pred_rescaled))[0])
# Ensure no NaNs or Infs in predictions
y_pred_rescaled = pd.Series(y_pred_rescaled).interpolate().bfill().ffill().values
y_true_rescaled = pd.Series(y_true_rescaled).interpolate().bfill().ffill().values

# --- Evaluation ---
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# --- Plotting ---
plt.figure(figsize=(14, 6))
plt.plot(y_true_rescaled, label='Actual Close')
plt.plot(y_pred_rescaled, label='Predicted Close')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Predicted vs Actual Close Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_true_rescaled[-200:], label='Actual Close')
plt.plot(y_pred_rescaled[-200:], label='Predicted Close')
plt.title('Last 200 Days: Predicted vs Actual')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/last200_trend.png")
plt.show()
