# eda_and_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("Pallets data.csv")

# Parse dates safely
df["POSTING DATE"] = pd.to_datetime(df["POSTING DATE"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["POSTING DATE"])

# Clean QUANTITY column (remove commas, convert to float)
df["QUANTITY"] = df["QUANTITY"].astype(str).str.replace(",", "").astype(float)

# Aggregate sales per day
df = df.groupby("POSTING DATE")["QUANTITY"].sum().reset_index()
df = df.sort_values("POSTING DATE")

print("Sample parsed dates:", df["POSTING DATE"].head())

# -------------------------------
# Train/Test Split
# -------------------------------
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# -------------------------------
# Helper to evaluate models
# -------------------------------
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return rmse, mae

results = {}

# -------------------------------
# 1. Naive Forecast
# -------------------------------
pred_naive = np.repeat(train["QUANTITY"].iloc[-1], len(test))
rmse, mae = evaluate_model("Naive", test["QUANTITY"].values, pred_naive)
results["Naive"] = {"RMSE": rmse, "MAE": mae}

# -------------------------------
# 2. Moving Average
# -------------------------------
window = 7
rolling_mean = train["QUANTITY"].rolling(window).mean().iloc[-1]
pred_ma = np.repeat(rolling_mean, len(test))
rmse, mae = evaluate_model("Moving Average", test["QUANTITY"].values, pred_ma)
results["Moving Average"] = {"RMSE": rmse, "MAE": mae}

# -------------------------------
# 3. ARIMA
# -------------------------------
try:
    arima_model = ARIMA(train["QUANTITY"], order=(5, 1, 0))
    arima_fit = arima_model.fit()
    pred_arima = arima_fit.forecast(len(test))
    rmse, mae = evaluate_model("ARIMA", test["QUANTITY"].values, pred_arima)
    results["ARIMA"] = {"RMSE": rmse, "MAE": mae}
except Exception as e:
    print("ARIMA failed:", e)

# -------------------------------
# 4. Prophet
# -------------------------------
try:
    prophet_df = train.rename(columns={"POSTING DATE": "ds", "QUANTITY": "y"})
    prophet = Prophet()
    prophet.fit(prophet_df)

    # Forecast for test period
    future = prophet.make_future_dataframe(periods=len(test))
    forecast = prophet.predict(future)

    pred_prophet = forecast["yhat"].iloc[-len(test):].values
    rmse, mae = evaluate_model("Prophet", test["QUANTITY"].values, pred_prophet)
    results["Prophet"] = {"RMSE": rmse, "MAE": mae}
except Exception as e:
    print("Prophet failed:", e)

# -------------------------------
# Pick Best Model
# -------------------------------
if results:
    best_model = min(results, key=lambda x: results[x]["RMSE"])
    print("\nBest Model:", best_model, results[best_model])
else:
    print("\nNo models were successfully trained.")

# -------------------------------
# Plot Historical vs Predictions
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(train["POSTING DATE"], train["QUANTITY"], label="Train (Historical)")
plt.plot(test["POSTING DATE"], test["QUANTITY"], label="Test (Actual)")

if "ARIMA" in results:
    plt.plot(test["POSTING DATE"], pred_arima, label="ARIMA Predictions")
if "Prophet" in results:
    plt.plot(test["POSTING DATE"], pred_prophet, label="Prophet Predictions")
plt.plot(test["POSTING DATE"], pred_naive, label="Naive Predictions")
plt.plot(test["POSTING DATE"], pred_ma, label="Moving Avg Predictions")

plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.title("Historical Sales vs Predictions")
plt.legend()
plt.tight_layout()
plt.show()
