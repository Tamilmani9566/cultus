# ============================================
# Advanced Hierarchical Time Series Forecasting
# Deep Learning (Hierarchical DeepAR-style)
# ============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import random

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Metrics
# ----------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def mase(y_true, y_pred, y_insample, m=1):
    naive_forecast = y_insample[m:]
    naive_actual = y_insample[:-m]
    scale = np.mean(np.abs(naive_forecast - naive_actual)) + 1e-8
    return np.mean(np.abs(y_true - y_pred)) / scale

# ----------------------------
# Synthetic Hierarchical Data
# ----------------------------
def generate_hierarchical_data(
    n_regions=3,
    n_products=4,
    timesteps=200
):
    data = []
    hierarchy = {}

    for r in range(n_regions):
        region_key = f"Region_{r}"
        hierarchy[region_key] = []
        region_trend = np.linspace(10, 50, timesteps)

        for p in range(n_products):
            product_key = f"{region_key}_Product_{p}"
            hierarchy[region_key].append(product_key)

            seasonal = 10 * np.sin(np.linspace(0, 20, timesteps))
            noise = np.random.normal(0, 3, timesteps)

            series = region_trend + seasonal + noise + np.random.uniform(0, 10)

            for t in range(timesteps):
                data.append([region_key, product_key, t, series[t]])

    df = pd.DataFrame(
        data, columns=["region", "product", "time", "value"]
    )
    return df, hierarchy

df, hierarchy = generate_hierarchical_data()

# ----------------------------
# Dataset
# ----------------------------
class HierarchicalDataset(Dataset):
    def __init__(self, df, input_window=24, forecast_horizon=12):
        self.series = []
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        for key, g in df.groupby("product"):
            values = g.sort_values("time")["value"].values
            for i in range(len(values) - input_window - forecast_horizon):
                x = values[i:i+input_window]
                y = values[i+input_window:i+input_window+forecast_horizon]
                self.series.append((x, y))

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x, y = self.series[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

# ----------------------------
# Hierarchical DeepAR Model
# ----------------------------
class HierarchicalDeepAR(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, horizon=12):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, loader, df):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(DEVICE))
            y_true.extend(y.numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    insample = df["value"].values

    return {
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, insample)
    }

# ----------------------------
# Local Baseline (Naive)
# ----------------------------
def naive_forecast(series, horizon):
    return np.repeat(series[-1], horizon)

def evaluate_naive(df, horizon=12):
    y_true, y_pred = [], []

    for _, g in df.groupby("product"):
        values = g.sort_values("time")["value"].values
        for i in range(len(values) - horizon - 1):
            y_true.extend(values[i+1:i+1+horizon])
            y_pred.extend(naive_forecast(values[:i+1], horizon))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, df["value"].values)
    }

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    INPUT_WINDOW = 24
    HORIZON = 12
    BATCH_SIZE = 64
    EPOCHS = 15

    dataset = HierarchicalDataset(df, INPUT_WINDOW, HORIZON)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HierarchicalDeepAR(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        horizon=HORIZON
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\nTraining Hierarchical Global Model\n")
    train_model(model, loader, optimizer, criterion, EPOCHS)

    print("\nEvaluating Global Model\n")
    global_metrics = evaluate_model(model, loader, df)
    print(global_metrics)

    print("\nEvaluating Local Naive Baseline\n")
    naive_metrics = evaluate_naive(df)
    print(naive_metrics)

    print("\nImprovement (Global vs Local)")
    for k in global_metrics:
        print(f"{k}: {naive_metrics[k] - global_metrics[k]:.4f}")
