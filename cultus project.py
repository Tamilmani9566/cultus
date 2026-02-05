# ============================================================
# HIERARCHICAL TIME SERIES FORECASTING WITH DEEP LEARNING
# SINGLE FILE – FULL END-TO-END IMPLEMENTATION
# ============================================================

# ------------------------
# IMPORTS
# ------------------------
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)
from pytorch_forecasting.metrics import QuantileLoss

# ------------------------
# CONFIGURATION
# ------------------------
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 15
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 32
ATTENTION_HEADS = 4
DROPOUT = 0.1
ENCODER_LENGTH = 24
PREDICTION_LENGTH = 12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------
# METRICS
# ------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def mase(y_true, y_pred, naive_forecast):
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - naive_forecast))
    return mae_model / (mae_naive + 1e-8)

# ------------------------
# BASELINE
# ------------------------
def naive_forecast(series, horizon):
    return np.repeat(series[-1], horizon)

# ------------------------
# DATA GENERATION (HIERARCHICAL)
# ------------------------
def generate_data():
    regions = ["North", "South"]
    products = ["A", "B", "C"]

    dates = pd.date_range("2018-01-01", periods=120, freq="M")
    rows = []

    for region in regions:
        for product in products:
            base = np.random.uniform(50, 200)
            trend = np.linspace(0, 30, len(dates))
            noise = np.random.normal(0, 10, len(dates))
            values = np.maximum(base + trend + noise, 0)

            for t, v in zip(dates, values):
                rows.append({
                    "time": t,
                    "region": region,
                    "product": product,
                    "series_id": f"{region}_{product}",
                    "value": v
                })

    df = pd.DataFrame(rows)
    return df

# ------------------------
# DATASET PREPARATION
# ------------------------
def build_datasets(df):
    df["time_idx"] = df.groupby("series_id").cumcount()

    cutoff = df["time_idx"].max() - PREDICTION_LENGTH

    training = TimeSeriesDataSet(
        df[df.time_idx <= cutoff],
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        static_categoricals=["region", "product"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True
    )

    return training, validation

# ------------------------
# MODEL
# ------------------------
def build_model(training_dataset):
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4
    )
    return model

# ------------------------
# TRAINING
# ------------------------
def train(model, training, validation):
    train_loader = DataLoader(
        training,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        validation,
        batch_size=BATCH_SIZE
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        gradient_clip_val=0.1,
        enable_checkpointing=False,
        logger=False
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    return model

# ------------------------
# EVALUATION
# ------------------------
def evaluate(model, validation, df):
    preds = model.predict(validation).detach().cpu().numpy()

    smape_scores = []
    mase_scores = []

    series_ids = df.series_id.unique()

    for i, sid in enumerate(series_ids):
        true_series = df[df.series_id == sid].value.values
        true = true_series[-PREDICTION_LENGTH:]
        pred = preds[i][:PREDICTION_LENGTH]
        naive = naive_forecast(true_series[:-1], PREDICTION_LENGTH)

        smape_scores.append(smape(true, pred))
        mase_scores.append(mase(true, pred, naive))

    return {
        "sMAPE": float(np.mean(smape_scores)),
        "MASE": float(np.mean(mase_scores))
    }

# ------------------------
# MAIN
# ------------------------
def main():
    print("\nGenerating hierarchical dataset...")
    df = generate_data()

    print("Building datasets...")
    training, validation = build_datasets(df)

    print("Building model...")
    model = build_model(training)

    print("Training model...")
    model = train(model, training, validation)

    print("Evaluating model...")
    metrics = evaluate(model, validation, df)

    print("\nFINAL RESULTS")
    print("======================")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

# ------------------------
# ENTRY POINT
# ------------------------
if __name__ == "__main__":
    main()
