import pandas as pd
import numpy as np
from src.models import LinearRegression
from src.metrics import mse
from src.preprocessing import normalize_train, normalize_test

def split_data(data):
    """Split data into 80/20 train/validation sets, stratified by city (Buenos Aires vs. New York)."""
    ba = data[data["lat"] < 0]
    ny = data[data["lat"] > 0]
    ba = ba.sample(frac=1, random_state=42)
    ny = ny.sample(frac=1, random_state=42)

    splitBA = int(0.8 * len(ba))
    splitNY = int(0.8 * len(ny))

    trainBA = ba[:splitBA]
    valBA = ba[splitBA:]
    trainNY = ny[:splitNY]
    valNY = ny[splitNY:]

    train = pd.concat([trainBA, trainNY])
    val = pd.concat([valBA, valNY])
    return train, val

def cross_val(data, features, target="precio", k=5, L1=0, L2=0):
    """K-fold cross-validation with per-fold normalization; returns mean and std of MSE across folds."""
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(data) // k
    scores = []

    for i in range(k):
        # Split
        val_idx   = list(range(i * fold_size, (i + 1) * fold_size))
        train_idx = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, len(data)))
        
        train_fold = data.iloc[train_idx].copy()
        val_fold   = data.iloc[val_idx].copy()

        # Normalize within fold — stats computed only from train_fold to avoid leakage
        train_fold_norm, fold_stats = normalize_train(train_fold, features + [target])
        val_fold_norm = normalize_test(val_fold, features + [target], fold_stats)

        X_train = train_fold_norm[features]
        y_train = train_fold_norm[target]
        X_val = val_fold_norm[features]
        y_val = val_fold_norm[target]

        # Train
        model = LinearRegression(X_train, y_train, L1, L2)
        if L1 > 0:
            model.gradient_descent()
        else:
            model.pseudo_inverse()

        # Evaluate in normalized space (no need to denormalize for comparison)
        y_val_pred = model.predict(X_val)
        scores.append(mse(y_val.values, y_val_pred))

    return np.mean(scores), np.std(scores)