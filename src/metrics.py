import numpy as np

# -------- MSE --------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# -------- RMSE --------
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


# -------- MAE --------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def print_metrics(y_true, y_pred, title=""):
    mse_val = mse(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    
    print(f"{title} - Metrics:")
    print(f"  MSE: {mse_val:.2f}")
    print(f"  MAE: {mae_val:.2f}")
    print(f"  RMSE: {rmse_val:.2f}")