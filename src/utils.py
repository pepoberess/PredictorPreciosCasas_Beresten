import matplotlib.pyplot as plt
import numpy as np

def plot_real_vs_pred(y_real_train, y_pred_train, y_real_val, y_pred_val, title=""):
    """Scatter real vs. predicted prices for both train and validation splits, with a diagonal reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_real, y_pred, split in zip(axes, 
                                          [y_real_train, y_real_val], 
                                          [y_pred_train, y_pred_val], 
                                          ["Train", "Validation"]):
        ax.scatter(y_real, y_pred, alpha=0.4, s=10)
        min_val = min(y_real.min(), y_pred.min())
        max_val = max(y_real.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red")
        ax.set_title(f"{title} — {split}")
        ax.set_xlabel("real price")
        ax.set_ylabel("predicted price")
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def learning_curve(model_class, X, y, X_val, y_val, reg_type, regularization, steps=10):
    """Train models on progressively larger subsets of data and return train/val MSE curves."""
    n = len(X)
    sizes = np.linspace(50, n, steps).astype(int)

    train_errors = []
    val_errors = []

    for size in sizes:
        X_sub = X.iloc[:size]
        y_sub = y.iloc[:size]

        model = model_class(X_sub, y_sub)
        if reg_type == "Ridge":
            model.L2 = regularization
            model.pseudo_inverse()
        elif reg_type == "Lasso":
            model.L1 = regularization
            model.gradient_descent()

        # TRAIN
        y_train_pred = model.predict(X_sub)
        train_mse = ((y_sub - y_train_pred) ** 2).mean()

        # VAL
        y_val_pred = model.predict(X_val)
        val_mse = ((y_val - y_val_pred) ** 2).mean()

        train_errors.append(train_mse)
        val_errors.append(val_mse)

    return sizes, train_errors, val_errors


def build_features(data, model_name):
    """Engineer polynomial, interaction, ratio, and log features for the given model variant (M4 or M5)."""
    data = data.copy()
    
    if model_name == "M4":
        data["ambientes_por_piso"] = data["ambientes"] / (data["pisos"])
        data["area_por_piso"] = data["Área"] / (data["pisos"])
        data["area_por_ambiente"] = data["Área"] / (data["ambientes"])
        data["ratio_cubierto"] = data["metros_cubiertos"] / (data["Área"])
        data["edad_cuadarada"] = data["edad"] ** 2
        data["Área^2"] = data["Área"] ** 2
        data["Área^3"] = data["Área"] ** 3
        data["metros_cubiertos^2"] = data["metros_cubiertos"] ** 2

    elif model_name == "M5":
        data["area^2"] = data["Área"] ** 2
        data["cubierto^2"] = data["metros_cubiertos"] ** 2
        data["edad^2"] = data["edad"] ** 2
        data["ambientes^2"] = data["ambientes"] ** 2
        data["pisos^2"] = data["pisos"] ** 2
        data["area^3"] = data["Área"] ** 3
        data["edad^3"] = data["edad"] ** 3
        data["area_x_cubierto"] = data["Área"] * data["metros_cubiertos"]
        data["area_x_ambientes"] = data["Área"] * data["ambientes"]
        data["area_x_edad"] = data["Área"] * data["edad"]
        data["cubierto_x_ambientes"] = data["metros_cubiertos"] * data["ambientes"]
        data["cubierto_x_edad"] = data["metros_cubiertos"] * data["edad"]
        data["ambientes_x_edad"] = data["ambientes"] * data["edad"]
        data["area_x_pisos"] = data["Área"] * data["pisos"]
        data["cubierto_x_pisos"] = data["metros_cubiertos"] * data["pisos"]
        data["area2_x_cubierto"] = (data["Área"] ** 2) * data["metros_cubiertos"]
        data["cubierto2_x_area"] = (data["metros_cubiertos"] ** 2) * data["Área"]
        data["area2_x_edad"] = (data["Área"] ** 2) * data["edad"]
        data["cubierto2_x_edad"] = (data["metros_cubiertos"] ** 2) * data["edad"]
        data["area_por_ambiente"] = data["Área"] / (data["ambientes"] + 1e-5)
        data["area_por_piso"] = data["Área"] / (data["pisos"] + 1e-5)
        data["cubierto_por_area"] = data["metros_cubiertos"] / (data["Área"] + 1e-5)
        data["densidad"] = data["ambientes"] / (data["Área"] + 1e-5)
        data["log_area"] = np.log(data["Área"] + 1)
        data["log_cubierto"] = np.log(data["metros_cubiertos"] + 1)
        data["sqrt_area"] = np.sqrt(data["Área"] + 1e-5)
        data["sqrt_cubierto"] = np.sqrt(data["metros_cubiertos"] + 1e-5)
        data["inv_area"] = 1 / (data["Área"] + 1e-5)
        data["inv_cubierto"] = 1 / (data["metros_cubiertos"] + 1e-5)
        data["sqrt_edad"] = np.sqrt(data["edad"] + 1e-5)
        data["lat^2"] = data["lat"] ** 2
        data["lon^2"] = data["lon"] ** 2
        data["lat_x_lon"] = data["lat"] * data["lon"]
        data["area_x_cubierto_x_ambientes"] = data["Área"] * data["metros_cubiertos"] * data["ambientes"]
        data["sqrt_area_por_ambiente"] = np.sqrt(
            data["Área"] / (data["ambientes"] + 1e-5)
        )
        data["sqrt_cubierto_por_area"] = np.sqrt(
            data["metros_cubiertos"] / (data["Área"] + 1e-5)
        )
        data["sqrt_area_x_cubierto"] = np.sqrt(
            data["Área"] * data["metros_cubiertos"] + 1e-5
        )
        data["sqrt_area_x_ambientes"] = np.sqrt(
            data["Área"] * data["ambientes"] + 1e-5
        )
    return data