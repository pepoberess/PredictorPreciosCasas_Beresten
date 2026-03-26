
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(train, test):
    """Load train and test CSVs into DataFrames."""
    data_train = pd.read_csv(train)
    data_test = pd.read_csv(test)
    return data_train, data_test

def boxplots(data, f1, f2):
    """Plot side-by-side boxplots for two features."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(data[f1])
    axes[0].set_title(f1)
    axes[1].boxplot(data[f2])
    axes[1].set_title(f2)
    plt.tight_layout()
    plt.show()

def one_hot_encoder_tipo(data):
    """One-hot encode the 'tipo' column and drop the 'tipo_ph' dummy."""
    data = pd.get_dummies(data, columns=["tipo"])
    data = data.drop(columns=["tipo_ph"])
    return data

def adjust_floors(data):
    """Fill missing floor values: 2 for expensive houses, 1 for everything else."""
    mean_price = data["precio"].mean()
    missing = data["pisos"].isna()
    millionaire = (data["tipo"] == "casa") & (data["precio"] > mean_price)
    data.loc[millionaire & missing, "pisos"] = 2
    data.loc[missing & ~millionaire, "pisos"] = 1
    return data

def adjust_age(data):
    """Fill missing age values with the column mean."""
    avg = data["edad"].mean()
    missing = data["edad"].isna()
    data.loc[missing, "edad"] = avg
    return data

def first_changes(data):
    """Remove zero-price rows and impute missing floors and age."""
    data = data[data["precio"] > 0]
    data = adjust_floors(data)
    data = adjust_age(data)
    return data


def change_units(data):
    """Convert area columns from sqft to m² and drop the units column."""
    factor = 0.092903  # sqft → m2
    sqft = data["unidades"] == "sqft"
    data.loc[sqft, "Área"] *= factor
    data.loc[sqft, "metros_cubiertos"] *= factor
    data = data.drop(columns=["unidades"])
    return data

def plot_city(houses1, apts1, ph1, houses2, apts2, ph2, title1, title2):
    """Scatter plot of property locations (lat/lon) by type for two cities side by side."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(houses1["lon"], houses1["lat"], label="Houses", alpha=0.6)
    axes[0].scatter(apts1["lon"], apts1["lat"], label="Apartments", alpha=0.6)
    axes[0].scatter(ph1["lon"], ph1["lat"], label="PH", alpha=0.6)

    axes[1].scatter(houses2["lon"], houses2["lat"], label="Houses", alpha=0.6)
    axes[1].scatter(apts2["lon"], apts2["lat"], label="Apartments", alpha=0.6)
    axes[1].scatter(ph2["lon"], ph2["lat"], label="PH", alpha=0.6)

    axes[0].set_title(title1)
    axes[1].set_title(title2)
    axes[0].set_xlabel("Longitud")
    axes[1].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")
    axes[1].set_ylabel("Latitud")
    axes[0].legend()
    axes[1].legend()
    plt.show()

def adjust_low_prices(data, thresholdBA, factorBA=20):
    """Multiply suspiciously low Buenos Aires prices by a correction factor."""
    is_ba = data["lat"] < 0
    low_prices_ba = is_ba & (data["precio"] < thresholdBA)
    data.loc[low_prices_ba, "precio"] *= factorBA
    return data

def choose_thresholds(data, division):
    """Return the price threshold at the given quantile (in log space) for Buenos Aires."""
    ba = data[data["lat"] < 0]
    ba_log = np.log1p(ba["precio"])
    threshold_logBA = ba_log.quantile(division)
    thresholdBA = np.expm1(threshold_logBA)
    return thresholdBA


def normalize_train(data, features):
    """Z-score normalize the given features and return the per-feature (mean, std) statistics."""
    statistics = {}
    for feature in features:
        mean = data[feature].mean()
        std = data[feature].std()
        data[feature] = (data[feature] - mean) / std
        statistics[feature] = (mean, std)
    return data, statistics

def normalize_test(data, features, statistics):
    """Apply train-set statistics to normalize test/validation features."""
    data = data.copy()
    for feature in features:
        mean, std = statistics[feature]
        data[feature] = (data[feature] - mean) / std
    return data

def eliminate_low_prices(data, thresholdBA):
    """Remove Buenos Aires rows whose price falls below the given threshold."""
    is_ba = data["lat"] < 0
    low_prices_ba = is_ba & (data["precio"] < thresholdBA)
    return data[~low_prices_ba].copy()



