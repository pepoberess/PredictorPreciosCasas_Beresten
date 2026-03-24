import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, X, y, L1=0, L2=0):
        self.data = X
        X_array = np.array(X, dtype=float)
        ones = np.ones((X_array.shape[0], 1))
        self.X = np.hstack([ones, X_array])
        self.y = np.array(y, dtype=float)
        self.L1 = L1
        self.L2 = L2

    
    def pseudo_inverse(self):
        X = self.X
        y = self.y
        ridge = self.L2
        pseudo_inverse = np.linalg.inv(X.T @ X + ridge * np.eye(X.shape[1])) @ X.T
        w_opt = pseudo_inverse @ y
        self.coef = w_opt
        return self.coef
    
    def gradient_descent(self, tol=1e-6, max_iter=1000):
        X = self.X
        y = self.y
        Wlasso = self.L1

        n_samples, n_features = X.shape
        w_opt = np.zeros(n_features)

        for _ in range(max_iter):
            y_pred = X @ w_opt
            error = y_pred - y
            grad = (1 / n_samples) * (X.T @ error)
            s = self.backtracking_line_search(X, y, w_opt, grad)
            w_new = w_opt - s * (grad + Wlasso * np.sign(w_opt))

            if np.linalg.norm(w_new - w_opt) < tol:
                break
            w_opt = w_new

        self.coef = w_opt
        return self.coef

    def print_coefficients(self):
        X = self.data
        features = ["bias"] + list(X.columns)
        for feature, coef in zip(features, self.coef):
            print(f"{feature}: optimum weight = {coef}")
    
    def get_stats(self, data, features):
        stats = {}
        for feature in features:
            mean = data[feature].mean()
            std = data[feature].std()
            stats[feature] = (mean, std)
        return stats
    
    
    def backtracking_line_search(self, X, y, w, grad, s=1.0, beta=0.5, c=1e-4):
        n = len(y)
        Wlasso = self.L1
        def error_function(w):
            lasso = Wlasso * np.sum(np.abs(w))
            return lasso + (1/n) * np.linalg.norm(X @ w - y)**2

        current_cost = error_function(w)

        while True:
            w_new = w - s * grad
            new_cost = error_function(w_new)

            if new_cost <= current_cost - c * s * np.linalg.norm(grad)**2:
                break
            s *= beta 
        
        return s


def denormalize_dataset(model, statistics):
    y_pred_norm = model.X @ model.coef
    
    mean_y, std_y = statistics["precio"]
    y_pred = y_pred_norm * std_y + mean_y
    y_real = model.y * std_y + mean_y
    
    features = list(model.data.columns)
    X_denorm = {}
    for i, feature in enumerate(features):
        if statistics.get(feature) is None:
            X_denorm[feature] = model.X[:, i + 1]
            continue
        mean, std = statistics[feature]
        X_denorm[feature] = model.X[:, i + 1] * std + mean  # i+1 por el bias
    
    X_df = pd.DataFrame(X_denorm)
    return X_df, y_real, y_pred

