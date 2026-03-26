import numpy as np
import pandas as pd

class LinearRegression:
    """Linear regression model supporting closed-form (pseudo-inverse) and gradient descent solvers,
    with optional L1 (Lasso) and L2 (Ridge) regularization."""

    def __init__(self, X, y, L1=0, L2=0):
        """Prepend a bias column to X and store hyperparameters."""
        self.data = X
        X_array = np.array(X, dtype=float)
        ones = np.ones((X_array.shape[0], 1))
        self.X = np.hstack([ones, X_array])
        self.y = np.array(y, dtype=float)
        self.L1 = L1
        self.L2 = L2

    def pseudo_inverse(self):
        """Solve for weights using the normal equation; applies Ridge regularization if L2 > 0."""
        X = self.X
        y = self.y
        ridge = self.L2
        reg = ridge * np.eye(X.shape[1])
        reg[0, 0] = 0

        if ridge == 0:
            pseudo_inverse = np.linalg.pinv(X)
        else:
            pseudo_inverse = np.linalg.inv(X.T @ X + reg) @ X.T

        w_opt = pseudo_inverse @ y
        self.coef = w_opt
        return self.coef
    
    def gradient_descent(self, tol=1e-6, max_iter=1000):
        """Solve for weights using gradient descent with L1 (Lasso) penalty and backtracking line search."""
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
    
    def predict(self, X_new):
        X_array = np.array(X_new, dtype=float)
        ones = np.ones((X_array.shape[0], 1))
        X_with_bias = np.hstack([ones, X_array])
        return X_with_bias @ self.coef

    def backtracking_line_search(self, X, y, w, grad, s=1.0, beta=0.5, c=1e-4):
        """Armijo-rule line search to find a step size that satisfies the sufficient decrease condition."""
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
    """Reverse normalization on both predictions and features using the stored statistics."""
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
        X_denorm[feature] = model.X[:, i + 1] * std + mean  # i+1 to skip the bias column
    
    X_df = pd.DataFrame(X_denorm)
    return X_df, y_real, y_pred

def denormalize_prediction(model, X_val, stats):
    """Return predictions for X_val scaled back to the original price range."""
    y_val_pred_norm = model.predict(X_val)
    mean_y, std_y = stats["precio"]
    y_val_pred = y_val_pred_norm * std_y + mean_y
    return y_val_pred
