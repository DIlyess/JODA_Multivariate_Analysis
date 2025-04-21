import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint


class RegressionModelSelector:
    """
    A class to try multiple regression models, tune them with randomized search,
    and select the best one based on RMSE.
    """

    def __init__(self, random_state=42, n_iter=50, cv=5, verbose=1):
        """
        Initialize the model selector.

        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        n_iter : int, default=50
            Number of parameter settings that are sampled in RandomizedSearchCV
        cv : int, default=5
            Number of cross-validation folds
        verbose : int, default=1
            Verbosity level
        """
        self.random_state = random_state
        self.n_iter = n_iter
        self.cv = cv
        self.verbose = verbose
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float("inf")  # For RMSE (lower is better)
        self.best_model_name = None
        self.scaler = StandardScaler()

        # Set up model dictionary with parameter grids
        self._setup_models()

    def _setup_models(self):
        """Set up the models to evaluate with their parameter grids."""

        # Linear Regression (no hyperparameters to tune)
        self.models["Linear"] = {"model": LinearRegression(), "params": {}}

        # Ridge Regression
        self.models["Ridge"] = {
            "model": Ridge(random_state=self.random_state),
            "params": {
                "alpha": uniform(0.001, 10.0),
                "fit_intercept": [True, False],
                "solver": [
                    "auto",
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                ],
            },
        }

        # Lasso Regression
        self.models["Lasso"] = {
            "model": Lasso(random_state=self.random_state),
            "params": {
                "alpha": uniform(0.001, 10.0),
                "fit_intercept": [True, False],
                "selection": ["cyclic", "random"],
            },
        }

        # ElasticNet
        self.models["ElasticNet"] = {
            "model": ElasticNet(random_state=self.random_state),
            "params": {
                "alpha": uniform(0.001, 10.0),
                "l1_ratio": uniform(0, 1),
                "fit_intercept": [True, False],
                "selection": ["cyclic", "random"],
            },
        }

        # SVR
        self.models["SVR"] = {
            "model": SVR(),
            "params": {
                "C": uniform(0.1, 10),
                "epsilon": uniform(0.01, 1.0),
                "kernel": ["linear", "poly", "rbf"],
                "gamma": ["scale", "auto"] + list(uniform(0.001, 1).rvs(5)),
            },
        }

        # Random Forest
        self.models["RandomForest"] = {
            "model": RandomForestRegressor(random_state=self.random_state),
            "params": {
                "n_estimators": randint(50, 300),
                "max_depth": [None] + list(randint(5, 30).rvs(5)),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["auto", "sqrt", "log2", None],
            },
        }

        # Gradient Boosting
        self.models["GradientBoosting"] = {
            "model": GradientBoostingRegressor(random_state=self.random_state),
            "params": {
                "n_estimators": randint(50, 300),
                "learning_rate": uniform(0.01, 0.3),
                "max_depth": randint(3, 10),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "subsample": uniform(0.6, 0.4),
                "max_features": ["sqrt", "log2", None],
            },
        }

    def fit(self, X, y):
        """
        Fit and tune all models.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples
        y : array-like of shape (n_samples,)
            The target values

        Returns:
        --------
        self : object
            Returns self.
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        for model_name, model_info in self.models.items():
            start_time = time.time()

            if self.verbose >= 1:
                print(f"\nTuning {model_name}...")

            # If model has parameters to tune
            if model_info["params"]:
                search = RandomizedSearchCV(
                    estimator=model_info["model"],
                    param_distributions=model_info["params"],
                    n_iter=self.n_iter,
                    cv=self.cv,
                    scoring="neg_root_mean_squared_error",
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=max(0, self.verbose - 1),
                )
                search.fit(X_scaled, y)
                best_model = search.best_estimator_
                best_params = search.best_params_

                # Get cross-validation scores for the best model
                cv_rmse = -search.best_score_
                cv_scores = cross_val_score(
                    best_model,
                    X_scaled,
                    y,
                    cv=self.cv,
                    scoring="neg_root_mean_squared_error",
                )
                cv_mae = -cross_val_score(
                    best_model,
                    X_scaled,
                    y,
                    cv=self.cv,
                    scoring="neg_mean_absolute_error",
                ).mean()
                cv_r2 = cross_val_score(
                    best_model, X_scaled, y, cv=self.cv, scoring="r2"
                ).mean()
            else:
                # For models with no hyperparameters to tune (Linear Regression)
                best_model = model_info["model"]
                best_params = {}

                # Get cross-validation scores
                cv_rmse_scores = cross_val_score(
                    best_model,
                    X_scaled,
                    y,
                    cv=self.cv,
                    scoring="neg_root_mean_squared_error",
                )
                cv_rmse = -cv_rmse_scores.mean()
                cv_mae = -cross_val_score(
                    best_model,
                    X_scaled,
                    y,
                    cv=self.cv,
                    scoring="neg_mean_absolute_error",
                ).mean()
                cv_r2 = cross_val_score(
                    best_model, X_scaled, y, cv=self.cv, scoring="r2"
                ).mean()
                cv_scores = cv_rmse_scores

            # Train the final model on the whole dataset
            best_model.fit(X_scaled, y)

            # Store results
            elapsed_time = time.time() - start_time

            self.results[model_name] = {
                "model": best_model,
                "params": best_params,
                "rmse": cv_rmse,
                "mae": cv_mae,
                "r2": cv_r2,
                "all_rmse_scores": -cv_scores,
                "training_time": elapsed_time,
            }

            if self.verbose >= 1:
                print(
                    f"  {model_name} - RMSE: {cv_rmse:.4f}, MAE: {cv_mae:.4f}, R²: {cv_r2:.4f}, Time: {elapsed_time:.2f} sec"
                )

            # Update best model if this one is better
            if cv_rmse < self.best_score:
                self.best_score = cv_rmse
                self.best_model = best_model
                self.best_model_name = model_name

        if self.verbose >= 1:
            print(
                f"\nBest model: {self.best_model_name} with RMSE: {self.best_score:.4f}"
            )

        return self

    def predict(self, X):
        """
        Predict using the best model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            The predicted values
        """
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)

    def get_best_model(self):
        """Return the best model."""
        return self.best_model

    def get_best_model_name(self):
        """Return the name of the best model."""
        return self.best_model_name

    def get_results_df(self):
        """
        Return a DataFrame with the results of all models.

        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with model results
        """
        results_list = []

        for model_name, model_result in self.results.items():
            results_list.append(
                {
                    "Model": model_name,
                    "RMSE": model_result["rmse"],
                    "MAE": model_result["mae"],
                    "R²": model_result["r2"],
                    "Training Time (s)": model_result["training_time"],
                }
            )

        return pd.DataFrame(results_list).sort_values("RMSE")

    def plot_results(self, figsize=(12, 10)):
        """
        Plot the results of all models.

        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Figure size

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        """
        results_df = self.get_results_df()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot RMSE
        sns.barplot(x="Model", y="RMSE", data=results_df, ax=axes[0, 0])
        axes[0, 0].set_title("RMSE by Model")
        axes[0, 0].set_xticklabels(
            axes[0, 0].get_xticklabels(), rotation=45, ha="right"
        )

        # Plot MAE
        sns.barplot(x="Model", y="MAE", data=results_df, ax=axes[0, 1])
        axes[0, 1].set_title("MAE by Model")
        axes[0, 1].set_xticklabels(
            axes[0, 1].get_xticklabels(), rotation=45, ha="right"
        )

        # Plot R²
        sns.barplot(x="Model", y="R²", data=results_df, ax=axes[1, 0])
        axes[1, 0].set_title("R² by Model")
        axes[1, 0].set_xticklabels(
            axes[1, 0].get_xticklabels(), rotation=45, ha="right"
        )

        # Plot Training Time
        sns.barplot(x="Model", y="Training Time (s)", data=results_df, ax=axes[1, 1])
        axes[1, 1].set_title("Training Time by Model")
        axes[1, 1].set_xticklabels(
            axes[1, 1].get_xticklabels(), rotation=45, ha="right"
        )

        plt.tight_layout()
        return fig

    def plot_model_comparison(self, figsize=(10, 6)):
        """
        Plot a comparison of all models with error bars.

        Parameters:
        -----------
        figsize : tuple, default=(10, 6)
            Figure size

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        """
        # Create DataFrame for plotting
        model_names = []
        rmse_means = []
        rmse_stds = []

        for model_name, model_result in self.results.items():
            model_names.append(model_name)
            scores = model_result["all_rmse_scores"]
            rmse_means.append(scores.mean())
            rmse_stds.append(scores.std())

        # Create DataFrame
        df = pd.DataFrame({"Model": model_names, "RMSE": rmse_means, "STD": rmse_stds})

        # Sort by RMSE
        df = df.sort_values("RMSE")

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot bars with error bars
        bars = ax.barh(df["Model"], df["RMSE"], xerr=df["STD"], alpha=0.7, capsize=5)

        # Highlight the best model
        best_idx = df["Model"].tolist().index(self.best_model_name)
        bars[best_idx].set_color("green")

        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("Model Comparison with Cross-Validation RMSE")
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Add RMSE values as text
        for i, (rmse, std) in enumerate(zip(df["RMSE"], df["STD"])):
            ax.text(rmse + std + 0.01, i, f"{rmse:.4f} ± {std:.4f}", va="center")

        plt.tight_layout()
        return fig


# Example usage:
# selector = RegressionModelSelector(n_iter=20, cv=5)
# selector.fit(X, y)
# best_model = selector.get_best_model()
# results_df = selector.get_results_df()
# selector.plot_results()
# y_pred = selector.predict(X_test)
