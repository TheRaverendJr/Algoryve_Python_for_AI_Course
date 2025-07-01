#!/usr/bin/env python3
"""
linear_regression_biomarker.py

What this does
--------------
Uses Ordinary Least Squares and Ridge regression to predict a continuous
biomarker level (e.g. cholesterol) from patient age and BMI. Compares models
by Mean Squared Error and plots fitted lines.

Why it matters in healthcare
----------------------------
Accurate regression on biomarkers supports non-invasive risk estimation and
monitoring trends over time.

Tools & Libraries
-----------------
- numpy, pandas       : data handling  
- scikit-learn        : `LinearRegression`, `Ridge`  
- matplotlib          : plotting  
- train_test_split, mean_squared_error  

Good Practices
--------------
- Data in pandas for clarity  
- Standard `main()` structure  
- Reporting multiple models for comparison
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_synthetic_biomarker(
    n_samples: int = 200, random_state: int = 2
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates synthetic data:
     - Features: age (20–80), BMI (18–35)
     - Target: cholesterol level approx 150 + 0.5*age + 2*BMI + noise
    """
    rng = np.random.RandomState(random_state)
    age = rng.uniform(20, 80, n_samples)
    bmi = rng.uniform(18, 35, n_samples)
    cholesterol = 150 + 0.5 * age + 2.0 * bmi + rng.randn(n_samples) * 10
    df = pd.DataFrame({"age": age, "bmi": bmi})
    return df, pd.Series(cholesterol, name="cholesterol")


def main() -> None:
    # 1. Load data
    df, y = load_synthetic_biomarker()
    X = df.values

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values, test_size=0.3, random_state=0
    )

    models = [
        (LinearRegression(), "OLS"),
        (Ridge(alpha=10.0, random_state=0), "Ridge α=10")
    ]

    plt.scatter(X_test[:, 0], y_test, label="Age vs Cholesterol", alpha=0.5)
    plt.scatter(X_test[:, 1], y_test, label="BMI vs Cholesterol", alpha=0.5)

    # 3. Train & plot each model
    for model, name in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"{name} MSE = {mse:.2f}")

        # If only one feature, plot a line; here we skip line plot for 2-D
        # but could plot partial dependence or residuals.

    plt.title("Regression Models on Biomarker Prediction")
    plt.xlabel("Feature Value")
    plt.ylabel("Cholesterol")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
