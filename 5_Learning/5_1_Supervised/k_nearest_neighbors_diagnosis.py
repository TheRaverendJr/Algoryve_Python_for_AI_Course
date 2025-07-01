#!/usr/bin/env python3
"""
k_nearest_neighbors_diagnosis.py

What this does
--------------
Trains a k-Nearest Neighbors (k-NN) classifier to predict whether a patient has
Type 2 Diabetes based on two clinical measurements: BMI (Body Mass Index) and
fasting blood glucose. After training and evaluation, it visualizes the
decision boundary and test points.

Why it matters in healthcare
----------------------------
Early screening for diabetes using easily-measured vitals helps prioritize
patients for further lab tests. k-NN offers a simple, interpretable “if your
measurements are like those of diabetics, flag you” approach.

Tools & Libraries
-----------------
- numpy, pandas       : numeric data and DataFrame handling  
- scikit-learn        : `KNeighborsClassifier`, `train_test_split`, `accuracy_score`  
- matplotlib          : plotting decision regions  
- type hints & logging: good practice for readability and debugging  

Good Practices
--------------
- Encapsulated in `main()` function  
- Uses `if __name__ == "__main__"` guard  
- Type hints on functions  
- Logging for progress and results
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_synthetic_data(
    n_samples: int = 500,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates a synthetic dataset with 2 informative features that simulate
    (BMI, fasting_glucose) and a binary target (0=no diabetes, 1=diabetes).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=random_state
    )
    df = pd.DataFrame(X, columns=["BMI", "FastingGlucose"])
    return df, pd.Series(y, name="Diabetes")


def plot_decision_boundary(
    model: KNeighborsClassifier,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "k-NN Decision Boundary"
) -> None:
    """
    Plots the decision boundary learned by `model` over the feature space,
    with test points overlaid.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=50)
    plt.xlabel("BMI")
    plt.ylabel("Fasting Glucose")
    plt.title(title)
    plt.show()


def main() -> None:
    # 1. Load data
    df, y = load_synthetic_data()
    X = df.values

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values, test_size=0.3, random_state=0
    )
    logging.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Train 5-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"5-NN Accuracy on test set: {acc:.2%}")

    # 5. Visualize
    plot_decision_boundary(knn, X_test, y_test,
                           title=f"5-NN Decision Boundary (Acc={acc:.2%})")


if __name__ == "__main__":
    main()
