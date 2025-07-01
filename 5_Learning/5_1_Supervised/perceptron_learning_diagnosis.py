#!/usr/bin/env python3
"""
perceptron_learning_diagnosis.py

What this file does
-------------------
This script demonstrates how to use the Perceptron algorithm to build a simple,
linear classifier for detecting hypertension (high blood pressure) in patients
based on two features:

  1. Systolic Blood Pressure (SBP)
  2. Diastolic Blood Pressure (DBP)

We generate a synthetic dataset of “normal” vs. “hypertensive” patients, fit a
Perceptron model using scikit-learn (which implements stochastic updates, early
stopping, and shuffling under the hood), evaluate its performance on held-out
data, and plot the resulting decision boundary overlaid on the test points.

Why it matters in healthcare
----------------------------
Early, automated flagging of potential hypertension allows for preventive
interventions—dietary counseling, further lab tests, or medication—before
serious cardiovascular events occur. A Perceptron (a basic single-layer linear
model) can be deployed on lightweight edge devices (wearables, clinic kiosks)
to give immediate, interpretable “yes/no” alerts.

Key libraries / tools
---------------------
- **numpy**            : numerical array operations  
- **matplotlib**       : plotting data and decision boundary  
- **scikit-learn**     :  
  - `Perceptron` for the model  
  - `StandardScaler` for feature normalization  
  - `train_test_split` to hold out test data  
  - `classification_report` to summarize precision/recall/F1  
- **logging**          : progress and results reporting  
- **type hints**       : function signatures for readability  

Good programming practices
--------------------------
- Encapsulate logic in a `main()` function and guard with
  `if __name__ == "__main__":`
- Use `train_test_split` with a fixed `random_state` for reproducibility.
- Scale features via `StandardScaler` before training.
- Leverage scikit-learn’s built-in shuffling, early stopping, and convergence
  criteria instead of rolling your own loop that may never converge on noisy data.
- Provide detailed docstrings and inline comments to explain each step.
"""

import logging
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_bp_data(
    n_samples_per_class: int = 200,
    random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset for blood pressure classification.

    Outputs:
      X : array of shape (2*n_samples_per_class, 2)
          Each row is [systolic_BP, diastolic_BP].
      y : array of shape (2*n_samples_per_class,)
          Labels: +1 for hypertensive, -1 for normal.
    """
    rng = np.random.RandomState(random_state)

    # Class +1: hypertensive patients, centered at (150, 95)
    hypertensive = rng.randn(n_samples_per_class, 2) * 5 + np.array([150, 95])
    # Class -1: normal patients, centered at (120, 80)
    normal = rng.randn(n_samples_per_class, 2) * 5 + np.array([120, 80])

    X = np.vstack([hypertensive, normal])
    y = np.hstack([np.ones(n_samples_per_class), -np.ones(n_samples_per_class)])

    return X, y


def plot_decision_boundary(
    model: Perceptron,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Perceptron Decision Boundary"
) -> None:
    """
    Plot the learned linear decision boundary of `model` in the original feature
    space, along with the data points (X, y).

    - We create a grid over the scaled feature space,
    - Predict class for each grid point,
    - Then transform back to original scale for plotting axes.
    """
    # Create a fine grid in the *scaled* space
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    # Flatten and stack, then scale
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    # Predict on grid
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=50)
    plt.xlabel("Systolic BP (mmHg)")
    plt.ylabel("Diastolic BP (mmHg)")
    plt.title(title)
    plt.show()


def main() -> None:
    # 1) Generate data
    X, y = generate_bp_data()
    logging.info(f"Generated {len(X)} samples: {np.bincount(y.astype(int)+1)} counts per class")

    # 2) Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    logging.info(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")

    # 3) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4) Initialize & train Perceptron
    #    - tol=1e-3   → stop when updates < tol
    #    - max_iter=1000
    #    - shuffle=True → randomize order each epoch
    #    - random_state for reproducibility
    model = Perceptron(
        tol=1e-3,
        max_iter=1000,
        shuffle=True,
        random_state=0
    )
    model.fit(X_train_scaled, y_train)
    logging.info("Training complete")
    logging.info(f"Number of iterations run: {model.n_iter_}")

    # 5) Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, digits=3,
                                   target_names=["Normal", "Hypertensive"])
    print("\nClassification Report on Test Set:")
    print(report)

    # 6) Plot decision boundary + test points
    plot_decision_boundary(
        model,
        scaler,
        X_test,
        y_test,
        title="Perceptron (Hypertension) Decision Boundary"
    )


if __name__ == "__main__":
    main()

















"""
In a production-grade healthcare setting, you’d layer on:

Data Ingestion & Validation

Pull actual EHR or device‐record data via secure APIs or database queries.

Validate ranges (e.g. 90 ≤ SBP ≤ 200) and handle missing values.

Feature Engineering & Pipelines

Compute additional features (pulse pressure, BMI, age-adjusted BP).

Use scikit-learn’s Pipeline (or ColumnTransformer) to chain scaling, imputation, and modeling.

Robust Model Selection

Wrap training in GridSearchCV or RandomizedSearchCV for hyperparameter tuning (learning rate, max_iter).

Cross-validation (e.g. 5-fold stratified) to estimate generalization.

Imbalanced Classes Handling

If hypertensive cases are rarer, apply oversampling (SMOTE) or class‐weighting in the loss.

Evaluation Beyond Accuracy

ROC curves / AUC thresholds to tune operating points for sensitivity vs. specificity.

Calibration plots to check predicted probabilities (if you switch to logistic regression).

Model Persistence & Deployment

Serialize your pipeline with joblib or pickle.

Wrap as a REST API (FastAPI, Flask), include input validation, versioning, logging of predictions.

Monitoring & Governance

Track data drift (are incoming BP values outside training distribution?).

Regularly re-validate performance on fresh hold-out sets.

Maintain audit trails for regulatory compliance (HIPAA, GDPR).

Documentation & Testing

Write unit tests for each component (data generator, pipeline, inference).

Provide end-to-end integration tests with synthetic EHR records.

Document expected input schemas, feature definitions, and model limitations.


"""













































































# #!/usr/bin/env python3
# """
# perceptron_learning_diagnosis.py

# What this does
# --------------
# Implements the Perceptron learning algorithm from scratch to classify whether
# a patient has hypertension based on (systolic_BP, diastolic_BP). Plots the
# learned linear boundary and reports the number of epochs to converge.

# Why it matters in healthcare
# ----------------------------
# A simple linear classifier can be deployed on low-power devices (e.g. wearables)
# to flag high-risk patients in real time.

# Tools & Libraries
# -----------------
# - numpy, matplotlib    : numeric operations and plotting  
# - type hints           : clarity of function signatures  
# - structured code      : separates data generation, training, plotting  

# Good Practices
# --------------
# - Vectorized operations where possible  
# - Early stopping when zero classification errors  
# - Clear docstrings & logging  
# """

# import logging
# from typing import Tuple

# import numpy as np
# import matplotlib.pyplot as plt

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# def generate_bp_data(
#     n_samples: int = 200, random_state: int = 0
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generate synthetic blood-pressure data:
#     - Class +1 (hypertensive) centered at (150, 95)
#     - Class -1 (normal) centered at (120, 80)
#     """
#     rng = np.random.RandomState(random_state)
#     pos = rng.randn(n_samples, 2) * 5 + np.array([150, 95])
#     neg = rng.randn(n_samples, 2) * 5 + np.array([120, 80])
#     X = np.vstack([pos, neg])
#     y = np.hstack([np.ones(n_samples), -np.ones(n_samples)])
#     return X, y


# def perceptron_train(
#     X: np.ndarray, y: np.ndarray, alpha: float = 0.01, max_epochs: int = 1000
# ) -> Tuple[np.ndarray, int]:
#     """
#     Train perceptron with bias:
#     - w has shape (3,), where w[0] is bias.
#     - Input vector is [1, x1, x2].
#     Returns learned weights and epochs to converge.
#     """
#     # Initialize weights to zero
#     w = np.zeros(3)
#     N = X.shape[0]

#     for epoch in range(1, max_epochs + 1):
#         errors = 0
#         for xi, yi in zip(X, y):
#             x_vec = np.hstack([1.0, xi])  # bias term
#             pred = np.sign(np.dot(w, x_vec))
#             if pred == 0:
#                 pred = -1  # break ties
#             if pred != yi:
#                 w += alpha * yi * x_vec
#                 errors += 1
#         logging.info(f"Epoch {epoch}: {errors} misclassifications")
#         if errors == 0:
#             return w, epoch
#     return w, max_epochs


# def plot_boundary(
#     w: np.ndarray, X: np.ndarray, y: np.ndarray, title: str = "Perceptron Boundary"
# ) -> None:
#     """
#     Plot 2D data points and the linear decision boundary defined by w.
#     """
#     # Decision line: w0 + w1*x + w2*y = 0 → y = -(w0 + w1*x)/w2
#     xs = np.linspace(X[:, 0].min() - 5, X[:, 0].max() + 5, 200)
#     ys = -(w[0] + w[1] * xs) / w[2]

#     plt.plot(xs, ys, "k-", label="Decision boundary")
#     plt.scatter(X[y == 1, 0], X[y == 1, 1], marker="o", label="Hypertensive")
#     plt.scatter(X[y == -1, 0], X[y == -1, 1], marker="x", label="Normal")
#     plt.xlabel("Systolic BP")
#     plt.ylabel("Diastolic BP")
#     plt.title(title)
#     plt.legend()
#     plt.show()


# def main() -> None:
#     X, y = generate_bp_data()
#     w, epochs = perceptron_train(X, y)
#     logging.info(f"Converged in {epochs} epochs. Weights: {w}")
#     plot_boundary(w, X, y,
#                   title=f"Perceptron (epochs={epochs})")


# if __name__ == "__main__":
#     main()








# """
# What you’re seeing—an endless oscillation between ~2 – 4 misclassifications per epoch and a decision boundary that sits “well below” your two clouds of hypertensive vs. normal points—is exactly what happens when the Perceptron algorithm:

# Can’t find a perfect linear separator in your data (because we added noise), and

# Uses a fixed presentation order and fixed learning rate, so it ends up “chasing its tail” around a suboptimal boundary rather than converging.


# """