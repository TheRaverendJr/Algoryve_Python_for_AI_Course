#!/usr/bin/env python3
"""
support_vector_machine_pathology.py

What this does
--------------
Uses a linear Support Vector Machine (SVM) to classify whether a tissue
sample is malignant vs. benign based on two extracted image features.
Visualizes support vectors and margin.

Why it matters in healthcare
----------------------------
Accurate, high-margin boundaries reduce risk of misdiagnosis in critical
pathology tasks. SVMs provide robust generalization.

Tools & Libraries
-----------------
- numpy               : numeric arrays  
- scikit-learn        : `SVC` for linear SVM  
- matplotlib          : plotting hyperplane & margins  
- joblib              : model persistence (example comment)

Good Practices
--------------
- Clear separation of data loading, training, and plotting  
- Use of `random_state` for reproducibility  
"""

import logging
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_synthetic_pathology(
    n_samples: int = 300, random_state: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate toy 2-D data representing two features extracted from
    pathology images, with a clear but noisy margin.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        class_sep=1.0,
        flip_y=0.1,
        random_state=random_state
    )
    return X, y


def plot_svm_decision(
    svm: SVC, X: np.ndarray, y: np.ndarray, title: str = "SVM Pathology Classifier"
) -> None:
    """
    Plot the decision boundary, margins, and support vectors.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=40)
    # Plot support vectors
    sv = svm.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors="none",
                edgecolors="k", label="Support Vectors")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


def main() -> None:
    X, y = load_synthetic_pathology()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    svm = SVC(kernel="linear", C=1.0, random_state=0)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    logging.info("\n" + classification_report(y_test, y_pred))

    plot_svm_decision(svm, X_test, y_test,
                      title="Linear SVM (Test Set)")


if __name__ == "__main__":
    main()
