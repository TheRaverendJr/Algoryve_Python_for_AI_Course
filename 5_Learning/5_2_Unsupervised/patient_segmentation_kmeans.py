#!/usr/bin/env python3
"""
patient_segmentation_kmeans.py

What this does
--------------
Performs k-means clustering on patient vital sign measurements (heart rate,
blood pressure). Visualizes clusters and their centroids.

Why it matters in healthcare
----------------------------
Unsupervised segmentation helps identify patient subgroups (e.g., “high risk”
vs. “stable”) for targeted interventions without requiring labeled outcomes.

Tools & Libraries
-----------------
- numpy, pandas       : data handling  
- scikit-learn        : `KMeans` for clustering  
- matplotlib          : plotting clusters  
- standard scaler     : feature normalization  

Good Practices
--------------
- Feature scaling before clustering  
- Reproducible randomness  
- Clear cluster visualization
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_vitals_data(
    n_samples: int = 300, random_state: int = 3
) -> pd.DataFrame:
    """
    Synthetic patient vital signs:
    - heart_rate ~ N(70, 10)
    - systolic_bp ~ N(120, 15)
    """
    rng = np.random.RandomState(random_state)
    hr = rng.normal(70, 10, n_samples)
    sbp = rng.normal(120, 15, n_samples)
    return pd.DataFrame({"heart_rate": hr, "systolic_bp": sbp})


def main() -> None:
    # 1. Load and scale
    df = load_vitals_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # 2. Fit k-means
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    logging.info("Cluster sizes: %s", np.bincount(labels))

    # 3. Visualize
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="X",
                s=200, c="k", label="Centroids")
    plt.xlabel("Scaled Heart Rate")
    plt.ylabel("Scaled Systolic BP")
    plt.title("Patient Segmentation via k-Means")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
