# Unsupervised Image Clustering with VGG16 Features

A compact project demonstrating unsupervised clustering of flower images using features extracted from a pre-trained VGG16 network. Two clustering algorithms are used: K-means (from scratch and via scikit-learn) and DBSCAN. Performance is evaluated with homogeneity and silhouette scores, with PCA visualizations.

## Overview

- Extract deep features from 210 images using `VGG16` (without fully-connected layers)
- Apply clustering: `K-means` (scratch + library) and `DBSCAN`
- Reduce dimensionality with `PCA` for visualization and DBSCAN stability
- Evaluate clustering quality using `homogeneity` and `silhouette` metrics

## Dataset

- Folder: `flower_images/`
- Images: 210 PNG files (e.g., `0001.png`)
- Labels: `flower_images/flower_labels.csv` (integer class per file)

## Pipeline

1) Preprocess images: resize to 224x224, convert to BGR-compatible format
2) Feature extraction: pass through `VGG16` (ImageNet weights, `include_top=False`)
3) Clustering:
   - K-means (scratch): random init, iterative assignment/update until convergence
   - K-means (sklearn): `KMeans(n_clusters=K)` tested for K in [2..15]
   - DBSCAN: run on PCA-reduced features, visualize core points and noise
4) Evaluation: compute `homogeneity_score` and `silhouette_score`; plot metric vs K
5) Visualization: PCA to 2D/3D scatter plots colored by cluster

## Key Findings

- VGG16 features provide separable structure enabling effective clustering
- K selection matters: evaluated via homogeneity and silhouette
- DBSCAN requires dimensionality reduction and careful tuning (eps, min_samples)

## Tech Stack

- Python, NumPy, Pandas, scikit-learn, TensorFlow/Keras (VGG16), OpenCV, Matplotlib

## How to Run

- Ensure images and labels are available under `CA3/flower_images/`
- Run the notebook `AI_CA3.ipynb` (GPU recommended for faster VGG16 feature extraction)
- Adjust hyperparameters: number of clusters (K), DBSCAN `eps` and `min_samples`

## Notes

- Metrics used:
  - Homogeneity: 1.0 indicates clusters contain samples from a single class
  - Silhouette: ranges [-1, 1], higher is better separation
- PCA is used both for visualization and to stabilize DBSCAN in high dimensions
