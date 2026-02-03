# **Exploring ML Concepts**

# Regression Models: kNN and Ridge Regression

This repository contains custom implementations of regression models and evaluation utilities developed for educational and experimental purposes. The focus is on understanding model behaviour, regularisation, and cross-validation rather than relying on high-level library abstractions.

## Contents

- **k-Nearest Neighbours Regression**
  - Custom kNN regressor
  - Custom cross-validation splitter
  - Internal cross-validation for optimal k selection

- **Ridge Regression**
  - Linear ridge regression trained using stochastic gradient descent (SGD)
  - Polynomial feature expansion
  - Analysis of regularisation effects using synthetic data

---

## Files

### `KNN_CV_experiment.ipynb`
Implements k-nearest neighbours regression and custom cross-validation utilities.

Includes:
- `KnnRegressor` – kNN regression using a KD-tree for efficient neighbour search  
- `LFold` – repeated random train–test splitting cross-validation scheme  
- `KnnRegressorCV` – kNN regressor with internal cross-validation for selecting the optimal number of neighbours  
- Evaluation utilities for analysing training and test error across different values of k  

---

### `Ridge_Regression_experiment.ipynb`
Implements ridge regression using stochastic gradient descent and explores regularisation effects.

Includes:
- `SGDRidgeRegression` – linear ridge regression trained via mini-batch SGD  
- `PolynomialFeatures` – polynomial feature expansion for modelling nonlinear relationships  
- Synthetic data generation with additive Gaussian noise  
- Visualisation utilities for comparing true functions, model predictions, and data  

---

## Dependencies

This project requires the following Python packages:

- `numpy`
- `scikit-learn`
- `matplotlib`

You can install them using:

```bash
pip install numpy scikit-learn matplotlib
