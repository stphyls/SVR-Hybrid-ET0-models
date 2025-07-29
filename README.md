# Hybrid SVR Models for Reference Evapotranspiration (ET₀) Estimation

This repository contains Python implementations of hybrid machine learning models for estimating reference evapotranspiration (ET₀) using Support Vector Regression (SVR) optimized with two metaheuristic algorithms:

- **SVR-CMAES**: Covariance Matrix Adaptation Evolution Strategy
- **SVR-GSA**: Gravitational Search Algorithm

The models are evaluated using multiple input configurations, and both sensitivity and uncertainty analyses are performed to assess model robustness and variable influence.

---

## 📁 Repository Structure

| File Name                   | Description |
|----------------------------|-------------|
| `7) C1-C4 CMAES.py`        | SVR-CMAES models using 4 input configurations (C1–C4) |
| `8) C1-C4 GSA.py`          | SVR-GSA models using 4 input configurations (C1–C4) |
| `9) LSA and GSA.py`        | Sensitivity analysis (Local and Global) |
| `Uncertainty CMAES.py`     | Bootstrap uncertainty analysis for SVR-CMAES |
| `Uncertainty GSA.py`       | Bootstrap uncertainty analysis for SVR-GSA |

---

## 🧠 Model Overview

The following hybrid SVR models are implemented:

- `SVR-GSA`: Combines Support Vector Regression with the Gravitational Search Algorithm
- `SVR-CMAES`: Combines Support Vector Regression with the Covariance Matrix Adaptation Evolution Strategy

Each model is evaluated with the following **input configurations**:

| Configuration | Input Variables |
|---------------|------------------|
| C1 | Temperature only |
| C2 | Temperature + Solar Radiation |
| C3 | Temperature + Solar Radiation + Wind Speed |
| C4 | Temperature + Solar Radiation + Wind Speed + Relative Humidity |

---

## 📊 Sensitivity Analysis

Two types of sensitivity analysis are conducted to evaluate the influence of input variables:

- **Local Sensitivity Analysis (LSA)**: One-at-a-time (OAT) method
- **Global Sensitivity Analysis (GSA)**: Extended Fourier Amplitude Sensitivity Test (eFAST)

---

## 🎯 Uncertainty Analysis

- **Bootstrap Resampling** is used to assess the uncertainty in ET₀ estimations from SVR-GSA and SVR-CMAES models.

---

## 📈 Evaluation Metrics

Model performance is assessed using the following statistical metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **Coefficient of Determination (R²)**
- **Global Performance Indicator (GPI)**
- **Willmott’s Modified Index of Agreement (d1)**
- **Kling-Gupta Efficiency (KGE)**

---

## ⚙️ Requirements

Install dependencies via `pip`:
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install SALib
pip install cma
