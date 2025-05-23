# CI513 London Crime Intelligent System

This project was developed for the CI513 – Intelligent Systems 2 module at the University of Brighton. It applies artificial intelligence techniques to analyse and forecast crime trends across London boroughs from 2008 to 2016. The system integrates clustering, time-series forecasting, and case-based reasoning (CBR) through an interactive Django web application.

## Features

- Clustering: K-Means (with PCA and silhouette scoring) and DBSCAN for borough-level crime pattern detection.
- Forecasting: ARIMA and LSTM models for short-term crime trend prediction.
- Case-Based Reasoning: Suggests similar boroughs based on historical crime profiles.
- Visualisation: Crime heatmaps with Leaflet, and Chart.js visualisations for forecasts and cluster evaluation.
- Preprocessing: Includes standardisation, PCA, and Savitzky–Golay smoothing.
- Django Web App: Web-based interface for selecting inputs and running analyses interactively.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Pip (Python package manager)
- Git (to clone the repository)
- Virtualenv (optional but recommended for environment isolation)

### Pip Requiremnts
Django>=4.0
pandas
numpy
matplotlib
seaborn
scikit-learn
pmdarima
tensorflow
colorama
scipy


