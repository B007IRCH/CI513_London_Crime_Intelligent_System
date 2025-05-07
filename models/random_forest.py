import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from utils.preprocess import load_and_prepare_data
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
from tqdm import tqdm

init(autoreset=True)

def run_random_forest():
    print(Fore.CYAN + "\n[Random Forest] Predicting borough-level crime count using important features..." + Style.RESET_ALL)

    df = load_and_prepare_data()

    # Define inputs and target
    X = df[['borough', 'crime_type', 'year', 'month']]
    y = df['crime_count']

    # Preprocessing (categorical: one-hot, numeric: passthrough)
    categorical_features = ['borough', 'crime_type']
    numeric_features = ['year', 'month']

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)

    # Manual Random Forest with progress bar
    n_trees = 50
    rf = RandomForestRegressor(n_estimators=1, warm_start=True, random_state=42, n_jobs=-1)

    print(Fore.BLUE + f"\n[Random Forest] Training model with {n_trees} trees..." + Style.RESET_ALL)
    for i in tqdm(range(1, n_trees + 1), desc="Training Progress"):
        rf.n_estimators = i
        rf.fit(X_processed, y)

    # Get feature names and importance
    feature_names = list(preprocessor.get_feature_names_out())
    importances = rf.feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Display Top 10
    top_10 = importance_df.head(10)
    print(Fore.YELLOW + "Top 10 most important features predicting crime count:" + Style.RESET_ALL)
    for i, row in top_10.iterrows():
        print(f"  {Fore.GREEN}• {row['Feature']}{Style.RESET_ALL} → {Fore.MAGENTA}{row['Importance']:.4f}{Style.RESET_ALL}")

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(top_10['Feature'], top_10['Importance'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.title("Top 10 Most Important Features (Random Forest)")
    plt.tight_layout()
    plt.show()

    # MSE on training data
    y_pred = rf.predict(X_processed)
    mse = mean_squared_error(y, y_pred)
    print(Fore.CYAN + f"\n[MSE] Mean Squared Error on training data: {mse:.2f}" + Style.RESET_ALL)
