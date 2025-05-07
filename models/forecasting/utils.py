import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(history, forecast, conf_int, future_dates, title):
    forecast_series = pd.Series(forecast, index=future_dates)

    plt.figure(figsize=(13, 6))
    plt.plot(history, label='Historical', color='blue', linewidth=2)
    plt.plot(forecast_series, label='Forecast', linestyle='--', color='darkorange', linewidth=2)

    if conf_int is not None:
        plt.fill_between(future_dates, conf_int[:, 0], conf_int[:, 1], color='orange', alpha=0.3, label='Confidence Interval')

    plt.axvline(x=history.index[-1], color='gray', linestyle='--', linewidth=1)
    plt.text(history.index[-1], forecast_series.max(), 'Forecast Start', fontsize=9, color='gray', ha='left')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Crime Count", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def select_borough(df):
    boroughs = sorted(df['borough'].unique())
    print("\nSelect a borough:")
    for i, b in enumerate(boroughs):
        print(f"{i + 1}. {b}")
    choice = input("Enter number (or press Enter for all boroughs): ")
    if choice.isdigit() and 1 <= int(choice) <= len(boroughs):
        return boroughs[int(choice) - 1]
    return None


def select_crime_type(df):
    crimes = sorted(df['crime_type'].unique())
    print("\nSelect a crime type:")
    for i, c in enumerate(crimes):
        print(f"{i + 1}. {c}")
    choice = input("Enter number (or press Enter for all crimes): ")
    if choice.isdigit() and 1 <= int(choice) <= len(crimes):
        return crimes[int(choice) - 1]
    return None
