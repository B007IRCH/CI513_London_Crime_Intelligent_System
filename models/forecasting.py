import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import load_and_prepare_data
from colorama import Fore, Style, init
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.signal import savgol_filter

init(autoreset=True)


def forecast_menu():
    while True:
        print(Fore.CYAN + "\n=== Forecasting Options ===" + Style.RESET_ALL)
        print("1. ARIMA Forecast (All London)")
        print("2. ARIMA Forecast (Specific Borough or Crime Type)")
        print("3. LSTM Forecast (Neural Network, with noise smoothing)")
        print("0. Back to Main Menu")
        choice = input("Select an option (0-3): ")

        if choice == "1":
            run_arima()
        elif choice == "2":
            run_arima_filtered()
        elif choice == "3":
            run_lstm_forecast()
        elif choice == "0":
            break
        else:
            print("Invalid choice.")


def run_arima():
    df = load_and_prepare_data()
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    monthly = df.groupby('date')['crime_count'].sum().sort_index()

    model = auto_arima(monthly, seasonal=True, m=12, trace=False, suppress_warnings=True)
    forecast, conf_int = model.predict(n_periods=24, return_conf_int=True)
    future_dates = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

    _plot_forecast(monthly, forecast, conf_int, future_dates, "ARIMA Forecast (All London)")


def run_arima_filtered():
    df = load_and_prepare_data()
    borough = input("Enter borough name (or leave blank): ").strip()
    crime = input("Enter crime type (or leave blank): ").strip()

    if borough:
        df = df[df['borough'].str.lower() == borough.lower()]
    if crime:
        df = df[df['crime_type'].str.lower() == crime.lower()]

    if df.empty:
        print(Fore.RED + "No data found for the given filter." + Style.RESET_ALL)
        return

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    monthly = df.groupby('date')['crime_count'].sum().sort_index()

    if len(monthly) < 24:
        print(Fore.RED + "Not enough data points to forecast." + Style.RESET_ALL)
        return

    model = auto_arima(monthly, seasonal=True, m=12, trace=False, suppress_warnings=True)
    forecast, conf_int = model.predict(n_periods=24, return_conf_int=True)
    future_dates = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

    title = f"ARIMA Forecast ({borough or 'All Boroughs'}, {crime or 'All Crime Types'})"
    _plot_forecast(monthly, forecast, conf_int, future_dates, title)


def run_lstm_forecast():
    df = load_and_prepare_data()
    borough = input("Enter borough (or leave blank): ").strip()
    crime = input("Enter crime type (or leave blank): ").strip()
    smooth = input("Apply noise smoothing? (y/n): ").lower() == 'y'

    if borough:
        df = df[df['borough'].str.lower() == borough.lower()]
    if crime:
        df = df[df['crime_type'].str.lower() == crime.lower()]

    if df.empty:
        print(Fore.RED + "No data found for the given filter." + Style.RESET_ALL)
        return

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    monthly = df.groupby('date')['crime_count'].sum().sort_index()

    if len(monthly) < 30:
        print(Fore.RED + "Insufficient time series length for LSTM training." + Style.RESET_ALL)
        return

    # Optional noise smoothing
    if smooth:
        monthly_values = savgol_filter(monthly.values, window_length=5, polyorder=2)
    else:
        monthly_values = monthly.values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(monthly_values.reshape(-1, 1))

    X, y = [], []
    lookback = 12
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    # Generate 24-month forecast
    forecast = []
    input_seq = scaled[-lookback:]
    for _ in range(24):
        pred = model.predict(input_seq.reshape(1, lookback, 1), verbose=0)
        forecast.append(pred[0, 0])
        input_seq = np.append(input_seq[1:], pred, axis=0)

    forecast_rescaled = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

    _plot_forecast(monthly, forecast_rescaled, None, future_dates, "LSTM Forecast")


def _plot_forecast(history, forecast, conf_int, future_dates, title):
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
