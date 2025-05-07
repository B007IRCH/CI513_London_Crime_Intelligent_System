import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.signal import savgol_filter
from utils.preprocess import load_and_prepare_data
from .utils import plot_forecast
from .utils import select_borough, select_crime_type


def run_lstm_forecast():
    df = load_and_prepare_data()
    borough = select_borough(df)
    crime = select_crime_type(df)
    smooth = input("Apply noise smoothing? (y/n): ").lower() == 'y'

    if borough:
        df = df[df['borough'].str.lower() == borough.lower()]
    if crime:
        df = df[df['crime_type'].str.lower() == crime.lower()]

    if df.empty:
        print("No data found for the given filter.")
        return

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    monthly = df.groupby('date')['crime_count'].sum().sort_index()

    if len(monthly) < 30:
        print("Insufficient time series length for LSTM training.")
        return

    monthly_values = savgol_filter(monthly.values, window_length=5, polyorder=2) if smooth else monthly.values

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

    forecast = []
    input_seq = scaled[-lookback:]
    for _ in range(24):
        pred = model.predict(input_seq.reshape(1, lookback, 1), verbose=0)
        forecast.append(pred[0, 0])
        input_seq = np.append(input_seq[1:], pred, axis=0)

    forecast_rescaled = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

    plot_forecast(monthly, forecast_rescaled, None, future_dates, "LSTM Forecast")