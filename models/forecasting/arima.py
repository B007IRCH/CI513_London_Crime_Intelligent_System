import pandas as pd
from pmdarima import auto_arima
from utils.preprocess import load_and_prepare_data
from .utils import plot_forecast
from .utils import select_borough, select_crime_type


def run_arima():
    df = load_and_prepare_data()
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    monthly = df.groupby('date')['crime_count'].sum().sort_index()

    model = auto_arima(monthly, seasonal=True, m=12, trace=False, suppress_warnings=True)
    forecast, conf_int = model.predict(n_periods=24, return_conf_int=True)
    future_dates = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

    plot_forecast(monthly, forecast, conf_int, future_dates, "ARIMA Forecast (All London)")


def run_arima_filtered():
    df = load_and_prepare_data()
    borough = select_borough(df)
    crime = select_crime_type(df)

    if borough:
        df = df[df['borough'].str.lower() == borough.lower()]
    if crime:
        df = df[df['crime_type'].str.lower() == crime.lower()]

    if df.empty:
        print("No data found for the given filter.")
        return

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    monthly = df.groupby('date')['crime_count'].sum().sort_index()

    if len(monthly) < 24:
        print("Not enough data points to forecast.")
        return

    model = auto_arima(monthly, seasonal=True, m=12, trace=False, suppress_warnings=True)
    forecast, conf_int = model.predict(n_periods=24, return_conf_int=True)
    future_dates = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

    title = f"ARIMA Forecast ({borough or 'All Boroughs'}, {crime or 'All Crime Types'})"
    plot_forecast(monthly, forecast, conf_int, future_dates, title)
