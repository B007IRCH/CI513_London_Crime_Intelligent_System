from .arima import run_arima, run_arima_filtered
from .lstm import run_lstm_forecast

def forecast_menu():
    while True:
        print("\n=== Forecasting Options ===")
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
