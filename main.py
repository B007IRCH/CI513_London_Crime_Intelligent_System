import sys
from colorama import Fore, Style, init

init(autoreset=True)

def show_menu():
    print(Fore.CYAN + "\n=== CI513 London Crime Intelligent System ===" + Style.RESET_ALL)
    print(Fore.YELLOW + "1." + Style.RESET_ALL + " Run " + Fore.GREEN + "K-Means Clustering" + Style.RESET_ALL)
    print(Fore.YELLOW + "2." + Style.RESET_ALL + " Run " + Fore.GREEN + "Random Forest Feature Importance")
    print(Fore.YELLOW + "3." + Style.RESET_ALL + " Run " + Fore.GREEN + "Forecasting Menu (ARIMA/LSTM)")
    print(Fore.YELLOW + "4." + Style.RESET_ALL + " Run " + Fore.GREEN + "Case-Based Reasoning (CBR)")
    print(Fore.YELLOW + "5." + Style.RESET_ALL + " Run " + Fore.GREEN + "DBSCAN Clustering")
    print(Fore.YELLOW + "0." + Style.RESET_ALL + " Exit")


def main():
    while True:
        show_menu()
        choice = input(Fore.MAGENTA + "Select an option (0-4): " + Style.RESET_ALL)

        if choice == "1":
            from models.clustering import run_kmeans
            run_kmeans()
        elif choice == "2":
            from models.random_forest import run_random_forest
            run_random_forest()
        elif choice == "3":
            from models.forecasting import forecast_menu
            forecast_menu()
        elif choice == "4":
            from models.cbr import run_cbr
            run_cbr()
        elif choice == "5":
            from models.clustering import run_dbscan 
            run_dbscan()
        elif choice == "0":
            print(Fore.RED + "Exiting program. Goodbye!" + Style.RESET_ALL)
            sys.exit()
        else:
            print(Fore.RED + "Invalid option. Please enter a number between 0 and 4." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
