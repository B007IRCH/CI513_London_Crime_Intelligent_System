import sys

def show_menu():
    print("\n=== CI513 London Crime Intelligent System ===")
    print("1. Run K-Means Clustering with Silhouette Score")
    print("2. Run Random Forest Feature Importance")
    print("3. Run Forecasting (ARIMA)")
    print("4. Run Case-Based Reasoning (CBR)")
    print("0. Exit")

def main():
    while True:
        show_menu()
        choice = input("Select an option (0-4): ")

        if choice == "1":
            from models.clustering import run_kmeans
            run_kmeans()
        elif choice == "2":
            from models.random_forest import run_random_forest
            run_random_forest()
        elif choice == "3":
            from models.forecasting import run_forecasting
            run_forecasting()
        elif choice == "4":
            from models.cbr import run_cbr
            run_cbr()
        elif choice == "0":
            print("Exiting program. Goodbye!")
            sys.exit()
        else:
            print("Invalid option. Please enter a number between 0 and 4.")

if __name__ == "__main__":
    main()
