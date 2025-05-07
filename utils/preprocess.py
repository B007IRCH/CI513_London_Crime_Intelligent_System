import pandas as pd

def load_and_prepare_data():
    print("[Preprocessing] Loading dataset...")
    df = pd.read_csv("data/London_Crime_2008_2016.csv")

    print(f"[Preprocessing] Columns in dataset:\n{df.columns.tolist()}")

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    print(f"[Preprocessing] Cleaned columns:\n{df.columns.tolist()}")

    # Try to print a few rows
    print("[Preprocessing] First 3 rows:")
    print(df.head(3))

    return df
