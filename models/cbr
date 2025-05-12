import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from utils.preprocess import load_and_prepare_data

def run_cbr():
    print("\n[CBR] Running Case-Based Reasoning...")

    df = load_and_prepare_data()

    # Create pivot table
    pivot = df.pivot_table(
        index='borough',
        columns='crime_type',
        values='crime_count',
        aggfunc='sum'
    ).fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot)
    scaled_df = pd.DataFrame(scaled_data, index=pivot.index, columns=pivot.columns)

    boroughs = list(scaled_df.index)
    print("\nAvailable Boroughs:")
    for i, b in enumerate(boroughs):
        print(f"{i+1}. {b}")

    try:
        choice = int(input("\nEnter the number of the borough to use as query case: ")) - 1
        query_borough = boroughs[choice]
    except (IndexError, ValueError):
        print("Invalid selection. Exiting CBR.")
        return

    query_vector = scaled_df.loc[query_borough].values.reshape(1, -1)
    distances = euclidean_distances(query_vector, scaled_df).flatten()
    top_indices = distances.argsort()[1:6]

    print(f"\nMost similar boroughs to '{query_borough}':")
    for idx in top_indices:
        print(f"  • {boroughs[idx]} (Distance: {distances[idx]:.3f})")

    # Prepare data for visualisation
    similar_names = [query_borough] + [boroughs[i] for i in top_indices]
    comparison_df = scaled_df.loc[similar_names]

    # Radar Chart
    categories = pivot.columns.tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    fig_radar = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)
    for name in comparison_df.index:
        values = comparison_df.loc[name].tolist()
        values += values[:1]
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], categories, size=8)
    ax.set_title(f"CBR Radar: Crime Profile Similarity - {query_borough}")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()

    # Heatmap
    dist_df = pd.DataFrame(euclidean_distances(comparison_df, comparison_df),
                           index=comparison_df.index, columns=comparison_df.index)

    fig_heatmap, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(dist_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, ax=ax)
    ax.set_title("CBR Similarity Heatmap (Euclidean Distance)")
    plt.tight_layout()
    plt.show()
