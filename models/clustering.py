import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from utils.preprocess import load_and_prepare_data
from colorama import Fore, Style, init

init(autoreset=True)

def run_kmeans():
    print(Fore.CYAN + "\n[Clustering] Running K-Means on borough crime type profiles..." + Style.RESET_ALL)

    df = load_and_prepare_data()

    pivot = df.pivot_table(
        index='borough',
        columns='crime_type',
        values='crime_count',
        aggfunc='sum'
    ).fillna(0)

    print(Fore.YELLOW + f"[Clustering] Created borough x crime_type matrix: {pivot.shape}" + Style.RESET_ALL)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(Fore.CYAN + "\nSilhouette Scores by Cluster Count:" + Style.RESET_ALL)
    scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        print(f"{Fore.GREEN}K={k}{Style.RESET_ALL} → {Fore.MAGENTA}Silhouette Score: {score:.4f}{Style.RESET_ALL}")

    plt.figure()
    plt.plot(range(2, 11), scores, marker='o')
    plt.title("Silhouette Score vs. K (Crime Type Clustering)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_k = 3
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', s=80, edgecolor='k')

    for i, borough in enumerate(pivot.index):
        plt.text(X_pca[i, 0] + 0.1, X_pca[i, 1] + 0.05, borough, fontsize=8, alpha=0.85)

    plt.title(f"Crime-Type Cluster Map (K={best_k})", fontsize=14, fontweight='bold')
    plt.xlabel("PCA 1", fontsize=12)
    plt.ylabel("PCA 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)

    handles = [plt.Line2D([], [], marker='o', linestyle='',
                          color=scatter.cmap(scatter.norm(i)),
                          label=f'Cluster {i}') for i in range(best_k)]
    plt.legend(handles=handles, loc='upper right', title="Clusters")
    plt.tight_layout()
    plt.show()

    components_df = pd.DataFrame(
        pca.components_,
        columns=pivot.columns,
        index=['PCA 1', 'PCA 2']
    )

    print(Fore.CYAN + "Top contributing crime types to PCA axes:" + Style.RESET_ALL)
    for axis in components_df.index:
        print(Fore.YELLOW + f"\n{axis} is influenced most by:" + Style.RESET_ALL)
        top_features = components_df.loc[axis].abs().sort_values(ascending=False).head(5)
        for crime_type, weight in top_features.items():
            print(f"  {Fore.GREEN}• {crime_type}{Style.RESET_ALL} (weight: {Fore.MAGENTA}{weight:.3f}{Style.RESET_ALL})")

def estimate_eps(X_scaled, min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, _ = neighbors_fit.kneighbors(X_scaled)
    k_distances = sorted(distances[:, -1])

    plt.figure(figsize=(8, 4))
    plt.plot(k_distances)
    plt.title(f"k-Distance Graph (k={min_samples})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {min_samples}th Nearest Neighbor")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(Fore.YELLOW + "\nCheck the plot above: look for the point where the curve has an 'elbow' — that's your optimal eps." + Style.RESET_ALL)

def run_dbscan():
    print(Fore.CYAN + "\n[Clustering] Running DBSCAN on borough crime type profiles..." + Style.RESET_ALL)

    df = load_and_prepare_data()
    pivot = df.pivot_table(
        index='borough',
        columns='crime_type',
        values='crime_count',
        aggfunc='sum'
    ).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    min_samples = int(input("Enter min_samples value (suggest 2–5): ") or 2)
    print(Fore.CYAN + f"\n[Info] Estimating optimal eps for DBSCAN with min_samples={min_samples}..." + Style.RESET_ALL)
    estimate_eps(X_scaled, min_samples)

    eps = float(input("Enter eps value based on the elbow of the plot: ") or 1.5)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(Fore.YELLOW + f"DBSCAN identified {n_clusters} clusters" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Cluster labels: {labels}" + Style.RESET_ALL)

    if n_clusters >= 2:
        score = silhouette_score(X_scaled, labels)
        print(Fore.MAGENTA + f"Silhouette Score: {score:.4f}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "Silhouette Score: N/A (fewer than 2 clusters)" + Style.RESET_ALL)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', s=100, edgecolor='k')

    for i, borough in enumerate(pivot.index):
        plt.text(X_pca[i, 0] + 0.1, X_pca[i, 1] + 0.05, borough, fontsize=8)

    plt.title(f"DBSCAN Cluster Map (eps={eps}, min_samples={min_samples})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
