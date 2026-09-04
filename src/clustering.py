import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# FIND OPTIMAL NUMBER OF CLUSTERS
# ==========================================

def find_optimal_clusters(scaled_features):

    # K values to test
    k_values = range(2, 9)

    inertia_values = []
    silhouette_values = []

    print("\nFinding optimal number of clusters...")
    print("\nTesting different numbers of clusters...")
    print("-" * 50)

    for k in k_values:

        # Create K-Means model
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        # Train model and get cluster labels
        cluster_labels = kmeans.fit_predict(
            scaled_features
        )

        # WCSS / Inertia
        inertia = kmeans.inertia_

        # Silhouette Score
        silhouette = silhouette_score(
            scaled_features,
            cluster_labels
        )

        # Store results
        inertia_values.append(inertia)
        silhouette_values.append(silhouette)

        print(
            f"K = {k} | "
            f"WCSS = {inertia:.2f} | "
            f"Silhouette Score = {silhouette:.4f}"
        )

    # ==========================================
    # ELBOW METHOD GRAPH
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        inertia_values,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS / Inertia")
    plt.title("Elbow Method for Optimal K")

    plt.xticks(list(k_values))
    plt.grid(True)

    plt.show()

    # ==========================================
    # SILHOUETTE SCORE GRAPH
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        silhouette_values,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Different K Values")

    plt.xticks(list(k_values))
    plt.grid(True)

    plt.show()

    return (
        list(k_values),
        inertia_values,
        silhouette_values
    )


# ==========================================
# APPLY FINAL K-MEANS
# ==========================================

def apply_kmeans(
    scaled_features,
    n_clusters=2
):

    # Create final K-Means model
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    # Assign every transaction to a cluster
    cluster_labels = kmeans.fit_predict(
        scaled_features
    )

    print("\nFinal K-Means clustering completed!")
    print(
        "Number of clusters:",
        n_clusters
    )

    return kmeans, cluster_labels
