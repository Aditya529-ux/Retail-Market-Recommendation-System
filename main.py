from src.preprocessing import load_data

from src.feature_engineering import (
    create_transaction_features,
    prepare_clustering_features
)

from src.clustering import (
    find_optimal_clusters,
    apply_kmeans
)

from src.basket_builder import (
    build_cluster_baskets
)


# ==========================================
# PATHS
# ==========================================

DATA_PATH = "data/online_retail_cleaned.csv"

BASKET_FOLDER = "data/baskets"

CLUSTERED_DATA_PATH = "data/transactions_clustered.csv"


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    # ==========================================
    # STEP 1: LOAD DATASET
    # ==========================================

    print("\n==========================================")
    print("STEP 1: LOADING CLEANED DATASET")
    print("==========================================")

    df = load_data(DATA_PATH)

    print("\nDataset loaded successfully!")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())


    # ==========================================
    # STEP 2: TRANSACTION FEATURE ENGINEERING
    # ==========================================

    print("\n==========================================")
    print("STEP 2: TRANSACTION FEATURE ENGINEERING")
    print("==========================================")

    transaction_features = create_transaction_features(df)

    print("\nTransaction-level features:")
    print(transaction_features.head())

    print(
        f"\nNumber of transactions: "
        f"{len(transaction_features)}"
    )

    print("\nFeature columns:")
    print(transaction_features.columns.tolist())

    print("\nTransaction feature statistics:")
    print(transaction_features.describe())

    print("\nMissing values:")
    print(transaction_features.isnull().sum())


    # ==========================================
    # STEP 3: PREPARE CLUSTERING FEATURES
    # ==========================================

    print("\n==========================================")
    print("STEP 3: PREPARING CLUSTERING FEATURES")
    print("==========================================")

    result = prepare_clustering_features(
        transaction_features
    )

    # Some versions of the function return
    # more than one value.
    if isinstance(result, tuple):

        scaled_features = result[0]

    else:

        scaled_features = result

    print(
        "\nClustering features prepared successfully!"
    )

    print(
        f"Feature matrix shape: "
        f"{scaled_features.shape}"
    )


    # ==========================================
    # STEP 4: FIND OPTIMAL CLUSTERS
    # ==========================================

    print("\n==========================================")
    print("STEP 4: FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("==========================================")

    (
        k_values,
        inertia_values,
        silhouette_values
    ) = find_optimal_clusters(
        scaled_features
    )

    print("\nClustering evaluation results:")

    for k, inertia, silhouette in zip(
        k_values,
        inertia_values,
        silhouette_values
    ):

        print(
            f"K = {k} | "
            f"WCSS = {inertia:.2f} | "
            f"Silhouette Score = {silhouette:.4f}"
        )


    # ==========================================
    # STEP 5: SELECT BEST K
    # ==========================================

    best_index = silhouette_values.index(
        max(silhouette_values)
    )

    best_k = k_values[best_index]

    print(
        f"\nSelected number of clusters: {best_k}"
    )


    # ==========================================
    # STEP 6: APPLY FINAL K-MEANS
    # ==========================================

    print("\n==========================================")
    print("STEP 6: APPLYING FINAL K-MEANS")
    print("==========================================")

    kmeans, cluster_labels = apply_kmeans(
        scaled_features,
        n_clusters=best_k
    )

    print(
        "\nFinal K-Means clustering completed!"
    )

    print(
        f"Number of clusters: {best_k}"
    )


    # ==========================================
    # STEP 7: ADD CLUSTER LABELS
    # ==========================================

    print("\n==========================================")
    print("STEP 7: ADDING CLUSTER LABELS")
    print("==========================================")

    transaction_features["Cluster"] = cluster_labels

    print(
        "\nCluster labels successfully added!"
    )

    print(
        f"Transactions: "
        f"{len(transaction_features)}"
    )

    print("\nCluster distribution:")

    print(
        transaction_features["Cluster"]
        .value_counts()
        .sort_index()
    )


    # ==========================================
    # STEP 8: CLUSTER CHARACTERISTICS
    # ==========================================

    print("\n==========================================")
    print("STEP 8: CLUSTER CHARACTERISTICS")
    print("==========================================")

    cluster_summary = (
        transaction_features
        .groupby("Cluster")
        .mean(numeric_only=True)
    )

    print("\nCluster characteristics:")

    print(cluster_summary)


    # ==========================================
    # STEP 9: SAMPLE TRANSACTIONS
    # ==========================================

    print("\n==========================================")
    print("STEP 9: SAMPLE TRANSACTIONS")
    print("==========================================")

    print(
        transaction_features.head(10)
    )


    # ==========================================
    # STEP 10: ADD CLUSTERS TO ORIGINAL DATA
    # ==========================================

    print("\n==========================================")
    print("STEP 10: ADDING CLUSTER LABELS TO DATA")
    print("==========================================")

    # Create InvoiceNo -> Cluster mapping
    invoice_cluster_map = (
        transaction_features[
            ["InvoiceNo", "Cluster"]
        ]
        .drop_duplicates(
            subset=["InvoiceNo"]
        )
    )

    # Remove existing Cluster column
    # if it already exists
    if "Cluster" in df.columns:

        df = df.drop(
            columns=["Cluster"]
        )

    # Merge cluster labels with original data
    clustered_data = df.merge(
        invoice_cluster_map,
        on="InvoiceNo",
        how="left"
    )

    print(
        "\nCluster labels successfully added!"
    )

    print(
        f"Original rows: {len(df)}"
    )

    print(
        f"Rows after adding clusters: "
        f"{len(clustered_data)}"
    )

    print("\nSample data with cluster labels:")

    print(
        clustered_data.head(10)
    )


    # ==========================================
    # STEP 11: SAVE CLUSTERED DATA
    # ==========================================

    print("\n==========================================")
    print("STEP 11: SAVING CLUSTERED DATA")
    print("==========================================")

    clustered_data.to_csv(
        CLUSTERED_DATA_PATH,
        index=False
    )

    print(
        "\nClustered data saved successfully!"
    )

    print(
        f"Saved to: {CLUSTERED_DATA_PATH}"
    )


    # ==========================================
    # STEP 12: BUILD CLUSTER BASKETS
    # ==========================================

    print("\n==========================================")
    print("STEP 12: BUILDING CLUSTER BASKETS")
    print("==========================================")

    basket_files = build_cluster_baskets(
        transaction_data=clustered_data,
        cluster_column="Cluster",
        output_folder=BASKET_FOLDER
    )


    # ==========================================
    # STEP 13: FINAL RESULTS
    # ==========================================

    print("\n==========================================")
    print("CLUSTERING + BASKET BUILDING COMPLETED")
    print("==========================================")

    print(
        f"Final number of clusters: {best_k}"
    )

    print("\nCluster transaction counts:")

    cluster_counts = (
        transaction_features["Cluster"]
        .value_counts()
        .sort_index()
    )

    for cluster, count in cluster_counts.items():

        print(
            f"Cluster {int(cluster)} transactions: "
            f"{count}"
        )

    print("\nBasket files created:")

    for file in basket_files:

        print(
            f"  {file}"
        )

    print(
        "\nProject pipeline completed successfully!"
    )


# ==========================================
# PROGRAM START
# ==========================================

if __name__ == "__main__":
    main()
