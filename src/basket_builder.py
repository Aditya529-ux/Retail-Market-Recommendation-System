import os
import pandas as pd


def build_cluster_baskets(
    transaction_data,
    cluster_column="Cluster",
    output_folder="data/baskets"
):
    print("\n==========================================")
    print("BUILDING CLUSTER-SPECIFIC BASKETS")
    print("==========================================")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Check required columns
    required_columns = [
        "InvoiceNo",
        "StockCode",
        cluster_column
    ]

    for column in required_columns:
        if column not in transaction_data.columns:
            raise ValueError(
                f"Column '{column}' not found in transaction data."
            )

    basket_files = []

    # Get cluster numbers
    clusters = sorted(
        transaction_data[cluster_column].dropna().unique()
    )

    # ==========================================
    # BUILD BASKET FOR EACH CLUSTER
    # ==========================================

    for cluster in clusters:

        print(
            f"\nBuilding basket for Cluster {int(cluster)}..."
        )

        # Select data for current cluster
        cluster_data = transaction_data[
            transaction_data[cluster_column] == cluster
        ].copy()

        # Create Invoice x Product matrix
        basket = pd.crosstab(
            cluster_data["InvoiceNo"],
            cluster_data["StockCode"]
        )

        # Convert quantity/count into binary values
        # 1 = product exists in transaction
        # 0 = product does not exist
        basket = (basket > 0).astype(int)

        # Move InvoiceNo from index to column
        basket.reset_index(inplace=True)

        # Output file
        output_file = os.path.join(
            output_folder,
            f"cluster_{int(cluster)}_basket.csv"
        )

        # Save basket
        basket.to_csv(
            output_file,
            index=False
        )

        basket_files.append(output_file)

        print(
            f"Cluster {int(cluster)} basket created successfully!"
        )

        print(
            f"Transactions: {len(basket)}"
        )

        print(
            f"Products: {basket.shape[1] - 1}"
        )

        print(
            f"Saved to: {output_file}"
        )

    print("\n==========================================")
    print("ALL CLUSTER BASKETS CREATED")
    print("==========================================")

    return basket_files
