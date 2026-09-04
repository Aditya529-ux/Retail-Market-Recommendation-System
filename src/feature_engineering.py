import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_transaction_features(df):

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Calculate total amount for each product record
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # Create transaction-level features
    transaction_features = df.groupby("InvoiceNo").agg(
        total_quantity=("Quantity", "sum"),
        total_spending=("TotalAmount", "sum"),
        unique_products=("StockCode", "nunique"),
        average_unit_price=("UnitPrice", "mean"),
        transaction_items=("StockCode", "count")
    ).reset_index()

    return transaction_features


def analyze_transaction_features(transaction_features):

    print("\nTransaction feature statistics:")
    print(transaction_features.describe())

    print("\nMissing values:")
    print(transaction_features.isnull().sum())


def prepare_clustering_features(transaction_features):

    # Features used for K-Means
    feature_columns = [
        "total_quantity",
        "total_spending",
        "unique_products",
        "average_unit_price",
        "transaction_items"
    ]

    # Select only numerical features
    clustering_data = transaction_features[feature_columns].copy()

    # Reduce the effect of extreme values
    clustering_data = np.log1p(clustering_data)

    # Scale all features
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(clustering_data)

    return scaled_features, scaler
