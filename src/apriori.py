import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules


def load_basket(file_path):
    """
    Load a cluster basket CSV file.
    """

    print(f"\nLoading basket: {file_path}")

    basket = pd.read_csv(file_path)

    print(f"Basket shape: {basket.shape}")

    return basket


def generate_frequent_itemsets(
    basket,
    min_support=0.01
):
    """
    Generate frequent itemsets using Apriori.
    """

    print("\nRunning Apriori...")
    print(f"Minimum support: {min_support}")

    # Remove transaction ID column if present
    columns_to_remove = []

    for column in basket.columns:
        if column.lower() in ["invoiceno", "invoice_no"]:
            columns_to_remove.append(column)

    if columns_to_remove:
        basket = basket.drop(columns=columns_to_remove)

    # Convert values to boolean
    basket = basket.astype(bool)

    # Generate frequent itemsets
    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    # Number of items in each itemset
    frequent_itemsets["itemset_size"] = (
        frequent_itemsets["itemsets"].apply(len)
    )

    print(
        f"Frequent itemsets found: "
        f"{len(frequent_itemsets)}"
    )

    return frequent_itemsets


def generate_association_rules(
    frequent_itemsets,
    min_confidence=0.3
):
    """
    Generate association rules.
    """

    print("\nGenerating association rules...")
    print(f"Minimum confidence: {min_confidence}")

    if frequent_itemsets.empty:
        print("No frequent itemsets found.")
        return pd.DataFrame()

    # At least two items are required for rules
    if not any(
        frequent_itemsets["itemset_size"] >= 2
    ):
        print("No itemsets with 2 or more products found.")
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        print("No association rules found.")
        return rules

    # Calculate useful values
    rules["antecedent_len"] = (
        rules["antecedents"].apply(len)
    )

    rules["consequent_len"] = (
        rules["consequents"].apply(len)
    )

    # Sort by confidence and lift
    rules = rules.sort_values(
        by=["confidence", "lift"],
        ascending=False
    )

    print(
        f"Association rules found: "
        f"{len(rules)}"
    )

    return rules


def save_results(
    frequent_itemsets,
    rules,
    cluster_number,
    output_folder="data/rules"
):
    """
    Save Apriori results for a cluster.
    """

    os.makedirs(output_folder, exist_ok=True)

    itemsets_file = os.path.join(
        output_folder,
        f"cluster_{cluster_number}_frequent_itemsets.csv"
    )

    rules_file = os.path.join(
        output_folder,
        f"cluster_{cluster_number}_association_rules.csv"
    )

    # Convert frozensets to readable text
    itemsets_to_save = frequent_itemsets.copy()

    if not itemsets_to_save.empty:
        itemsets_to_save["itemsets"] = (
            itemsets_to_save["itemsets"]
            .apply(lambda x: ", ".join(map(str, x)))
        )

    itemsets_to_save.to_csv(
        itemsets_file,
        index=False
    )

    # Save rules
    rules_to_save = rules.copy()

    if not rules_to_save.empty:

        rules_to_save["antecedents"] = (
            rules_to_save["antecedents"]
            .apply(lambda x: ", ".join(map(str, x)))
        )

        rules_to_save["consequents"] = (
            rules_to_save["consequents"]
            .apply(lambda x: ", ".join(map(str, x)))
        )

    rules_to_save.to_csv(
        rules_file,
        index=False
    )

    print("\nResults saved:")
    print(f"Frequent itemsets: {itemsets_file}")
    print(f"Association rules: {rules_file}")

    return itemsets_file, rules_file


def run_apriori_for_cluster(
    cluster_number,
    basket_file,
    min_support=0.01,
    min_confidence=0.3
):
    """
    Complete Apriori pipeline for one cluster.
    """

    print("\n==========================================")
    print(
        f"APRlORI FOR CLUSTER {cluster_number}"
    )
    print("==========================================")

    # Load basket
    basket = load_basket(basket_file)

    # Generate frequent itemsets
    frequent_itemsets = generate_frequent_itemsets(
        basket,
        min_support=min_support
    )

    # Generate association rules
    rules = generate_association_rules(
        frequent_itemsets,
        min_confidence=min_confidence
    )

    # Save results
    itemsets_file, rules_file = save_results(
        frequent_itemsets,
        rules,
        cluster_number
    )

    # Display top results
    print("\nTop frequent itemsets:")

    if not frequent_itemsets.empty:

        print(
            frequent_itemsets
            .sort_values(
                by="support",
                ascending=False
            )
            .head(10)
        )

    print("\nTop association rules:")

    if not rules.empty:

        print(
            rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].head(10)
        )

    return (
        frequent_itemsets,
        rules,
        itemsets_file,
        rules_file
    )
