import pandas as pd
import os


def load_evaluated_rules(file_path):
    """
    Load evaluated association rules.
    """

    print(f"\nLoading evaluated rules: {file_path}")

    if not os.path.exists(file_path):
        print("WARNING: Rules file does not exist.")
        return pd.DataFrame()

    try:
        rules = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print("WARNING: Rules file is empty.")
        return pd.DataFrame()

    if rules.empty:
        print("WARNING: No rules available.")
        return pd.DataFrame()

    print(f"Rules loaded: {len(rules)}")

    return rules


def parse_items(value):
    """
    Convert comma-separated products into a set.
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def generate_recommendations(
    rules,
    customer_products,
    top_n=5
):
    """
    Generate product recommendations
    using evaluated association rules.
    """

    print("\nGenerating recommendations...")

    if rules.empty:
        print("No rules available.")
        return pd.DataFrame()

    # Convert customer products to set
    customer_products = {
        str(product).strip()
        for product in customer_products
    }

    print(f"Customer products: {customer_products}")

    recommendations = []

    for _, rule in rules.iterrows():

        # Products on left side of rule
        antecedents = parse_items(
            rule["antecedents"]
        )

        # Products on right side of rule
        consequents = parse_items(
            rule["consequents"]
        )

        # Rule applies only when ALL antecedents
        # are present in customer's basket
        if antecedents.issubset(customer_products):

            for product in consequents:

                # Do not recommend something
                # customer already has
                if product in customer_products:
                    continue

                recommendations.append(
                    {
                        "product": product,
                        "support": rule["support"],
                        "confidence": rule["confidence"],
                        "lift": rule["lift"],
                        "rule_score": rule["rule_score"]
                    }
                )

    if not recommendations:
        print("No recommendations found.")

        return pd.DataFrame()

    recommendations_df = pd.DataFrame(
        recommendations
    )

    # If same product is recommended
    # by multiple rules, keep the strongest rule
    recommendations_df = (
        recommendations_df
        .sort_values(
            by=[
                "rule_score",
                "confidence",
                "lift"
            ],
            ascending=False
        )
        .drop_duplicates(
            subset=["product"]
        )
        .reset_index(drop=True)
    )

    # Keep top N recommendations
    recommendations_df = (
        recommendations_df
        .head(top_n)
    )

    # Add ranking
    recommendations_df.insert(
        0,
        "rank",
        range(
            1,
            len(recommendations_df) + 1
        )
    )

    print(
        f"Recommendations found: "
        f"{len(recommendations_df)}"
    )

    return recommendations_df


def recommend_for_cluster(
    cluster_number,
    customer_products,
    top_n=5
):
    """
    Generate recommendations for
    a specific cluster.
    """

    print("\n==========================================")
    print(
        f"RECOMMENDATIONS FOR CLUSTER "
        f"{cluster_number}"
    )
    print("==========================================")

    rules_file = (
        f"data/rules/"
        f"cluster_{cluster_number}_evaluated_rules.csv"
    )

    rules = load_evaluated_rules(
        rules_file
    )

    recommendations = generate_recommendations(
        rules,
        customer_products,
        top_n=top_n
    )

    if recommendations.empty:

        print(
            "\nNo recommendations available "
            "for this customer."
        )

        return recommendations

    print("\nTop Recommendations:")

    print(
        recommendations[
            [
                "rank",
                "product",
                "support",
                "confidence",
                "lift",
                "rule_score"
            ]
        ].to_string(
            index=False
        )
    )

    return recommendations
