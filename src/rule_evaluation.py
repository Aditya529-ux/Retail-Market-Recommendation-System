import pandas as pd
import os


def load_rules(file_path):
    """
    Load association rules from CSV file.

    Handles empty files safely.
    """

    print(f"\nLoading rules: {file_path}")

    try:
        rules = pd.read_csv(file_path)

    except pd.errors.EmptyDataError:
        print("WARNING: Rules file is empty.")
        print("No association rules are available for this cluster.")
        return pd.DataFrame()

    except FileNotFoundError:
        print("ERROR: Rules file not found.")
        return pd.DataFrame()

    print(f"Rules loaded: {len(rules)}")

    return rules


def evaluate_rules(
    rules,
    min_confidence=0.3,
    min_lift=1.0
):
    """
    Evaluate association rules using
    confidence and lift.
    """

    print("\nEvaluating association rules...")
    print(f"Minimum confidence: {min_confidence}")
    print(f"Minimum lift: {min_lift}")

    # If no rules exist
    if rules.empty:
        print("No rules available for evaluation.")
        return pd.DataFrame()

    # Check required columns
    required_columns = [
        "confidence",
        "lift"
    ]

    for column in required_columns:

        if column not in rules.columns:
            print(f"ERROR: Missing column: {column}")
            return pd.DataFrame()

    # Filter rules
    evaluated_rules = rules[
        (rules["confidence"] >= min_confidence) &
        (rules["lift"] >= min_lift)
    ].copy()

    print(
        f"High-quality rules found: "
        f"{len(evaluated_rules)}"
    )

    # If no rules satisfy conditions
    if evaluated_rules.empty:
        print("No high-quality rules found.")
        return evaluated_rules

    # Calculate antecedent length
    if "antecedents" in evaluated_rules.columns:

        evaluated_rules["antecedent_len"] = (
            evaluated_rules["antecedents"]
            .apply(
                lambda x: len(str(x).split(","))
                if str(x).strip()
                else 0
            )
        )

    # Calculate consequent length
    if "consequents" in evaluated_rules.columns:

        evaluated_rules["consequent_len"] = (
            evaluated_rules["consequents"]
            .apply(
                lambda x: len(str(x).split(","))
                if str(x).strip()
                else 0
            )
        )

    # Create rule score
    #
    # Higher:
    #   support
    #   confidence
    #   lift
    #
    # means a stronger rule.

    if "support" in evaluated_rules.columns:

        evaluated_rules["rule_score"] = (
            evaluated_rules["support"]
            * evaluated_rules["confidence"]
            * evaluated_rules["lift"]
            * 100
        )

    else:

        evaluated_rules["rule_score"] = (
            evaluated_rules["confidence"]
            * evaluated_rules["lift"]
        )

    # Sort rules
    evaluated_rules = evaluated_rules.sort_values(
        by="rule_score",
        ascending=False
    )

    return evaluated_rules


def save_evaluated_rules(
    evaluated_rules,
    output_file
):
    """
    Save evaluated rules to CSV.
    """

    # Create folder if required
    output_folder = os.path.dirname(output_file)

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    # Save empty result safely
    if evaluated_rules.empty:

        print(
            f"\nNo evaluated rules to save."
        )

        # Create an empty CSV with useful columns
        empty_columns = [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "rule_score"
        ]

        empty_df = pd.DataFrame(
            columns=empty_columns
        )

        empty_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"Empty result saved to: "
            f"{output_file}"
        )

        return

    evaluated_rules.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nEvaluated rules saved to: "
        f"{output_file}"
    )


def display_top_rules(
    evaluated_rules,
    number_of_rules=10
):
    """
    Display top evaluated rules.
    """

    print("\nTop evaluated rules:")

    if evaluated_rules.empty:

        print(
            "No evaluated rules available."
        )

        return

    # Columns to display
    display_columns = []

    possible_columns = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "rule_score"
    ]

    for column in possible_columns:

        if column in evaluated_rules.columns:
            display_columns.append(column)

    print(
        evaluated_rules[
            display_columns
        ].head(number_of_rules)
    )


def run_rule_evaluation(
    rules_file,
    output_file,
    min_confidence=0.3,
    min_lift=1.0
):
    """
    Complete rule evaluation pipeline.
    """

    # STEP 1
    rules = load_rules(
        rules_file
    )

    # STEP 2
    if rules.empty:

        print(
            "\nNo association rules available "
            "for this cluster."
        )

        # Still create an empty output file
        save_evaluated_rules(
            pd.DataFrame(),
            output_file
        )

        return pd.DataFrame()

    # STEP 3
    evaluated_rules = evaluate_rules(
        rules,
        min_confidence=min_confidence,
        min_lift=min_lift
    )

    # STEP 4
    save_evaluated_rules(
        evaluated_rules,
        output_file
    )

    # STEP 5
    display_top_rules(
        evaluated_rules,
        number_of_rules=10
    )

    return evaluated_rules
