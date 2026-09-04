from src.rule_evaluation import run_rule_evaluation


# ==========================================================
# RULE EVALUATION FOR CLUSTER 0
# ==========================================================

print("\n==========================================")
print("RULE EVALUATION FOR CLUSTER 0")
print("==========================================")

run_rule_evaluation(
    rules_file="data/rules/cluster_0_association_rules.csv",
    output_file="data/rules/cluster_0_evaluated_rules.csv",
    min_confidence=0.3,
    min_lift=1.0
)


# ==========================================================
# RULE EVALUATION FOR CLUSTER 1
# ==========================================================

print("\n==========================================")
print("RULE EVALUATION FOR CLUSTER 1")
print("==========================================")

run_rule_evaluation(
    rules_file="data/rules/cluster_1_association_rules.csv",
    output_file="data/rules/cluster_1_evaluated_rules.csv",
    min_confidence=0.3,
    min_lift=1.0
)


# ==========================================================
# COMPLETED
# ==========================================================

print("\n==========================================")
print("RULE EVALUATION COMPLETED")
print("==========================================")
