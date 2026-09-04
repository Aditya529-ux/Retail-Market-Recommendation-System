from src.apriori import run_apriori_for_cluster


print("\n==========================================")
print("APRlORI FOR CLUSTER 0")
print("==========================================")

run_apriori_for_cluster(
    cluster_number=0,
    basket_file="data/baskets/cluster_0_basket.csv",
    min_support=0.01,
    min_confidence=0.3
)


print("\n==========================================")
print("APRlORI FOR CLUSTER 1")
print("==========================================")

run_apriori_for_cluster(
    cluster_number=1,
    basket_file="data/baskets/cluster_1_basket.csv",
    min_support=0.005,
    min_confidence=0.3
)


print("\n==========================================")
print("APRIORI ANALYSIS COMPLETED")
print("==========================================")
