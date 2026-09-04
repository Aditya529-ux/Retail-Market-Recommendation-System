from src.recommendation import recommend_for_cluster


print("\n==========================================")
print("TESTING RECOMMENDATION SYSTEM")
print("==========================================")


# ==========================================================
# TEST CLUSTER 0
# ==========================================================

recommend_for_cluster(
    cluster_number=0,
    customer_products={
        "22917",
        "22916"
    },
    top_n=5
)


# ==========================================================
# TEST CLUSTER 1
# ==========================================================

recommend_for_cluster(
    cluster_number=1,
    customer_products={
        "22698"
    },
    top_n=5
)


print("\n==========================================")
print("RECOMMENDATION TEST COMPLETED")
print("==========================================")
