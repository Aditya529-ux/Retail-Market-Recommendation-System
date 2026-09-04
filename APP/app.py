import streamlit as st
import pandas as pd
from pathlib import Path
import ast


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

TRANSACTIONS_FILE = DATA_DIR / "transactions_clustered.csv"
RULES_DIR = DATA_DIR / "rules"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Market Recommendation System",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #888888;
        margin-bottom: 30px;
    }

    .recommendation-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444444;
        margin-bottom: 15px;
    }

    .product-name {
        font-size: 22px;
        font-weight: 600;
    }

    .product-code {
        color: #999999;
        font-size: 14px;
    }

    .metric {
        font-size: 15px;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRANSACTION DATA
# ============================================================

@st.cache_data
def load_transactions():

    if not TRANSACTIONS_FILE.exists():

        st.error(
            f"Transaction file not found:\n{TRANSACTIONS_FILE}"
        )

        return pd.DataFrame()

    df = pd.read_csv(
        TRANSACTIONS_FILE,
        low_memory=False
    )

    return df


transactions = load_transactions()


# ============================================================
# PRODUCT DESCRIPTION MAPPING
# ============================================================

@st.cache_data
def create_product_mapping(df):

    if df.empty:
        return {}

    required_columns = [
        "StockCode",
        "Description"
    ]

    for column in required_columns:

        if column not in df.columns:
            return {}

    product_df = df[
        ["StockCode", "Description"]
    ].dropna()

    product_df["StockCode"] = (
        product_df["StockCode"]
        .astype(str)
        .str.strip()
    )

    product_df["Description"] = (
        product_df["Description"]
        .astype(str)
        .str.strip()
    )

    # Remove duplicate StockCodes
    product_df = product_df.drop_duplicates(
        subset=["StockCode"]
    )

    mapping = dict(
        zip(
            product_df["StockCode"],
            product_df["Description"]
        )
    )

    return mapping


product_mapping = create_product_mapping(
    transactions
)


# ============================================================
# LOAD EVALUATED RULES
# ============================================================

@st.cache_data
def load_rules(cluster):

    rules_file = (
        RULES_DIR /
        f"cluster_{cluster}_evaluated_rules.csv"
    )

    if not rules_file.exists():

        return pd.DataFrame()

    try:

        rules = pd.read_csv(
            rules_file,
            low_memory=False
        )

        return rules

    except Exception:

        return pd.DataFrame()


# ============================================================
# PARSE PRODUCT SET
# ============================================================

def parse_product_set(value):

    if pd.isna(value):

        return set()

    value = str(value).strip()

    if not value:

        return set()

    # Handle strings such as:
    # frozenset({'22916', '22917'})
    if value.startswith("frozenset"):

        try:

            value = value.replace(
                "frozenset",
                ""
            )

            value = value.strip("()")

            parsed = ast.literal_eval(value)

            return {
                str(x).strip()
                for x in parsed
            }

        except Exception:

            pass

    # Normal saved format:
    # 22916, 22917

    products = value.split(",")

    return {
        str(product).strip()
        for product in products
        if str(product).strip()
    }


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    rules,
    customer_products,
    number_of_recommendations
):

    if rules.empty:

        return pd.DataFrame()

    customer_products = {
        str(product).strip()
        for product in customer_products
    }

    recommendations = []

    for _, row in rules.iterrows():

        antecedents = parse_product_set(
            row["antecedents"]
        )

        consequents = parse_product_set(
            row["consequents"]
        )

        # Rule can be used only when
        # customer has every antecedent product
        if antecedents.issubset(
            customer_products
        ):

            for product in consequents:

                # Do not recommend products
                # the customer already has
                if product in customer_products:
                    continue

                recommendations.append(
                    {
                        "product": product,
                        "support": float(
                            row.get(
                                "support",
                                0
                            )
                        ),
                        "confidence": float(
                            row.get(
                                "confidence",
                                0
                            )
                        ),
                        "lift": float(
                            row.get(
                                "lift",
                                0
                            )
                        ),
                        "rule_score": float(
                            row.get(
                                "rule_score",
                                0
                            )
                        )
                    }
                )

    if not recommendations:

        return pd.DataFrame()

    result = pd.DataFrame(
        recommendations
    )

    # If the same product is recommended
    # by multiple rules, keep the strongest rule
    result = (
        result
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
    )

    result = result.head(
        number_of_recommendations
    )

    result.insert(
        0,
        "rank",
        range(
            1,
            len(result) + 1
        )
    )

    return result


# ============================================================
# CLUSTER INFORMATION
# ============================================================

def get_cluster_information(
    df,
    cluster
):

    if df.empty:
        return 0, 0, ""

    cluster_df = df[
        df["Cluster"] == cluster
    ]

    transactions_count = (
        cluster_df["InvoiceNo"]
        .nunique()
    )

    products_count = (
        cluster_df["StockCode"]
        .nunique()
    )

    if "transaction_items" in cluster_df.columns:

        avg_items = (
            cluster_df["transaction_items"]
            .mean()
        )

    else:

        # Calculate average products
        # per invoice if feature is unavailable

        avg_items = (
            cluster_df
            .groupby("InvoiceNo")["StockCode"]
            .nunique()
            .mean()
        )

    if avg_items >= 15:

        description = (
            "Higher-volume customers with larger "
            "transactions and more products per transaction."
        )

    else:

        description = (
            "Lower-volume customers with smaller "
            "transactions and fewer products per transaction."
        )

    return (
        transactions_count,
        products_count,
        description
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛒 Retail Market Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Customer segmentation + Apriori association rules + product recommendations'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Recommendation Settings"
)

cluster = st.sidebar.selectbox(
    "Select Customer Cluster",
    options=[0, 1],
    index=1
)

number_of_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=3
)


# ============================================================
# CLUSTER INFORMATION
# ============================================================

transactions_count, products_count, cluster_description = (
    get_cluster_information(
        transactions,
        cluster
    )
)


st.subheader(
    "📊 Customer Cluster Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Cluster",
        cluster
    )

with col2:

    st.metric(
        "Transactions",
        f"{transactions_count:,}"
    )

with col3:

    st.metric(
        "Products",
        f"{products_count:,}"
    )


if cluster_description:

    st.info(
        cluster_description
    )


# ============================================================
# CUSTOMER BASKET
# ============================================================

st.subheader(
    "🛍️ Customer Basket"
)

st.write(
    "Enter product StockCodes that the customer has purchased."
)

product_input = st.text_input(
    "Product StockCodes",
    placeholder="Example: 22916, 22917"
)


generate_button = st.button(
    "🚀 Generate Recommendations",
    use_container_width=True
)


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

if generate_button:

    if not product_input.strip():

        st.warning(
            "Please enter at least one product StockCode."
        )

    else:

        # Convert input into product set

        customer_products = {
            product.strip()
            for product in product_input.split(",")
            if product.strip()
        }

        # Load rules for selected cluster

        rules = load_rules(
            cluster
        )

        if rules.empty:

            st.error(
                f"No evaluated association rules "
                f"are available for Cluster {cluster}."
            )

        else:

            recommendations = generate_recommendations(
                rules,
                customer_products,
                number_of_recommendations
            )

            # =================================================
            # RESULTS
            # =================================================

            if recommendations.empty:

                st.warning(
                    "No recommendations were found "
                    "for the entered products."
                )

                st.info(
                    "Try another StockCode or a different "
                    "combination of products."
                )

            else:

                st.success(
                    f"{len(recommendations)} "
                    "recommendations generated!"
                )

                st.subheader(
                    "🎯 Recommended Products"
                )

                # =================================================
                # DISPLAY RECOMMENDATIONS
                # =================================================

                for _, recommendation in recommendations.iterrows():

                    product_code = str(
                        recommendation["product"]
                    )

                    product_description = (
                        product_mapping.get(
                            product_code,
                            "Product description unavailable"
                        )
                    )

                    rank = int(
                        recommendation["rank"]
                    )

                    support = (
                        recommendation["support"]
                    )

                    confidence = (
                        recommendation["confidence"]
                    )

                    lift = (
                        recommendation["lift"]
                    )

                    rule_score = (
                        recommendation["rule_score"]
                    )

                    st.markdown(
                        f"""
                        <div class="recommendation-card">

                        <div class="product-name">
                        🛍️ {product_description}
                        </div>

                        <div class="product-code">
                        StockCode: {product_code}
                        </div>

                        <div class="metric">
                        <b>Rank:</b> {rank}
                        </div>

                        <div class="metric">
                        <b>Confidence:</b>
                        {confidence * 100:.2f}%
                        </div>

                        <div class="metric">
                        <b>Lift:</b>
                        {lift:.2f}
                        </div>

                        <div class="metric">
                        <b>Support:</b>
                        {support * 100:.2f}%
                        </div>

                        <div class="metric">
                        <b>Rule Score:</b>
                        {rule_score:.2f}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Retail Market Recommendation System | "
    "K-Means + Apriori + Association Rule Evaluation"
)
