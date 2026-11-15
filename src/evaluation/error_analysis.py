
from pathlib import Path
import pandas as pd

from src.models.category_based import CategoryBasedRecommender
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports"

K = 10


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train_validation.csv")
    test = pd.read_csv(DATA_DIR / "final_warm_test.csv")
    products = pd.read_csv(DATA_DIR / "products_enriched.csv")
    cold_test = pd.read_csv(DATA_DIR / "cold_product_test.csv")

    # Fit the selected model using development data only.
    model = CategoryBasedRecommender()
    model.fit(products, train)

    # Relevant outcomes: add-to-cart and purchase.
    relevant_test = test[
        test["interaction_type"].isin(
            ["add_to_cart", "purchase"]
        )
    ]

    user_relevant = (
        relevant_test.groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    product_categories = products.set_index(
        "product_id"
    )["category"].to_dict()

    user_results = []
    all_recommended = []
    category_counts = {}

    for user_id, relevant_products in user_relevant.items():

        recommendations = model.recommend(
            user_id, train, top_n=K
        )

        recommended_ids = [
            item["product_id"]
            for item in recommendations
        ]

        hits = set(recommended_ids) & relevant_products

        precision = precision_at_k(
            recommended_ids, relevant_products, K
        )
        recall = recall_at_k(
            recommended_ids, relevant_products, K
        )
        ndcg = ndcg_at_k(
            recommended_ids, relevant_products, K
        )

        user_results.append({
            "user_id": user_id,
            "relevant_products": len(relevant_products),
            "recommended_count": len(recommended_ids),
            "hits": len(hits),
            "Precision@10": precision,
            "Recall@10": recall,
            "NDCG@10": ndcg,
            "zero_hit": len(hits) == 0
        })

        all_recommended.extend(recommended_ids)

        for product_id in recommended_ids:
            category = product_categories.get(
                product_id, "Unknown"
            )
            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

    user_df = pd.DataFrame(user_results)

    # Catalogue coverage.
    unique_recommended = set(all_recommended)
    unique_catalogue = set(products["product_id"])

    eligible_products = set(
        train["product_id"]
    )

    coverage_catalogue = (
        len(unique_recommended) / len(unique_catalogue)
        if unique_catalogue else 0
    )

    coverage_eligible = (
        len(unique_recommended) / len(eligible_products)
        if eligible_products else 0
    )

    # Recommendation category distribution.
    category_df = pd.DataFrame(
        [
            {
                "Category": category,
                "Recommendation count": count,
                "Share (%)": count / len(all_recommended) * 100
            }
            for category, count in category_counts.items()
        ]
    ).sort_values(
        "Recommendation count",
        ascending=False
    )

    # Cold-product test diagnostics.
    cold_relevant = cold_test[
        cold_test["interaction_type"].isin(
            ["add_to_cart", "purchase"]
        )
    ]

    summary = {
        "Evaluated users": len(user_df),
        "Users with zero relevant recommendations": int(
            user_df["zero_hit"].sum()
        ),
        "Users with at least one relevant recommendation": int(
            (~user_df["zero_hit"]).sum()
        ),
        "Average Precision@10": user_df["Precision@10"].mean(),
        "Average Recall@10": user_df["Recall@10"].mean(),
        "Average NDCG@10": user_df["NDCG@10"].mean(),
        "Unique recommended products": len(unique_recommended),
        "Total catalogue products": len(unique_catalogue),
        "Eligible products in development data": len(eligible_products),
        "Catalogue coverage (%)": coverage_catalogue * 100,
        "Eligible-product coverage (%)": coverage_eligible * 100,
        "Cold-product test interactions": len(cold_test),
        "Cold-product relevant interactions": len(cold_relevant),
        "Cold-product relevant products": cold_relevant["product_id"].nunique()
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Measure", "Value"]
    )

    # Save diagnostic reports.
    user_df.to_csv(
        REPORT_DIR / "error_analysis_users.csv",
        index=False
    )

    category_df.to_csv(
        REPORT_DIR / "recommendation_category_distribution.csv",
        index=False
    )

    summary_df.to_csv(
        REPORT_DIR / "error_analysis_summary.csv",
        index=False
    )

    print("\n========== ERROR ANALYSIS SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\n========== RECOMMENDATION CATEGORY DISTRIBUTION ==========")
    print(category_df.to_string(index=False))

    print("\n========== FIVE LOWEST NDCG USERS ==========")
    print(
        user_df.sort_values("NDCG@10")
        .head(5)
        .to_string(index=False)
    )

    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
