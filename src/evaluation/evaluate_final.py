
from pathlib import Path
import pandas as pd

from src.models.popularity import PopularityRecommender
from src.models.category_based import CategoryBasedRecommender

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"

K = 10


def evaluate_model(model, model_name, train, test):

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

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id, relevant_products in user_relevant.items():

        recommendations = model.recommend(
            user_id, train, top_n=K
        )

        recommended_ids = [
            item["product_id"]
            for item in recommendations
        ]

        precision_scores.append(
            precision_at_k(recommended_ids, relevant_products, K)
        )

        recall_scores.append(
            recall_at_k(recommended_ids, relevant_products, K)
        )

        ndcg_scores.append(
            ndcg_at_k(recommended_ids, relevant_products, K)
        )

    if not precision_scores:
        raise ValueError(
            f"No users with relevant test interactions for {model_name}"
        )

    return {
        "Model": model_name,
        "Precision@10": sum(precision_scores) / len(precision_scores),
        "Recall@10": sum(recall_scores) / len(recall_scores),
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores),
        "Evaluated users": len(user_relevant)
    }


def main():

    # Use training + validation data to fit final models.
    train = pd.read_csv(DATA_DIR / "train_validation.csv")

    # Use only warm-start interactions from the untouched test period.
    test = pd.read_csv(DATA_DIR / "final_warm_test.csv")

    products = pd.read_csv(DATA_DIR / "products_enriched.csv")

    results = []

    # 1. Final selected model
    category_model = CategoryBasedRecommender()
    category_model.fit(products, train)

    results.append(
        evaluate_model(
            category_model,
            "Category-Based Filtering",
            train,
            test
        )
    )

    # 2. Popularity baseline
    popularity_model = PopularityRecommender()
    popularity_model.fit(train)

    results.append(
        evaluate_model(
            popularity_model,
            "Popularity",
            train,
            test
        )
    )

    comparison = pd.DataFrame(results)

    print("\n========== FINAL HELD-OUT TEST RESULTS ==========")
    print(comparison.to_string(index=False))

    Path("reports").mkdir(exist_ok=True)

    output_path = Path("reports/final_test_results.csv")
    comparison.to_csv(output_path, index=False)

    print(f"\nFinal test results saved to: {output_path}")
    print(f"Training + validation interactions: {len(train)}")
    print(f"Warm-start test interactions: {len(test)}")


if __name__ == "__main__":
    main()
