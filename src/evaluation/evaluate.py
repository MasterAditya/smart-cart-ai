
from pathlib import Path
import pandas as pd

from src.models.popularity import PopularityRecommender
from src.models.collaborative import ItemBasedCollaborativeFiltering
from src.models.content_based import ContentBasedRecommender
from src.models.category_based import CategoryBasedRecommender

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)

RAW_DIR = Path("data/raw")
DATA_DIR = Path("data/processed")
K = 10


def evaluate_model(model, model_name, train, test):

    # Cart and purchase are treated as relevant outcomes
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

        if model_name == "Popularity":
            recommendations = model.recommend(
                user_id, train, top_n=K
            )

        elif model_name == "Collaborative Filtering":
            recommendations = model.recommend(
                user_id, top_n=K
            )

        else:
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

    return {
        "Model": model_name,
        "Precision@10": sum(precision_scores) / len(precision_scores),
        "Recall@10": sum(recall_scores) / len(recall_scores),
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores),
        "Evaluated users": len(user_relevant)
    }


def main():

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "validation_warm_test.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")

    results = []

    # 1. Popularity baseline
    popularity = PopularityRecommender()
    popularity.fit(train)

    results.append(
        evaluate_model(
            popularity, "Popularity", train, test
        )
    )

    # 2. Collaborative filtering
    collaborative = ItemBasedCollaborativeFiltering(
        top_k_similar=50
    )
    collaborative.fit(train)

    results.append(
        evaluate_model(
            collaborative,
            "Collaborative Filtering",
            train,
            test
        )
    )

    # 3. Content-based filtering
    content = ContentBasedRecommender()
    content.fit(products, train)

    results.append(
        evaluate_model(
            content,
            "Content-Based Filtering",
            train,
            test
        )
    )

    # 4. Category-aware recommendation
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

    comparison = pd.DataFrame(results)

    print("\n========== FOUR-MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    Path("reports").mkdir(exist_ok=True)

    comparison.to_csv(
        "reports/validation_model_comparison.csv",
        index=False
    )

    print("\nResults saved to reports/model_comparison.csv")


if __name__ == "__main__":
    main()
