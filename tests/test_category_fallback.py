
import pandas as pd

from src.models.category_based import CategoryBasedRecommender


def test_unknown_user_gets_globally_popular_products():

    products = pd.DataFrame({
        "product_id": ["P001", "P002", "P003"],
        "category": ["Electronics", "Books", "Sports"]
    })

    interactions = pd.DataFrame({
        "user_id": ["U001", "U001", "U002", "U002"],
        "product_id": ["P001", "P002", "P002", "P003"],
        "interaction_type": [
            "purchase", "view", "add_to_cart", "view"
        ],
        "interaction_weight": [5, 1, 3, 2]
    })

    model = CategoryBasedRecommender()
    model.fit(products, interactions)

    recommendations = model.recommend(
        user_id="UNKNOWN_USER",
        interactions=interactions,
        top_n=2
    )

    recommended_ids = [
        item["product_id"] for item in recommendations
    ]

    # P002 has the highest total popularity (4),
    # followed by P001 (5) and P003 (2).
    assert recommended_ids == ["P001", "P002"]
    assert len(recommendations) == 2
