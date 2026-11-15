
import pandas as pd


class PopularityRecommender:
    def __init__(self):
        self.popular_products = None

    def fit(self, interactions: pd.DataFrame):
        """
        Rank products using weighted historical engagement.
        Only training interactions should be supplied.
        """
        self.popular_products = (
            interactions.groupby("product_id")
            .agg(
                popularity_score=("interaction_weight", "sum"),
                interaction_count=("interaction_id", "count"),
                purchase_count=(
                    "interaction_type",
                    lambda x: (x == "purchase").sum()
                )
            )
            .reset_index()
            .sort_values(
                ["popularity_score", "purchase_count"],
                ascending=False
            )
            .reset_index(drop=True)
        )

        return self

    def recommend(self, user_id, train_interactions, top_n=10):
        """
        Recommend popular products the user has not interacted with.
        """
        if self.popular_products is None:
            raise ValueError("Model must be fitted before recommendation.")

        seen_products = set(
            train_interactions.loc[
                train_interactions["user_id"] == user_id,
                "product_id"
            ]
        )

        recommendations = self.popular_products[
            ~self.popular_products["product_id"].isin(seen_products)
        ].head(top_n)

        return recommendations[
            ["product_id", "popularity_score"]
        ].to_dict("records")
