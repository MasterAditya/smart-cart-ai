
import pandas as pd


class CategoryBasedRecommender:

    def __init__(self):
        self.products = None
        self.category_popularity = None
        self.product_popularity = None
        self.user_category_preferences = {}

    def fit(self, products, interactions):

        self.products = products.copy()

        # Global product popularity from training interactions
        self.product_popularity = (
            interactions.groupby("product_id")
            .agg(
                popularity_score=("interaction_weight", "sum"),
                purchase_count=(
                    "interaction_type",
                    lambda x: (x == "purchase").sum()
                )
            )
            .reset_index()
        )

        # Attach category information
        product_data = self.product_popularity.merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left"
        )

        self.category_popularity = (
            product_data.groupby("category")["popularity_score"]
            .sum()
            .sort_values(ascending=False)
        )

        # Learn individual category preferences
        user_data = interactions.merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left"
        )

        preferences = (
            user_data.groupby(["user_id", "category"])
            ["interaction_weight"]
            .sum()
            .reset_index()
        )

        for user_id, group in preferences.groupby("user_id"):

            group = group.sort_values(
                "interaction_weight",
                ascending=False
            )

            self.user_category_preferences[user_id] = (
                group["category"].tolist()
            )

        return self

    def recommend(self, user_id, interactions, top_n=10):

        if user_id not in self.user_category_preferences:
            return self._fallback_recommendations(
                interactions, top_n
            )

        preferred_categories = (
            self.user_category_preferences[user_id]
        )

        seen_products = set(
            interactions.loc[
                interactions["user_id"] == user_id,
                "product_id"
            ]
        )

        candidates = self.product_popularity.merge(
            self.products[["product_id", "category"]],
            on="product_id",
            how="left"
        )

        candidates = candidates[
            ~candidates["product_id"].isin(seen_products)
        ].copy()

        # Rank preferred categories first, then product popularity
        category_rank = {
            category: rank
            for rank, category in enumerate(preferred_categories)
        }

        candidates["category_rank"] = (
            candidates["category"].map(category_rank)
            .fillna(len(category_rank))
        )

        candidates = candidates.sort_values(
            ["category_rank", "popularity_score", "purchase_count"],
            ascending=[True, False, False]
        )

        return candidates.head(top_n)[
            ["product_id", "popularity_score", "category"]
        ].to_dict("records")
    def _fallback_recommendations(self, interactions, top_n):

        candidates = self.product_popularity.sort_values(
            ["popularity_score", "purchase_count"],
            ascending=[False, False]
        )

        return candidates.head(top_n)[
            ["product_id", "popularity_score"]
        ].to_dict("records")