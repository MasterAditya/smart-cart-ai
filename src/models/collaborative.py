
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCollaborativeFiltering:

    def __init__(self, top_k_similar=50):
        self.top_k_similar = top_k_similar
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_index = None
        self.item_index = None
        self.item_ids = None

    def fit(self, interactions: pd.DataFrame):
        """
        Build a weighted user-item matrix and calculate
        item-item cosine similarity using training data only.
        """

        data = interactions.copy()

        # Aggregate repeated user-product interactions
        aggregated = (
            data.groupby(["user_id", "product_id"])
            ["interaction_weight"]
            .sum()
            .reset_index()
        )

        users = sorted(aggregated["user_id"].unique())
        items = sorted(aggregated["product_id"].unique())

        self.user_index = {
            user: idx for idx, user in enumerate(users)
        }

        self.item_index = {
            item: idx for idx, item in enumerate(items)
        }

        self.item_ids = items

        rows = aggregated["user_id"].map(self.user_index)
        cols = aggregated["product_id"].map(self.item_index)

        self.user_item_matrix = csr_matrix(
            (
                aggregated["interaction_weight"].values,
                (rows, cols)
            ),
            shape=(len(users), len(items))
        )

        # Item-item cosine similarity
        self.item_similarity = cosine_similarity(
            self.user_item_matrix.T,
            dense_output=True
        )

        # Exclude each item's similarity with itself
        np.fill_diagonal(self.item_similarity, 0)

        return self

    def recommend(self, user_id, top_n=10):
        if self.user_item_matrix is None:
            raise ValueError("Model must be fitted first.")

        if user_id not in self.user_index:
            return []

        user_vector = self.user_item_matrix.getrow(
            self.user_index[user_id]
        )

        # Predict scores using user history and item similarity
        scores = user_vector.dot(self.item_similarity).ravel()

        # Remove products the user has already interacted with
        seen_indices = user_vector.indices
        scores[seen_indices] = -np.inf

        # Get top recommendations
        candidate_count = min(top_n, len(scores))

        top_indices = np.argsort(scores)[::-1][:candidate_count]

        recommendations = []

        for idx in top_indices:
            if not np.isfinite(scores[idx]) or scores[idx] <= 0:
                continue

            recommendations.append({
                "product_id": self.item_ids[idx],
                "score": float(scores[idx])
            })

        return recommendations
