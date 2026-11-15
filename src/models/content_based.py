
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix


class ContentBasedRecommender:

    def __init__(self):
        self.products = None
        self.product_ids = None
        self.product_index = None
        self.product_features = None
        self.user_profiles = {}

    def fit(self, products, interactions):
        self.products = products.reset_index(drop=True).copy()

        self.product_ids = self.products["product_id"].tolist()

        self.product_index = {
            pid: idx for idx, pid in enumerate(self.product_ids)
        }

        # Text features
        text = (
            self.products["description"].fillna("")
            + " "
            + self.products["product_name"].fillna("")
        )

        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=1000
        )

        text_features = tfidf.fit_transform(text)

        # Categorical features
        encoder = OneHotEncoder(handle_unknown="ignore")

        categorical_features = encoder.fit_transform(
            self.products[["category", "brand"]]
        )

        # Numerical features
        scaler = StandardScaler()

        numerical_features = scaler.fit_transform(
            self.products[["price", "average_rating"]]
        )

        numerical_features = csr_matrix(numerical_features)

        # Combined product representation
        self.product_features = hstack([
            text_features,
            categorical_features,
            numerical_features
        ]).tocsr()

        # Build user preference profiles from training data
        weighted = interactions.copy()

        weighted["product_index"] = weighted["product_id"].map(
            self.product_index
        )

        weighted = weighted.dropna(subset=["product_index"])

        weighted["product_index"] = (
            weighted["product_index"].astype(int)
        )

        for user_id, group in weighted.groupby("user_id"):

            indices = group["product_index"].to_numpy()

            weights = group["interaction_weight"].to_numpy(
                dtype=float
            )

            profile = self.product_features[indices].multiply(
                weights[:, None]
            ).sum(axis=0)

            profile = np.asarray(profile).ravel()

            norm = np.linalg.norm(profile)

            if norm > 0:
                profile = profile / norm

            self.user_profiles[user_id] = profile

        return self

    def recommend(self, user_id, interactions, top_n=10):

        if user_id not in self.user_profiles:
            return []

        profile = self.user_profiles[user_id]

        scores = cosine_similarity(
            profile.reshape(1, -1),
            self.product_features
        ).ravel()

        # Exclude products already seen in training
        seen = set(
            interactions.loc[
                interactions["user_id"] == user_id,
                "product_id"
            ]
        )

        for product_id in seen:
            idx = self.product_index.get(product_id)
            if idx is not None:
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1]

        recommendations = []

        for idx in top_indices:
            if not np.isfinite(scores[idx]):
                continue

            recommendations.append({
                "product_id": self.product_ids[idx],
                "score": float(scores[idx])
            })

            if len(recommendations) == top_n:
                break

        return recommendations
