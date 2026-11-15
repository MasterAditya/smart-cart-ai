from pathlib import Path

import joblib
import pandas as pd

from src.models.category_based import CategoryBasedRecommender
from src.models.popularity import PopularityRecommender


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACT_DIR = ROOT_DIR / "artifacts"


def train_and_save():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(DATA_DIR / "products_enriched.csv")
    train = pd.read_csv(DATA_DIR / "train_validation.csv")

    category_model = CategoryBasedRecommender()
    category_model.fit(products, train)

    popularity_model = PopularityRecommender()
    popularity_model.fit(train)

    artifacts = {
        "category_model": category_model,
        "popularity_model": popularity_model,
        "train_interactions": train,
        "products": products,
    }

    artifact_path = ARTIFACT_DIR / "recommender.joblib"
    joblib.dump(artifacts, artifact_path)

    print(f"Models saved successfully: {artifact_path}")
    print(f"Training + validation interactions: {len(train)}")
    print(f"Products: {len(products)}")


if __name__ == "__main__":
    train_and_save()