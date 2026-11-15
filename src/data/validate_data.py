
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")


def validate_data():
    users = pd.read_csv(DATA_DIR / "users.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")

    print("\n========== DATA QUALITY REPORT ==========")

    datasets = {
        "Users": users,
        "Products": products,
        "Interactions": interactions,
    }

    for name, df in datasets.items():
        print(f"\n{name}")
        print("Shape:", df.shape)
        print("Missing values:\n", df.isnull().sum())
        print("Duplicate rows:", df.duplicated().sum())

    print("\n========== RELATIONSHIP CHECKS ==========")

    print(
        "Unknown user IDs:",
        (~interactions["user_id"].isin(users["user_id"])).sum()
    )

    print(
        "Unknown product IDs:",
        (~interactions["product_id"].isin(products["product_id"])).sum()
    )

    print(
        "Duplicate interaction IDs:",
        interactions["interaction_id"].duplicated().sum()
    )

    print("\nInteraction types:")
    print(interactions["interaction_type"].value_counts())

    print("\nInteraction weights:")
    print(interactions.groupby("interaction_type")["interaction_weight"].unique())

    print("\nTimestamp range:")
    timestamps = pd.to_datetime(
        interactions["interaction_timestamp"], utc=True
    )
    print("Earliest:", timestamps.min())
    print("Latest:", timestamps.max())

    print("\n========== VALIDATION COMPLETE ==========")


if __name__ == "__main__":
    validate_data()
