
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_data():
    interactions = pd.read_csv(RAW_DIR / "interactions.csv")

    interactions["interaction_timestamp"] = pd.to_datetime(
        interactions["interaction_timestamp"],
        utc=True
    )

    interactions = interactions.sort_values(
        "interaction_timestamp"
    ).reset_index(drop=True)

    # Chronological 70% / 15% / 15% split.
    train_cutoff = interactions["interaction_timestamp"].quantile(0.70)
    validation_cutoff = interactions["interaction_timestamp"].quantile(0.85)

    train = interactions[
        interactions["interaction_timestamp"] <= train_cutoff
    ].copy()

    validation = interactions[
        (interactions["interaction_timestamp"] > train_cutoff)
        & (interactions["interaction_timestamp"] <= validation_cutoff)
    ].copy()

    test = interactions[
        interactions["interaction_timestamp"] > validation_cutoff
    ].copy()

    # Validation warm-start: users and products known during training.
    train_users = set(train["user_id"])
    train_products = set(train["product_id"])

    validation_warm = validation[
        validation["user_id"].isin(train_users)
        & validation["product_id"].isin(train_products)
    ].copy()

    # Final training data after model selection.
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True
    ).sort_values("interaction_timestamp").reset_index(drop=True)

    # Test warm-start: known from training + validation.
    train_validation_users = set(train_validation["user_id"])
    train_validation_products = set(train_validation["product_id"])

    final_warm_test = test[
        test["user_id"].isin(train_validation_users)
        & test["product_id"].isin(train_validation_products)
    ].copy()

    cold_user_test = test[
        ~test["user_id"].isin(train_validation_users)
    ].copy()

    cold_product_test = test[
        test["user_id"].isin(train_validation_users)
        & ~test["product_id"].isin(train_validation_products)
    ].copy()

    # Save all splits.
    datasets = {
        "train.csv": train,
        "validation.csv": validation,
        "test.csv": test,
        "validation_warm_test.csv": validation_warm,
        "train_validation.csv": train_validation,
        "final_warm_test.csv": final_warm_test,
        "cold_user_test.csv": cold_user_test,
        "cold_product_test.csv": cold_product_test,
    }

    for filename, dataframe in datasets.items():
        dataframe.to_csv(PROCESSED_DIR / filename, index=False)

    print("\n========== CHRONOLOGICAL SPLIT ==========")
    print("Training cutoff:", train_cutoff)
    print("Validation cutoff:", validation_cutoff)
    print("Training interactions:", len(train))
    print("Validation interactions:", len(validation))
    print("Test interactions:", len(test))
    print("Validation warm-start interactions:", len(validation_warm))
    print("Final test warm-start interactions:", len(final_warm_test))
    print("Cold-user test interactions:", len(cold_user_test))
    print("Cold-product test interactions:", len(cold_product_test))

    print("\n========== DATE RANGES ==========")
    for name, dataframe in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(f"\n{name}:")
        print(dataframe["interaction_timestamp"].min())
        print(dataframe["interaction_timestamp"].max())

    # Verify chronological separation.
    assert train["interaction_timestamp"].max() <= train_cutoff
    assert validation["interaction_timestamp"].min() > train_cutoff
    assert validation["interaction_timestamp"].max() <= validation_cutoff
    assert test["interaction_timestamp"].min() > validation_cutoff

    assert len(train) + len(validation) + len(test) == len(interactions)

    print("\nChronological train/validation/test split validated successfully.")
    print("Saved split files to:", PROCESSED_DIR)


if __name__ == "__main__":
    preprocess_data()
