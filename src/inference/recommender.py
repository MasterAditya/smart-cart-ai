from pathlib import Path
import joblib


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT_DIR / "artifacts" / "recommender.joblib"


def load_recommender():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {ARTIFACT_PATH}. "
            "Run the training script first."
        )

    artifacts = joblib.load(ARTIFACT_PATH)

    required_keys = {
        "category_model",
        "popularity_model",
        "train_interactions",
        "products",
    }

    missing_keys = required_keys - artifacts.keys()

    if missing_keys:
        raise ValueError(
            f"Artifact is missing required components: {missing_keys}"
        )

    return artifacts