"""Generate reproducible synthetic e-commerce data. Run: python -m src.data.generate_data"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RANDOM_SEED

CATEGORIES = ["Electronics", "Clothing", "Home", "Books", "Sports", "Beauty", "Toys", "Accessories"]
WEIGHTS = {"view": 1, "click": 2, "add_to_cart": 3, "purchase": 5}
VOCAB = [
    "wireless smart portable digital gaming audio",
    "casual formal cotton summer comfortable fashion",
    "kitchen storage decor cleaning furniture lighting",
    "fiction history science learning mystery career",
    "fitness training outdoor running cycling exercise",
    "skincare haircare natural moisturizing daily beauty",
    "educational creative building puzzle children fun",
    "minimal durable travel everyday premium style",
]

def generate_data(seed=RANDOM_SEED, n_users=500, n_products=1000, n_interactions=15000):
    rng = np.random.default_rng(seed)
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    cats = np.resize(np.array(CATEGORIES), n_products)
    rng.shuffle(cats)
    popularity = rng.lognormal(0, 0.9, n_products)
    brands = {c: [f"{c[:3].upper()}Brand{i}" for i in range(1, 6)] for c in CATEGORIES}
    product_rows = []
    for idx, category in enumerate(cats):
        brand = rng.choice(brands[category])
        terms = rng.choice(VOCAB[CATEGORIES.index(category)].split(), 3, replace=False)
        product_rows.append({
            "product_id": f"P{idx+1:04d}",
            "product_name": f"{category} Product {idx+1:04d}",
            "category": category, "brand": brand,
            "price": round(float(rng.lognormal(3.7, 0.75)), 2),
            "description": f"{brand} {' '.join(terms)} {category.lower()} product",
            "average_rating": round(float(rng.uniform(2.8, 5.0)), 1),
        })
    products = pd.DataFrame(product_rows)
    users = pd.DataFrame({
        "user_id": [f"U{i+1:04d}" for i in range(n_users)],
        "signup_timestamp": pd.Timestamp("2026-01-01", tz="UTC") +
                            pd.to_timedelta(rng.integers(0, 150, n_users), unit="D"),
        "price_sensitivity": rng.choice(["budget", "mid_range", "premium"], n_users,
                                        p=[0.35, 0.45, 0.20]),
    })

    # Latent preferences drive the simulator but are deliberately not exported.
    preferences = rng.dirichlet(np.full(len(CATEGORIES), 0.65), size=n_users)
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    product_cat_idx = np.array([cat_to_idx[c] for c in cats])
    popularity /= popularity.sum()
    active_users = max(1, n_users - 10)       # Last 10 users have no events.
    eligible_products = np.arange(max(1, n_products - 20))  # Last 20 products have no events.
    timestamps = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(
        rng.integers(0, 180 * 24 * 60, n_interactions), unit="m")
    event_types = np.array(["view", "click", "add_to_cart", "purchase"])
    event_probs = [0.58, 0.22, 0.12, 0.08]
    rows = []
    for j in range(n_interactions):
        user_idx = int(rng.integers(0, active_users))
        category_idx = int(rng.choice(len(CATEGORIES), p=preferences[user_idx]))
        candidates = eligible_products[product_cat_idx[eligible_products] == category_idx]
        if len(candidates) == 0:
            candidates = eligible_products
        probs = popularity[candidates] / popularity[candidates].sum()
        product_idx = int(rng.choice(candidates, p=probs))
        action = str(rng.choice(event_types, p=event_probs))
        rows.append({
            "interaction_id": f"I{j+1:06d}",
            "user_id": f"U{user_idx+1:04d}",
            "product_id": f"P{product_idx+1:04d}",
            "interaction_type": action,
            "interaction_weight": WEIGHTS[action],
            "interaction_timestamp": timestamps[j],
        })
    interactions = pd.DataFrame(rows).sort_values("interaction_timestamp").reset_index(drop=True)
    users.to_csv(out / "users.csv", index=False)
    products.to_csv(out / "products.csv", index=False)
    interactions.to_csv(out / "interactions.csv", index=False)
    return users, products, interactions

if __name__ == "__main__":
    u, p, i = generate_data()
    print(f"Generated {len(u)} users, {len(p)} products, {len(i)} interactions.")
    print("Cold-start users reserved: 10; cold-start products reserved: 20.")