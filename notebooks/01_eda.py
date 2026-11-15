
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

users = pd.read_csv(DATA_DIR / "users.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
interactions = pd.read_csv(DATA_DIR / "interactions.csv")

interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"], utc=True
)

# 1. Interaction distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=interactions, x="interaction_type")
plt.title("Interaction Type Distribution")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "interaction_distribution.png")
plt.close()

# 2. Category distribution
plt.figure(figsize=(9, 5))
sns.countplot(data=products, y="category",
              order=products["category"].value_counts().index)
plt.title("Product Category Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_distribution.png")
plt.close()

# 3. Most popular products
popular = interactions["product_id"].value_counts().head(10)

plt.figure(figsize=(10, 5))
popular.sort_values().plot(kind="barh")
plt.title("Top 10 Most Interacted Products")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_products.png")
plt.close()

# 4. User activity
user_activity = interactions.groupby("user_id").size()

plt.figure(figsize=(8, 5))
sns.histplot(user_activity, bins=30)
plt.title("Interactions Per User")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "user_activity.png")
plt.close()

# 5. Daily interaction trend
daily = (
    interactions.set_index("interaction_timestamp")
    .resample("D")
    .size()
)

plt.figure(figsize=(12, 5))
daily.plot()
plt.title("Daily Interaction Trend")
plt.ylabel("Number of interactions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "daily_trend.png")
plt.close()

# 6. Weighted category engagement
merged = interactions.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_engagement = (
    merged.groupby("category")["interaction_weight"]
    .sum()
    .sort_values(ascending=False)
)

print("\nCategory engagement:")
print(category_engagement)

print("\nUser activity summary:")
print(user_activity.describe())

print("\nPurchase count:")
print((interactions["interaction_type"] == "purchase").sum())

print("\nEDA complete. Charts saved in reports/figures/")
