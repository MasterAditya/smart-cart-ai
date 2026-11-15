
from fastapi import FastAPI, HTTPException, Query

from src.inference.recommender import load_recommender


app = FastAPI(
    title="SmartCart AI",
    description="Personalized E-Commerce Product Recommendation API",
    version="0.3.0"
)


# Load saved models and data — no retraining at startup
artifacts = load_recommender()

category_model = artifacts["category_model"]
popularity_model = artifacts["popularity_model"]
train = artifacts["train_interactions"]
products = artifacts["products"]

product_lookup = products.set_index("product_id")


@app.get("/")
def root():
    return {
        "message": "SmartCart AI recommendation API",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": True
    }


@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: str,
    top_n: int = Query(default=10, ge=1, le=50)
):
    if user_id in category_model.user_category_preferences:
        recommendations = category_model.recommend(
            user_id,
            train,
            top_n=top_n
        )
        strategy = "category_based"

    else:
        recommendations = popularity_model.recommend(
            user_id,
            train,
            top_n=top_n
        )
        strategy = "popularity_fallback"

    output = []

    for item in recommendations:
        product_id = item["product_id"]

        if product_id not in product_lookup.index:
            continue

        product_info = product_lookup.loc[product_id].to_dict()

        if strategy == "category_based":
            reason = "Recommended based on your historical category interactions."
        else:
            reason = "Recommended based on popularity across training interactions."

        output.append({
            "product_id": product_id,
            **product_info,
            "recommendation_score": float(
                item.get("popularity_score", 0)
            ),
            "recommendation_reason": reason
        })

    return {
        "user_id": user_id,
        "strategy": strategy,
        "count": len(output),
        "recommendations": output,
        "recommendation_score_type": "training_interaction_popularity",
    }


@app.get("/products/{product_id}")
def get_product(product_id: str):
    if product_id not in product_lookup.index:
        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )

    return {
        "product_id": product_id,
        **product_lookup.loc[product_id].to_dict()
    }
