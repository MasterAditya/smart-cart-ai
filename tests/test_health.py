
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models_loaded"] is True


def test_personalized_recommendations():
    response = client.get("/recommendations/U0001")

    assert response.status_code == 200

    data = response.json()

    assert data["strategy"] == "category_based"
    assert data["count"] == 10
    assert len(data["recommendations"]) == 10


def test_popularity_fallback():
    response = client.get("/recommendations/U9999?top_n=5")

    assert response.status_code == 200

    data = response.json()

    assert data["strategy"] == "popularity_fallback"
    assert data["count"] == 5


def test_product_lookup():
    response = client.get("/products/P0189")

    assert response.status_code == 200
    assert response.json()["product_id"] == "P0189"


def test_invalid_product():
    response = client.get("/products/INVALID_PRODUCT")

    assert response.status_code == 404


def test_top_n_validation():
    too_small = client.get("/recommendations/U0001?top_n=0")
    too_large = client.get("/recommendations/U0001?top_n=51")

    assert too_small.status_code == 422
    assert too_large.status_code == 422
