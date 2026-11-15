# SmartCart AI — E-Commerce Product Recommendation Engine

An end-to-end product recommendation system built using Python, machine learning, FastAPI, and React. SmartCart AI compares four recommendation approaches using synthetic e-commerce interaction data and serves personalized product recommendations through an API and web interface.

## 1. Project Overview

SmartCart AI recommends products using historical user interaction signals.

| Interaction | Weight |
|---|---:|
| View | 1 |
| Click | 2 |
| Add to cart | 3 |
| Purchase | 5 |

The project implements and evaluates:
- Popularity-based recommendation
- Item-based collaborative filtering
- Content-based filtering
- Category-based recommendation

The selected category-based recommender uses learned user category preferences and historical product popularity. Unknown users receive a global popularity fallback.

**Project objective:** Build a reproducible recommendation pipeline with defensible evaluation, a saved model artifact, and a working inference interface.

This is an experimental decision-support prototype evaluated on synthetic data. Its results do not establish real-world commercial performance.

## 2. Technology Stack

- Python 3.11
- Pandas and NumPy
- SciPy and scikit-learn
- Joblib
- FastAPI and Uvicorn
- React and Vite
- Pytest
- Matplotlib and Seaborn

## 3. System Architecture

```text
Synthetic Data Generator
        |
        v
Raw User, Product and Interaction Data
        |
        v
Data Validation and EDA
        |
        v
Chronological Data Preparation
        |
        +-------------------------+
        |                         |
        v                         v
   Training Set              Validation Set
        |                         |
        +------------+------------+
                     |
                     v
          Model Development
                     |
       +-------------+-------------+
       |             |             |
       v             v             v
  Popularity    Collaborative   Content-Based
  Baseline      Filtering       Filtering
                     |
                     v
             Category-Based Model
                     |
                     v
           Validation Comparison
                     |
                     v
             Selected Model
                     |
                     v
         Train + Validation Data
                     |
                     v
             Saved Model Artifact
                     |
                     v
                  FastAPI
                     |
                     v
                React UI
```

The deployed artifact contains the category-based recommender and popularity recommender. Collaborative filtering and content-based filtering are implemented for offline comparison.

## 4. Dataset

The project uses synthetically generated data with a fixed random seed of 42.

| Dataset | Records |
|---|---:|
| Users | 500 |
| Products | 1,000 |
| Interactions | 15,000 |

Interaction distribution:

| Interaction | Count |
|---|---:|
| Views | 8,666 |
| Clicks | 3,341 |
| Add to cart | 1,787 |
| Purchases | 1,206 |

Data files:
- `data/raw/users.csv`
- `data/raw/products.csv`
- `data/raw/interactions.csv`
- `data/processed/products_enriched.csv`

Product enrichment adds descriptive names, brands, and descriptions for recommendation experiments.

### Data limitations

The timestamps are assigned independently of the underlying interaction-generation sequence. Consequently, chronological splitting prevents future-dated interactions from entering earlier training data, but does not reproduce realistic sequential shopping behaviour.

The generated preferences and interaction patterns are simplified assumptions.

## 5. Data Preparation and Evaluation

The dataset is partitioned chronologically into three sets:

| Split | Interactions | Share |
|---|---:|---:|
| Training | 10,500 | 70% |
| Validation | 2,250 | 15% |
| Testing | 2,250 | 15% |
| **Total** | **15,000** | **100%** |

The split uses chronological timestamp boundaries.

- Training cutoff: 7 May 2026, 07:21:48 UTC
- Validation cutoff: 2 June 2026, 13:25:33 UTC
- Validation warm-start interactions: 2,240
- Final test warm-start interactions: 2,238
- Cold-product test interactions: 12
- Cold-user test interactions: 0

### Evaluation methodology

Add-to-cart and purchase interactions are treated as relevant outcomes.

The evaluation metrics are:
- Precision@10
- Recall@10
- NDCG@10

The four candidate approaches are compared using validation data. The selected model is subsequently trained using the combined training and validation data.

The final held-out test set is reserved for evaluation after model selection.

## 6. Model Comparison

Four approaches were evaluated.

1. **Popularity:** Ranks products using weighted historical interactions.
2. **Item-Based Collaborative Filtering:** Uses item-item cosine similarity from the user-item interaction matrix.
3. **Content-Based Filtering:** Uses product text and metadata to construct user preference profiles.
4. **Category-Based Filtering:** Learns user category preferences and ranks historically interacted-with products within preferred categories.

### Validation results

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.008621 | 0.053448 | 0.027347 |
| Collaborative Filtering | 0.007586 | 0.047989 | 0.025824 |
| Content-Based Filtering | 0.006207 | 0.045402 | 0.021438 |
| **Category-Based Filtering** | **0.010690** | **0.059195** | **0.034862** |

Category-Based Filtering achieved the highest validation performance across all three metrics.

### Final held-out test results

The selected category-based model was compared against the popularity baseline on the untouched test set.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| **Category-Based Filtering** | **0.009722** | **0.066088** | **0.039912** |
| Popularity | 0.009028 | 0.053935 | 0.030704 |

The selected model outperformed the popularity baseline on all three held-out test metrics.

These results demonstrate an improvement within this synthetic dataset and evaluation design. They should not be interpreted as evidence of production-level recommendation quality.

## 7. Error Analysis

The selected model was further examined using per-user recommendation diagnostics.

| Diagnostic | Result |
|---|---:|
| Evaluated users | 288 |
| Users with zero relevant recommendations | 262 |
| Users with at least one relevant recommendation | 26 |
| Unique recommended products | 145 |
| Total catalogue products | 1,000 |
| Catalogue coverage | 14.5% |
| Eligible-product coverage | 15.17% |

### Cold-start observations

- Unknown users receive globally popular recommendations.
- Products with no historical development interactions cannot receive learned popularity scores.
- The final test contains 12 cold-product interactions, including 3 relevant interactions.
- No cold-user interactions were present in the final test period.

The model therefore has limited cold-start product coverage.

### Diagnostic reports

- `reports/error_analysis_summary.csv`
- `reports/error_analysis_users.csv`
- `reports/recommendation_category_distribution.csv`

## 8. Model Artifact and Inference

The final selected recommender is trained on 12,750 training and validation interactions.

The saved artifact is:

`artifacts/recommender.joblib`

The backend loads the saved model artifact rather than retraining models for every request.

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Root endpoint |
| `GET /health` | Backend health |
| `GET /recommendations/{user_id}?top_n=10` | Product recommendations |
| `GET /products/{product_id}` | Product lookup |

Known users receive category-based recommendations. Unknown users receive popularity-based recommendations.

The recommendation score represents a historical interaction-popularity signal, not a predicted purchase probability or calibrated confidence score.

## 9. Installation and Setup — Windows CMD

Open CMD in the project root.

### Step 1: Create the environment

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Generate and prepare data

```cmd
python -m src.data.generate_data
python -m src.data.validate_data
python notebooks/01_eda.py
python -m src.data.preprocess
python -m src.data.enrich_catalogue
```

### Step 3: Compare models using validation data

```cmd
python -m src.evaluation.evaluate
```

### Step 4: Train and save the selected model

```cmd
python -m src.inference.train_and_save
```

### Step 5: Evaluate on the held-out test set

```cmd
python -m src.evaluation.evaluate_final
```

### Step 6: Generate error-analysis reports

```cmd
python -m src.evaluation.error_analysis
```

### Step 7: Start the backend

```cmd
uvicorn backend.main:app --reload --port 8000
```

Backend:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### Step 8: Start the frontend

Open a second CMD terminal:

```cmd
cd /d D:\Projects\SmartCart-AI\frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually:

`http://localhost:5173`

## 10. Testing

Run backend tests from the project root:

```cmd
python -m pytest tests -v
```

The current test suite contains seven passing tests, covering:

- Backend health
- Personalized recommendations
- Popularity fallback
- Product lookup
- Invalid product handling
- Top-N request validation
- Unknown-user global popularity fallback

Frontend checks:

```cmd
cd /d D:\Projects\SmartCart-AI\frontend
npm run lint
npm run build
```

## 11. Responsible Use and Limitations

SmartCart AI is a recommendation prototype, not an autonomous purchasing or high-stakes decision system.

Known limitations:

- The dataset is synthetic and does not represent actual customer behaviour.
- Synthetic timestamps do not reproduce realistic sequential shopping journeys.
- Absolute ranking metrics remain low.
- A large proportion of evaluated users receive no relevant recommendation in their top ten.
- Catalogue coverage is limited.
- Products with no historical interactions are not recommended by the learned popularity-based approach.
- Cold-user performance cannot be evaluated from the available test data.
- Collaborative filtering and content-based models are not deployed through the API.
- Recommendation scores are historical interaction signals, not probabilities.
- Popularity-based ranking can reinforce exposure to already popular products.
- Production-scale performance and generalization have not been established.

Recommendations should be treated as suggestions and reviewed by a human decision-maker.

## 12. AI Tool Attribution

AI-assisted development tools were used for development support, debugging, code review, and documentation assistance. The project owner is responsible for understanding, validating, and explaining the implementation.

The dataset is synthetically generated using the project's own generator. No external customer dataset is claimed.

---

**Project:** SmartCart AI
**Domain:** E-Commerce Product Recommendation
**Implementation:** Python, FastAPI, React
**Dataset:** Synthetic
**Random seed:** 42
