import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = "/api";

const CATEGORY_ICONS = {
  Electronics: "⌁",
  Clothing: "◇",
  Home: "⌂",
  Books: "▤",
  Sports: "↗",
  Beauty: "✿",
  Toys: "✧",
  Accessories: "◈",
};

function categoryIcon(category) {
  return CATEGORY_ICONS[category] || "✦";
}

function formatPrice(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? `$${amount.toFixed(2)}` : "Price unavailable";
}

function App() {
  const [inputId, setInputId] = useState("");
  const [activeCustomerId, setActiveCustomerId] = useState("U9999");
  const [products, setProducts] = useState([]);
  const [strategy, setStrategy] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

 const loadRecommendations = useCallback(async (customerId) => {
  setLoading(true);
  setError("");

  try {
    const response = await fetch(
      `${API}/recommendations/${encodeURIComponent(customerId)}?top_n=10`
    );

    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(
        detail?.detail ||
        "We couldn't retrieve recommendations for that customer."
      );
    }

    const data = await response.json();

    setProducts(Array.isArray(data.recommendations) ? data.recommendations : []);
    setStrategy(data.strategy || "");
    setActiveCustomerId(customerId);
  } catch (err) {
    setProducts([]);
    setStrategy("");
    setError(
      err instanceof TypeError
        ? "SmartCart couldn't reach the recommendation service. Please check that the API is running and try again."
        : err.message || "Something went wrong while loading recommendations."
    );
  } finally {
    setLoading(false);
  }
}, []);

useEffect(() => {
  let cancelled = false;

  async function fetchInitialRecommendations() {
    try {
      const response = await fetch(`${API}/recommendations/U9999?top_n=10`);

      if (!response.ok) {
        throw new Error("Unable to load initial recommendations.");
      }

      const data = await response.json();

      if (!cancelled) {
        setProducts(Array.isArray(data.recommendations) ? data.recommendations : []);
        setStrategy(data.strategy || "");
      }
    } catch (err) {
      if (!cancelled) {
        setError(
          err instanceof TypeError
            ? "SmartCart couldn't reach the recommendation service."
            : err.message
        );
      }
    } finally {
      if (!cancelled) {
        setLoading(false);
      }
    }
  }

  fetchInitialRecommendations();

  return () => {
    cancelled = true;
  };
}, []);


  function handleSearch(event) {
  event.preventDefault();
  const customerId = inputId.trim();

  if (!customerId) {
    setError("Enter a customer ID to retrieve recommendations.");
    return;
  }

  setLoading(true);
  setError("");
  loadRecommendations(customerId);
}

  const isPersonalized = strategy === "category_based";
  const hasResults = products.length > 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="SmartCart AI home">
          <span className="brand-icon">S</span>
          <span>SmartCart<span className="brand-ai"> AI</span></span>
        </a>
        <nav className="nav-links" aria-label="Main navigation">
          <a className="active" href="#recommendations">Discover</a>
          <a href="#about">How it works</a>
        </nav>
        <div className="topbar-status"><span className="status-dot" /> Recommendation engine</div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">SMARTER PRODUCT DISCOVERY</span>
            <h1>Find what fits.<br /><span>Discover what’s next.</span></h1>
            <p>Explore product recommendations informed by customer interaction patterns and catalogue insights.</p>

            <form className="search-form" onSubmit={handleSearch}>
              <label className="search-field">
                <span className="search-icon" aria-hidden="true">⌕</span>
                <input
                  value={inputId}
                  onChange={(event) => setInputId(event.target.value)}
                  placeholder="Enter customer ID"
                  aria-label="Customer ID"
                  autoComplete="off"
                />
              </label>
              <button type="submit" disabled={loading}>
                {loading ? "Loading…" : "Find recommendations"}
                <span aria-hidden="true"> →</span>
              </button>
            </form>
            <p className="search-note">Enter a customer ID to explore a tailored recommendation set.</p>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="visual-orbit orbit-one" />
            <div className="visual-orbit orbit-two" />
            <div className="visual-center"><span>✦</span><strong>AI</strong><small>PRODUCT<br />DISCOVERY</small></div>
            <div className="float-card float-card-one"><span>✧</span><div><strong>Relevant picks</strong><small>Ranked from interactions</small></div></div>
            <div className="float-card float-card-two"><span>↗</span><div><strong>Smart discovery</strong><small>Explore the catalogue</small></div></div>
          </div>
        </section>

        <section className="recommendations" id="recommendations">
          <div className="section-heading">
            <div>
              <span className="eyebrow">YOUR DISCOVERY FEED</span>
              <h2>{isPersonalized ? "Picked around your interests" : strategy === "popularity_fallback" ? "Popular across the catalogue" : "Recommended products"}</h2>
              <p>
                {activeCustomerId
                  ? `Recommendation results for customer ${activeCustomerId}.`
                  : "A selection generated by the SmartCart recommendation engine."}
              </p>
            </div>
            {strategy && (
              <div className={`strategy-pill ${isPersonalized ? "personalized" : "fallback"}`}>
                <span className="status-dot" />
                {isPersonalized ? "Personalized recommendations" : "Popularity-based recommendations"}
              </div>
            )}
          </div>

          {error && <div className="error-message" role="alert">{error}</div>}

          {loading ? (
            <div className="loading-state" aria-live="polite"><div className="spinner" /><p>Finding relevant products…</p></div>
          ) : hasResults ? (
            <div className="product-grid">
              {products.map((product, index) => (
                <article className="product-card" key={product.product_id || `${product.product_name}-${index}`}>
                  <div className={`product-art art-${index % 6}`}>
                    <span className="product-category">{product.category || "Product"}</span>
                    <div className="product-symbol" aria-hidden="true">{categoryIcon(product.category)}</div>
                    <span className="product-number">{product.product_id}</span>
                  </div>
                  <div className="product-info">
                    <div className="product-brand">{product.brand || "SmartCart catalogue"}</div>
                    <h3>{product.product_name || "Unnamed product"}</h3>
                    <p className="product-description">{product.description || "Explore this item from the product catalogue."}</p>
                    <div className="product-meta">
                      <span>★ {Number.isFinite(Number(product.average_rating)) ? Number(product.average_rating).toFixed(1) : "—"}</span>
                      <span>{product.category || "General"}</span>
                    </div>
                    <p className="recommendation-reason">
                      {product.recommendation_reason || (isPersonalized
                        ? "Selected using your historical category interactions."
                        : "Selected using popularity across training interactions.")}
                    </p>
                    <div className="product-bottom">
                      <strong className="product-price">{formatPrice(product.price)}</strong>
                      <span className="score" title="Historical interaction-weight score, not a purchase probability">
                        Interaction score <strong>{Number.isFinite(Number(product.recommendation_score)) ? Number(product.recommendation_score).toFixed(1) : "—"}</strong>
                      </span>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          ) : !error ? (
            <div className="empty-state"><strong>No recommendations available</strong><p>Try another customer ID or check the catalogue data.</p></div>
          ) : null}
        </section>

        <section className="about-section" id="about">
          <div>
            <span className="eyebrow">HOW SMARTCART WORKS</span>
            <h2>Recommendations grounded in interaction data.</h2>
            <p>SmartCart compares recommendation approaches using historical synthetic shopping interactions. Returning customers can receive category-aware recommendations; customers without a usable profile receive popularity-based suggestions.</p>
          </div>
          <div className="about-stat"><strong>4</strong><span>Recommendation approaches evaluated</span></div>
          <div className="about-stat"><strong>1,000</strong><span>Products in the synthetic catalogue</span></div>
        </section>
      </main>

      <footer>
        <span>SmartCart AI</span>
        <span>React + FastAPI · E-commerce recommendation research MVP</span>
      </footer>
    </div>
  );
}

export default App;
