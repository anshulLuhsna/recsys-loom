"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ProductCard } from "@/components/ProductCard";
import {
  fetchDemoCustomers,
  fetchRecommendations,
  fetchTrending,
  type DemoCustomer,
  type ProductCard as Product,
} from "@/lib/api";

export default function HomePage() {
  const [customers, setCustomers] = useState<DemoCustomer[]>([]);
  const [customerId, setCustomerId] = useState("");
  const [customId, setCustomId] = useState("");
  const [items, setItems] = useState<Product[]>([]);
  const [trending, setTrending] = useState<Product[]>([]);
  const [serving, setServing] = useState<{
    mode?: string;
    candidates?: number;
    latencyMs?: number;
  } | null>(null);
  const [showSignals, setShowSignals] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const selected = customers.find((customer) => customer.customer_id === customerId);

  useEffect(() => {
    fetchDemoCustomers()
      .then((rows) => {
        setCustomers(rows);
        if (rows[0]) {
          setCustomerId(rows[0].customer_id);
        }
      })
      .catch((err: Error) => setError(err.message));
    fetchTrending()
      .then(setTrending)
      .catch(() => setTrending([]));
  }, []);

  useEffect(() => {
    if (!customerId) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    fetchRecommendations(customerId)
      .then((payload) => {
        setItems(payload.recommendations ?? []);
        setServing({
          mode: payload.serving_mode,
          candidates: payload.candidate_count,
          latencyMs: payload.latency_ms,
        });
      })
      .catch((err: Error) => {
        setItems([]);
        setServing(null);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, [customerId]);

  return (
    <main>
      <section className="hero">
        <div>
          <h1>For You.</h1>
          <p className="lede">
            Twelve next-week purchase candidates from a leakage-safe retrieval
            and ranking stack trained on H&M history. Search is a separate
            query-first catalog path.
          </p>
        </div>
        <div className="panel">
          <label htmlFor="customer">Demo customer</label>
          <select
            id="customer"
            value={customerId}
            onChange={(event) => setCustomerId(event.target.value)}
          >
            {customers.map((customer) => (
              <option key={customer.customer_id} value={customer.customer_id}>
                {customer.display_name} · {customer.history_bucket}
              </option>
            ))}
          </select>
          {selected ? <p>{selected.summary}</p> : null}
          <label htmlFor="custom">Advanced customer ID</label>
          <input
            id="custom"
            value={customId}
            onChange={(event) => setCustomId(event.target.value)}
            placeholder="Paste an anonymized customer id"
          />
          <button
            className="primary"
            type="button"
            onClick={() => customId && setCustomerId(customId.trim())}
          >
            Load ID
          </button>
        </div>
      </section>
      <section className="split">
        <article>
          <h2>Personalized home</h2>
          <p className="lede">
            Uses purchase history, multi-source retrieval, and CatBoost
            YetiRank. The slate answers “what might this customer buy next
            week?”
          </p>
        </article>
        <article>
          <h2>Explicit search</h2>
          <p className="lede">
            Uses query intent, BM25, attributes, and CLIP visual retrieval.
            MiniLM remains optional. The grid answers “what matches this
            wording?” Personalization cannot override the query.
          </p>
          <p className="lede">
            <Link href="/search">Open catalog search</Link>
          </p>
        </article>
      </section>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <div>
          <h2>Personalized Top 12</h2>
          {serving?.mode ? (
            <p>
              {serving.mode === "live_catboost"
                ? `Live CatBoost inference over ${serving.candidates ?? 0} candidates · ${(serving.latencyMs ?? 0).toFixed(1)} ms`
                : "Precomputed demo fallback"}
            </p>
          ) : null}
        </div>
        <button className="toggle" type="button" onClick={() => setShowSignals((value) => !value)}>
          {showSignals ? "Hide signals" : "Show signals"}
        </button>
      </div>
      {loading ? <div className="loading">Scoring the slate…</div> : null}
      {error ? (
        <div className="error">
          {error}. Choose one of the demo customers or an ID from the exported
          serving set. Trending below is the recent-popularity baseline.
        </div>
      ) : null}
      {!loading && !error && items.length === 0 ? (
        <div className="empty">No recommendations for this customer yet.</div>
      ) : null}
      <div className="grid">
        {items.map((product) => (
          <ProductCard
            key={product.article_id}
            product={product}
            showSignals={showSignals}
          />
        ))}
      </div>
      {trending.length ? (
        <>
          <h2>Trending last week</h2>
          <p className="lede">
            Same 12 articles for every customer: recent-popularity before
            2020-09-16. This is the non-personalized baseline.
          </p>
          <div className="grid">
            {trending.map((product) => (
              <ProductCard
                key={product.article_id}
                product={product}
                showSignals={showSignals}
              />
            ))}
          </div>
        </>
      ) : null}
    </main>
  );
}
