"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ProductCard } from "@/components/ProductCard";
import {
  fetchDemoCustomers,
  fetchRecommendations,
  type DemoCustomer,
  type ProductCard as Product,
} from "@/lib/api";

export default function HomePage() {
  const [customers, setCustomers] = useState<DemoCustomer[]>([]);
  const [customerId, setCustomerId] = useState("");
  const [customId, setCustomId] = useState("");
  const [items, setItems] = useState<Product[]>([]);
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
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!customerId) {
      return;
    }
    setLoading(true);
    setError("");
    fetchRecommendations(customerId)
      .then((payload) => setItems(payload.recommendations ?? []))
      .catch((err: Error) => {
        setItems([]);
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
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <h2>Personalized Top 12</h2>
        <button className="toggle" type="button" onClick={() => setShowSignals((value) => !value)}>
          {showSignals ? "Hide signals" : "Show signals"}
        </button>
      </div>
      {loading ? <div className="loading">Scoring the slate…</div> : null}
      {error ? <div className="error">{error}</div> : null}
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
      <p className="lede">
        Looking for something specific? <Link href="/search">Search the catalog</Link>.
      </p>
    </main>
  );
}
