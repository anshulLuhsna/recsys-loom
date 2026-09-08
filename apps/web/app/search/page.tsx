"use client";

import { FormEvent, useEffect, useState } from "react";
import { ProductCard } from "@/components/ProductCard";
import {
  fetchDemoCustomers,
  searchCatalog,
  type DemoCustomer,
  type ProductCard as Product,
} from "@/lib/api";

export default function SearchPage() {
  const [query, setQuery] = useState("black oversized hoodie");
  const [customers, setCustomers] = useState<DemoCustomer[]>([]);
  const [customerId, setCustomerId] = useState("");
  const [items, setItems] = useState<Product[]>([]);
  const [chips, setChips] = useState<string[]>([]);
  const [showSignals, setShowSignals] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDemoCustomers()
      .then(setCustomers)
      .catch((err: Error) => setError(err.message));
  }, []);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const payload = await searchCatalog(query.trim(), customerId || undefined);
      setItems(payload.results ?? []);
      const intent = payload.parsed_intent ?? {};
      setChips(
        [
          intent.color,
          intent.product_type,
          intent.section,
          ...(intent.style_terms ?? []),
        ].filter((value): value is string => Boolean(value)),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <section className="hero">
        <div>
          <h1>Search.</h1>
          <p className="lede">
            Query relevance first, personalization second. This is a hybrid
            lexical + semantic + structured catalog search, not a click-trained
            search ranker.
          </p>
        </div>
        <div className="panel">
          <label htmlFor="search-customer">Optional customer context</label>
          <select
            id="search-customer"
            value={customerId}
            onChange={(event) => setCustomerId(event.target.value)}
          >
            <option value="">Anonymous search</option>
            {customers.map((customer) => (
              <option key={customer.customer_id} value={customer.customer_id}>
                {customer.display_name}
              </option>
            ))}
          </select>
        </div>
      </section>
      <form className="searchbar" onSubmit={onSubmit}>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="linen summer shirt"
        />
        <button type="submit">Search</button>
      </form>
      {chips.length ? (
        <div className="chips">
          {chips.map((chip) => (
            <span className="chip" key={chip}>
              {chip}
            </span>
          ))}
        </div>
      ) : null}
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <h2>{items.length ? `${items.length} results` : "Results"}</h2>
        <button className="toggle" type="button" onClick={() => setShowSignals((value) => !value)}>
          {showSignals ? "Hide signals" : "Show signals"}
        </button>
      </div>
      {loading ? <div className="loading">Retrieving the catalog…</div> : null}
      {error ? <div className="error">{error}</div> : null}
      {!loading && !error && items.length === 0 ? (
        <div className="empty">No products matched that query.</div>
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
    </main>
  );
}
