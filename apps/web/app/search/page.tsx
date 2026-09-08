"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { ProductCard } from "@/components/ProductCard";
import {
  fetchDemoCustomers,
  searchCatalog,
  type DemoCustomer,
  type ProductCard as Product,
} from "@/lib/api";

const EXAMPLES = [
  "black oversized hoodie",
  "linen shirt for summer",
  "blue women's dress",
  "minimal black trousers",
  "casual green jacket",
  "white top for office",
];

export default function SearchPage() {
  const [query, setQuery] = useState("black oversized hoodie");
  const [customers, setCustomers] = useState<DemoCustomer[]>([]);
  const [customerId, setCustomerId] = useState("");
  const [items, setItems] = useState<Product[]>([]);
  const [chips, setChips] = useState<string[]>([]);
  const [typeFilter, setTypeFilter] = useState("");
  const [colorFilter, setColorFilter] = useState("");
  const [sectionFilter, setSectionFilter] = useState("");
  const [showSignals, setShowSignals] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    fetchDemoCustomers()
      .then(setCustomers)
      .catch((err: Error) => setError(err.message));
  }, []);

  const filtered = useMemo(
    () =>
      items.filter((item) => {
        if (typeFilter && item.product_type !== typeFilter) {
          return false;
        }
        if (colorFilter && item.color !== colorFilter) {
          return false;
        }
        if (sectionFilter && item.section !== sectionFilter) {
          return false;
        }
        return true;
      }),
    [items, typeFilter, colorFilter, sectionFilter],
  );

  const types = useMemo(
    () => Array.from(new Set(items.map((item) => item.product_type).filter(Boolean))),
    [items],
  );
  const colors = useMemo(
    () => Array.from(new Set(items.map((item) => item.color).filter(Boolean))),
    [items],
  );
  const sections = useMemo(
    () => Array.from(new Set(items.map((item) => item.section).filter(Boolean))),
    [items],
  );

  async function runSearch(nextQuery: string, nextCustomerId = customerId) {
    if (!nextQuery.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    setSearched(true);
    setTypeFilter("");
    setColorFilter("");
    setSectionFilter("");
    try {
      const payload = await searchCatalog(
        nextQuery.trim(),
        nextCustomerId || undefined,
      );
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

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    await runSearch(query);
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
            onChange={(event) => {
              const nextCustomerId = event.target.value;
              setCustomerId(nextCustomerId);
              if (searched) {
                void runSearch(query, nextCustomerId);
              }
            }}
          >
            <option value="">Anonymous search</option>
            {customers.map((customer) => (
              <option key={customer.customer_id} value={customer.customer_id}>
                {customer.display_name}
              </option>
            ))}
          </select>
          <p>
            {customerId
              ? "Personalized search can bump in-query items this customer already likes. It cannot replace the query."
              : "Anonymous search is query relevance only."}
          </p>
        </div>
      </section>
      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            className="chip"
            key={example}
            type="button"
            onClick={() => {
              setQuery(example);
              void runSearch(example);
            }}
          >
            {example}
          </button>
        ))}
      </div>
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
      {items.length ? (
        <div className="filters">
          <div>
            <label htmlFor="type-filter">Type</label>
            <select
              id="type-filter"
              value={typeFilter}
              onChange={(event) => setTypeFilter(event.target.value)}
            >
              <option value="">All types</option>
              {types.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="color-filter">Color</label>
            <select
              id="color-filter"
              value={colorFilter}
              onChange={(event) => setColorFilter(event.target.value)}
            >
              <option value="">All colors</option>
              {colors.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="section-filter">Section</label>
            <select
              id="section-filter"
              value={sectionFilter}
              onChange={(event) => setSectionFilter(event.target.value)}
            >
              <option value="">All sections</option>
              {sections.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : null}
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <h2>
          {loading ? "Searching…" : searched ? `${filtered.length} results` : "Results"}
        </h2>
        <button className="toggle" type="button" onClick={() => setShowSignals((value) => !value)}>
          {showSignals ? "Hide signals" : "Show signals"}
        </button>
      </div>
      {loading ? <div className="loading">Retrieving the catalog…</div> : null}
      {error ? <div className="error">{error}</div> : null}
      {!loading && !error && searched && filtered.length === 0 ? (
        <div className="empty">No products matched that query.</div>
      ) : null}
      <div className="grid">
        {filtered.map((product) => (
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
