"use client";

import { useState } from "react";
import { imageUrl, type ProductCard as Product } from "@/lib/api";

function signalLines(product: Product): string[] {
  const lines: string[] = [];
  const sources = product.sources ?? [];
  if (sources.includes("recent_7d_pop") || sources.includes("popularity")) {
    lines.push("Popular recently");
  }
  if (sources.includes("als")) {
    lines.push("Similar customers bought this");
  }
  if (sources.includes("repeat_purchase")) {
    lines.push("Previously purchased");
  }
  if (sources.includes("content")) {
    lines.push("Matches categories you frequently purchase");
  }
  if (sources.includes("two_tower")) {
    lines.push("High two-tower compatibility");
  }
  if (sources.includes("bm25")) {
    lines.push("Matches query wording");
  }
  if (sources.includes("semantic")) {
    lines.push("Semantic match");
  }
  if (sources.includes("visual")) {
    lines.push("Visual match");
  }
  if (sources.includes("structured")) {
    lines.push("Matches parsed attributes");
  }
  if (sources.length >= 2) {
    lines.push("Found by multiple retrieval systems");
  }
  return lines;
}

export function ProductCard({
  product,
  showSignals,
}: {
  product: Product;
  showSignals: boolean;
}) {
  const [broken, setBroken] = useState(false);
  const [open, setOpen] = useState(false);
  const lines = signalLines(product);

  return (
    <>
      <article className="card">
        <button className="card-hit" type="button" onClick={() => setOpen(true)}>
          {broken ? (
            <div className="ph" />
          ) : (
            <img
              src={imageUrl(product.image_path)}
              alt={product.name}
              onError={() => setBroken(true)}
            />
          )}
          <div className="meta">
            <h3>{product.name}</h3>
            <div className="sub">
              {product.product_type} · {product.color}
            </div>
            <div className="sub">
              {product.section} · {product.garment_group}
            </div>
            {showSignals && lines.length ? (
              <div className="signals">
                <strong>Recommendation signals</strong>
                {lines.map((line) => (
                  <div key={line}>{line}</div>
                ))}
                {product.sources?.length ? (
                  <div>{product.sources.join(" · ")}</div>
                ) : null}
              </div>
            ) : null}
          </div>
        </button>
      </article>
      {open ? (
        <div className="modal-backdrop" onClick={() => setOpen(false)}>
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby={`product-${product.article_id}`}
            onClick={(event) => event.stopPropagation()}
          >
            {broken ? (
              <div className="ph" />
            ) : (
              <img src={imageUrl(product.image_path)} alt={product.name} />
            )}
            <div className="meta">
              <h3 id={`product-${product.article_id}`}>{product.name}</h3>
              <div className="sub">
                {product.product_type} · {product.color}
              </div>
              <div className="sub">
                {product.section} · {product.garment_group}
              </div>
              {product.department ? (
                <div className="sub">{product.department}</div>
              ) : null}
              {product.description ? <p className="lede">{product.description}</p> : null}
              <div className="signals">
                <strong>Recommendation signals</strong>
                <p className="sub">
                  These are observed retrieval/ranking features, not causal
                  explanations.
                </p>
                {lines.length ? (
                  lines.map((line) => <div key={line}>{line}</div>)
                ) : (
                  <div>No retrieval signals attached.</div>
                )}
                {product.sources?.length ? (
                  <div>{product.sources.join(" · ")}</div>
                ) : null}
              </div>
              <button className="toggle" type="button" onClick={() => setOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
