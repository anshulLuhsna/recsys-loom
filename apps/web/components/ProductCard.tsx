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
  if (sources.includes("two_tower")) {
    lines.push("High two-tower compatibility");
  }
  if (sources.includes("bm25")) {
    lines.push("Matches query wording");
  }
  if (sources.includes("semantic")) {
    lines.push("Semantic match");
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
  return (
    <article className="card">
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
        {showSignals ? (
          <div className="signals">
            <strong>Recommendation signals</strong>
            {signalLines(product).map((line) => (
              <div key={line}>{line}</div>
            ))}
            {product.sources?.length ? (
              <div>{product.sources.join(" · ")}</div>
            ) : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}
