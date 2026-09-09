const configuredApiUrl =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export const API_URL = configuredApiUrl.replace(/\/+$/, "");

export type ProductCard = {
  article_id: string;
  name: string;
  product_type: string;
  color: string;
  section: string;
  garment_group: string;
  department?: string;
  description?: string;
  image_path: string;
  score?: number;
  sources?: string[];
  signals?: Record<string, number>;
};

export type DemoCustomer = {
  customer_id: string;
  display_name: string;
  history_bucket: string;
  summary: string;
};

export function imageUrl(path: string): string {
  return `${API_URL}/${path.replace(/^\/+/, "")}`;
}

export async function fetchDemoCustomers(): Promise<DemoCustomer[]> {
  const response = await fetch(`${API_URL}/api/demo-customers`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Could not load demo customers");
  }
  const payload = (await response.json()) as { customers: DemoCustomer[] };
  return payload.customers ?? [];
}

export async function fetchRecommendations(
  customerId: string,
): Promise<{
  recommendations: ProductCard[];
  model_version?: string;
  serving_mode?: string;
  candidate_count?: number;
  latency_ms?: number;
}> {
  const response = await fetch(
    `${API_URL}/api/recommendations/${encodeURIComponent(customerId)}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new Error("Could not load recommendations");
  }
  return response.json();
}

export async function fetchTrending(): Promise<ProductCard[]> {
  const response = await fetch(`${API_URL}/api/trending`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load trending");
  }
  const payload = (await response.json()) as { items?: ProductCard[] };
  return payload.items ?? [];
}

export async function searchCatalog(
  query: string,
  customerId?: string,
): Promise<{
  parsed_intent: {
    color?: string | null;
    product_type?: string | null;
    section?: string | null;
    style_terms?: string[];
    hard_constraints?: Record<string, string>;
  };
  results: ProductCard[];
}> {
  const response = await fetch(`${API_URL}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      customer_id: customerId || null,
      limit: 24,
    }),
  });
  if (!response.ok) {
    throw new Error("Search failed");
  }
  return response.json();
}
