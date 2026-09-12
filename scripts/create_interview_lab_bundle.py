#!/usr/bin/env python3
"""Create a small synthetic serving bundle for the disposable DevOps lab."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ARTICLE_FIELDS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]
PRODUCTS = [
    ("Black Tailored Trousers", "Trousers", "Black", "Womenswear"),
    ("Blue Linen Shirt", "Shirt", "Blue", "Menswear"),
    ("Green Casual Jacket", "Jacket", "Green", "Menswear"),
    ("White Office Top", "Top", "White", "Womenswear"),
    ("Red Summer Dress", "Dress", "Red", "Womenswear"),
    ("Grey Oversized Hoodie", "Hoodie", "Grey", "Divided"),
    ("Beige Knit Sweater", "Sweater", "Beige", "Womenswear"),
    ("Navy Chino Shorts", "Shorts", "Blue", "Menswear"),
    ("Pink Cotton Blouse", "Blouse", "Pink", "Womenswear"),
    ("Brown Utility Coat", "Coat", "Brown", "Menswear"),
    ("Yellow Ribbed Cardigan", "Cardigan", "Yellow", "Womenswear"),
    ("White Canvas Trainers", "Sneakers", "White", "Divided"),
]
JPEG_1X1 = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
    "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/"
    "9oADAMBAAIQAxAAAAEf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAA"
    "AAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAA"
    "AAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/"
    "xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EF//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EF//"
    "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EF//2Q=="
)


def card(row: dict[str, str]) -> dict[str, object]:
    article_id = row["article_id"]
    return {
        "article_id": article_id,
        "name": row["prod_name"],
        "product_type": row["product_type_name"],
        "color": row["colour_group_name"],
        "section": row["section_name"],
        "garment_group": row["garment_group_name"],
        "department": row["department_name"],
        "description": row["detail_desc"],
        "image_path": f"images/{article_id[:3]}/{article_id}.jpg",
        "score": 1.0,
        "sources": ["synthetic_lab"],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (name, product_type, color, section) in enumerate(PRODUCTS, start=1):
        article_id = f"001{index:07d}"
        rows.append(
            {
                "article_id": article_id,
                "prod_name": name,
                "product_type_name": product_type,
                "product_group_name": "Garment",
                "colour_group_name": color,
                "perceived_colour_master_name": color,
                "department_name": f"{section} Department",
                "index_name": section,
                "index_group_name": section,
                "section_name": section,
                "garment_group_name": product_type,
                "detail_desc": f"Synthetic {color.lower()} {product_type.lower()} for operations testing.",
            }
        )

    catalog_path = output / "articles.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARTICLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    overnight = output / "artifacts" / "overnight"
    overnight.mkdir(parents=True)
    customers = [
        {
            "customer_id": "synthetic-customer-a",
            "display_name": "Synthetic customer A",
            "history_bucket": "3-5 purchases",
            "summary": "Generated identity for the disposable interview lab.",
        },
        {
            "customer_id": "synthetic-customer-b",
            "display_name": "Synthetic customer B",
            "history_bucket": "11-20 purchases",
            "summary": "Generated identity for the disposable interview lab.",
        },
    ]
    (overnight / "demo_customers.json").write_text(
        json.dumps({"customers": customers}, indent=2),
        encoding="utf-8",
    )

    recommendations = {}
    for customer_index, customer in enumerate(customers):
        ordered = rows[customer_index:] + rows[:customer_index]
        recommendations[customer["customer_id"]] = {
            "customer_id": customer["customer_id"],
            "model_version": "synthetic-lab-v1",
            "candidate_count": len(rows),
            "latency_ms": 0.0,
            "recommendations": [card(row) for row in ordered[:12]],
        }
    (overnight / "demo_recommendations.json").write_text(
        json.dumps(recommendations, indent=2),
        encoding="utf-8",
    )

    popularity = output / "artifacts" / "recent_popularity_7d"
    popularity.mkdir(parents=True)
    (popularity / "metrics.json").write_text(
        json.dumps(
            {
                "top_articles": [
                    {
                        "article_id": row["article_id"],
                        "purchase_count": len(rows) - index,
                    }
                    for index, row in enumerate(rows)
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for row in rows:
        article_id = row["article_id"]
        image_dir = output / "images" / article_id[:3]
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{article_id}.jpg").write_bytes(JPEG_1X1)

    manifest = {
        "bundle_type": "synthetic_interview_lab",
        "bundle_version": "synthetic-lab-v1",
        "synthetic": True,
        "catalog_sha256": sha256(catalog_path),
        "article_count": len(rows),
        "customer_count": len(customers),
    }
    (output / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    write_bundle(args.output)
    if args.archive:
        if args.archive.resolve().is_relative_to(args.output.resolve()):
            raise RuntimeError("archive path must be outside the bundle directory")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, "w:gz") as archive:
            archive.add(args.output, arcname=".")
    print(json.dumps({"output": str(args.output), "archive": str(args.archive or "")}))


if __name__ == "__main__":
    main()
