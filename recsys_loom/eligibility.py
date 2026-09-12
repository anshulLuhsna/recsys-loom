"""Eligibility policies and ceiling metrics for retrieval evaluation.

The H&M dataset has no live stock, sizes, or availability data. Eligibility
policies are honest proxies that must be versioned and compared, not invented
to look like real inventory.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from enum import Enum


class EligibilityPolicy(Enum):
    """Named eligibility policies for candidate filtering."""

    FULL_CATALOG = "full_catalog"
    OBSERVED_BEFORE_CUTOFF = "observed_before_cutoff"

    def _exhaustive_check(self) -> None:
        _: Enum
        match self:
            case EligibilityPolicy.FULL_CATALOG:
                pass
            case EligibilityPolicy.OBSERVED_BEFORE_CUTOFF:
                pass
            case _ as unreachable:
                _never: never = unreachable  # noqa: F841


def eligible_articles(
    policy: EligibilityPolicy,
    full_catalog: Set[str],
    observed_before_cutoff: Set[str] | None = None,
) -> set[str]:
    """Return the set of article IDs eligible under the given policy."""
    match policy:
        case EligibilityPolicy.FULL_CATALOG:
            return set(full_catalog)
        case EligibilityPolicy.OBSERVED_BEFORE_CUTOFF:
            if observed_before_cutoff is None:
                raise ValueError(
                    "observed_before_cutoff must be provided for this policy"
                )
            return set(observed_before_cutoff)
        case _ as unreachable:
            _never: never = unreachable  # noqa: F841
            raise ValueError(f"Unknown policy: {unreachable}")


def eligibility_ceiling(
    relevant_by_customer: Mapping[str, set[str]],
    eligible: Set[str],
) -> dict[str, float | int]:
    """Measure how many relevant items survive the eligibility filter.

    The ceiling is the upper bound on recall before any retrieval model
    runs. If 10% of future purchases are ineligible articles, no retriever
    can recover them and the ceiling is 90%.
    """
    total_relevant = 0
    surviving_relevant = 0
    customers_with_surviving = 0

    for customer_id, relevant in relevant_by_customer.items():
        total_relevant += len(relevant)
        surviving = len(relevant.intersection(eligible))
        surviving_relevant += surviving
        if surviving > 0:
            customers_with_surviving += 1

    customer_count = len(relevant_by_customer)
    return {
        "policy_article_count": len(eligible),
        "customers": customer_count,
        "total_relevant_pairs": total_relevant,
        "surviving_relevant_pairs": surviving_relevant,
        "lost_relevant_pairs": total_relevant - surviving_relevant,
        "ceiling": surviving_relevant / total_relevant if total_relevant else 0.0,
        "customers_with_surviving_relevant": customers_with_surviving,
        "customer_ceiling": (
            customers_with_surviving / customer_count if customer_count else 0.0
        ),
    }
