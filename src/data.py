"""
Data loading and merging layer for the Olist Brazilian E-commerce dataset.

Responsibilities:
  - Load raw CSVs from data/raw/
  - Merge the 9 Olist tables into a single analytical base table (ABT)
  - Compute the two-stage LTV targets per customer:
      Stage 1 — will_return   : 1 if customer places a delivered order in the
                                 prediction window (cutoff → cutoff + horizon_days)
      Stage 2 — future_spend  : total payment value in that window (0 for
                                 non-returners; model is only trained on returners)
  - Split customers into train/test using a chronological cutoff

This module owns the raw data — nothing above this layer should read CSVs directly.
Feature engineering happens in features.py, not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the central YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_tables(raw_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all Olist CSV files from raw_dir.

    Returns a dict keyed by short table name:
        orders, order_items, payments, reviews,
        customers, products, category, sellers, geolocation
    """
    p = Path(raw_dir)
    tables = {
        "orders":      pd.read_csv(p / "olist_orders_dataset.csv"),
        "order_items": pd.read_csv(p / "olist_order_items_dataset.csv"),
        "payments":    pd.read_csv(p / "olist_order_payments_dataset.csv"),
        "reviews":     pd.read_csv(p / "olist_order_reviews_dataset.csv"),
        "customers":   pd.read_csv(p / "olist_customers_dataset.csv"),
        "products":    pd.read_csv(p / "olist_products_dataset.csv"),
        "category":    pd.read_csv(p / "product_category_name_translation.csv"),
        "sellers":     pd.read_csv(p / "olist_sellers_dataset.csv"),
    }

    # Parse all timestamp columns upfront
    for ts_col in ["order_purchase_timestamp", "order_delivered_customer_date",
                   "order_estimated_delivery_date"]:
        if ts_col in tables["orders"].columns:
            tables["orders"][ts_col] = pd.to_datetime(tables["orders"][ts_col])

    print("Loaded tables:")
    for name, df in tables.items():
        print(f"  {name:<15} {df.shape}")

    return tables


def build_analytical_base_table(
    tables: dict[str, pd.DataFrame],
    valid_statuses: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge Olist tables into one row-per-order analytical base table (ABT).

    Pipeline:
      1. Filter orders to valid_statuses (default: delivered only).
      2. Attach customer_unique_id from customers table.
      3. Aggregate order_items per order: total price, freight, item count,
         distinct sellers, distinct product categories.
      4. Aggregate payments per order: total payment value, max installments,
         payment type (voucher flag).
      5. Aggregate reviews per order: mean review score.
      6. Translate product_category_name to English.

    Args:
        tables:         Dict from load_raw_tables().
        valid_statuses: Order statuses to keep (default: ["delivered"]).

    Returns:
        DataFrame with one row per order and all relevant columns joined.
    """
    if valid_statuses is None:
        valid_statuses = ["delivered"]

    orders   = tables["orders"]
    items    = tables["order_items"]
    payments = tables["payments"]
    reviews  = tables["reviews"]
    customers = tables["customers"]
    products  = tables["products"]
    category  = tables["category"]

    # 1. Filter to valid order statuses
    orders = orders[orders["order_status"].isin(valid_statuses)].copy()
    print(f"Orders after status filter: {len(orders):,}")

    # 2. Attach customer_unique_id — this is the stable customer identifier
    #    (customer_id changes per order; customer_unique_id persists across orders)
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left",
    )

    # 3. Aggregate order_items → one row per order
    #    Translate product categories to English first
    products = products.merge(
        category[["product_category_name", "product_category_name_english"]],
        on="product_category_name",
        how="left",
    )
    items = items.merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    items_agg = items.groupby("order_id").agg(
        total_price         = ("price", "sum"),
        total_freight       = ("freight_value", "sum"),
        item_count          = ("order_item_id", "count"),
        unique_sellers      = ("seller_id", "nunique"),
        unique_categories   = ("product_category_name_english", "nunique"),
    ).reset_index()

    # 4. Aggregate payments → one row per order
    #    Some orders use multiple payment methods (e.g. voucher + credit card)
    payments_agg = payments.groupby("order_id").agg(
        payment_value        = ("payment_value", "sum"),
        max_installments     = ("payment_installments", "max"),
        has_voucher          = ("payment_type", lambda x: int("voucher" in x.values)),
    ).reset_index()

    # 5. Aggregate reviews → one row per order (take most recent if duplicates)
    reviews_agg = reviews.sort_values("review_creation_date").groupby("order_id").agg(
        review_score = ("review_score", "last"),
    ).reset_index()

    # 6. Join everything onto orders
    abt = (
        orders
        .merge(items_agg,    on="order_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
        .merge(reviews_agg,  on="order_id", how="left")
    )

    print(f"ABT shape: {abt.shape}")
    print(f"Unique customers in ABT: {abt['customer_unique_id'].nunique():,}")
    return abt


def compute_ltv_targets(
    abt: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = 90,
    date_col: str = "order_purchase_timestamp",
    customer_col: str = "customer_unique_id",
    value_col: str = "payment_value",
) -> pd.DataFrame:
    """
    Compute two-stage LTV targets per customer.

    Stage 1 — will_return (classifier target):
        1 if the customer places at least one order in
        (cutoff_date, cutoff_date + horizon_days], else 0.

    Stage 2 — future_spend (regressor target):
        Sum of payment_value in the prediction window.
        0 for customers who do not return.

    Only customers with at least one order strictly before cutoff_date
    are included — we need observation history to build features.

    Args:
        abt:          Analytical base table (one row per order).
        cutoff_date:  Feature observation cutoff.
        horizon_days: Length of prediction window in days.
        date_col:     Timestamp column for purchase date.
        customer_col: Column identifying the customer.
        value_col:    Column with order payment value.

    Returns:
        DataFrame with one row per customer, columns:
        customer_unique_id, will_return, future_spend.
    """
    horizon_end = cutoff_date + pd.Timedelta(days=horizon_days)

    # Customers who have at least one order before the cutoff
    pre_cutoff = abt[abt[date_col] < cutoff_date]
    eligible   = set(pre_cutoff[customer_col].unique())
    print(f"Eligible customers (have pre-cutoff orders): {len(eligible):,}")

    # Orders in the prediction window
    post = abt[
        (abt[date_col] >= cutoff_date) &
        (abt[date_col] <  horizon_end) &
        (abt[customer_col].isin(eligible))
    ]

    # Aggregate future spend per customer
    future = post.groupby(customer_col)[value_col].sum().reset_index()
    future.columns = [customer_col, "future_spend"]

    # Build targets for all eligible customers — non-returners get 0
    targets = pd.DataFrame({customer_col: list(eligible)})
    targets = targets.merge(future, on=customer_col, how="left")
    targets["future_spend"] = targets["future_spend"].fillna(0.0)
    targets["will_return"]  = (targets["future_spend"] > 0).astype(int)

    returners = targets["will_return"].sum()
    print(f"Returners  (will_return=1): {returners:,} ({returners/len(targets):.1%})")
    print(f"Non-returners (will_return=0): {len(targets)-returners:,}")

    # Stage 2 target: spend tier (tertiles among returners)
    # 0 = Low (bottom third), 1 = Mid, 2 = High (top third)
    # Non-returners get -1 — they are excluded from Stage 2 training
    spend_tier = pd.qcut(
        targets.loc[targets["will_return"] == 1, "future_spend"],
        q=3,
        labels=[0, 1, 2],
    ).astype(float)
    targets["spend_tier"] = -1
    targets.loc[targets["will_return"] == 1, "spend_tier"] = spend_tier.values
    targets["spend_tier"] = targets["spend_tier"].astype(int)

    tier_counts = targets[targets["will_return"] == 1]["spend_tier"].value_counts().sort_index()
    print(f"Spend tiers (returners only): Low={tier_counts.get(0,0)}  Mid={tier_counts.get(1,0)}  High={tier_counts.get(2,0)}")

    return targets


def chronological_split(
    targets: pd.DataFrame,
    abt: pd.DataFrame,
    train_frac: float = 0.80,
    date_col: str = "order_purchase_timestamp",
    customer_col: str = "customer_unique_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split customers into train/test by their first order date.

    The earliest train_frac of customers (ordered by first purchase date)
    go to train; the rest go to test. No random shuffling — preserves
    temporal ordering so test customers are always newer than train customers.

    Args:
        targets:     Customer-level targets from compute_ltv_targets().
        abt:         Analytical base table (for first order date lookup).
        train_frac:  Fraction of customers assigned to train (default 0.80).
        date_col:    Timestamp column for purchase date.
        customer_col: Column identifying the customer.

    Returns:
        (train_targets, test_targets) — both have the same columns as targets.
    """
    # First order date per customer
    first_order = (
        abt.groupby(customer_col)[date_col]
        .min()
        .reset_index()
        .rename(columns={date_col: "first_order_date"})
    )

    targets = targets.merge(first_order, on=customer_col, how="left")
    targets = targets.sort_values("first_order_date")

    cutoff_idx = int(len(targets) * train_frac)
    train = targets.iloc[:cutoff_idx].drop(columns=["first_order_date"])
    test  = targets.iloc[cutoff_idx:].drop(columns=["first_order_date"])

    print(f"Train customers : {len(train):,} | returners: {train['will_return'].sum():,} ({train['will_return'].mean():.1%})")
    print(f"Test  customers : {len(test):,}  | returners: {test['will_return'].sum():,} ({test['will_return'].mean():.1%})")

    return train, test
