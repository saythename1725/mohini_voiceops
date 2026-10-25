"""Pandas-backed loader for the synthetic CSV tables; identifiers stay strings."""

from pathlib import Path

import pandas as pd


class DataStore:
    TABLES = ("orders", "customers", "order_items", "products", "pincodes", "holidays")

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.frames = {
            name: pd.read_csv(self.data_dir / f"{name}.csv", dtype=str,
                              keep_default_na=False, encoding="utf-8-sig")
            for name in self.TABLES
        }
        self.orders = self._index("orders", "order_id")
        self.customers = self._index("customers", "customer_id")
        self.products = self._index("products", "sku")
        self.pincodes = self._index("pincodes", "pincode")
        self.order_items = self.frames["order_items"].to_dict(orient="records")

    def _index(self, table: str, key: str) -> dict[str, dict[str, str]]:
        frame = self.frames[table]
        return {str(row[key]): row for row in frame.to_dict(orient="records")}

    def order_items_for(self, order_id: str) -> list[dict[str, str]]:
        """Join order lines with product catalog metadata, preserving text IDs."""
        related = []
        for item in self.order_items:
            if item["order_id"] != str(order_id):
                continue
            product = self.products.get(item["sku"], {})
            related.append({**item, "product": product})
        return related

    def order_details(self, order_id: str) -> dict | None:
        """Return a joined synthetic order view; caller must enforce verification."""
        order = self.orders.get(str(order_id))
        if not order:
            return None
        customer = self.customers.get(order.get("customer_id", ""), {})
        pincode = self.pincodes.get(order.get("delivery_pincode", ""), {})
        return {"order": order, "customer": customer,
                "items": self.order_items_for(str(order_id)), "delivery_area": pincode}

    def product(self, sku: str) -> dict[str, str] | None:
        return self.products.get(str(sku))

    def product_named_in(self, text: str) -> dict[str, str] | None:
        """Resolve a catalog row from a distinctive title phrase, if variants agree on key facts."""
        matches = self.products_named_in(text)
        return matches[0] if matches else None

    def products_named_in(self, text: str,
                          stable_fields: tuple[str, ...] = ("return_category", "assured", "warranty_months")) -> list[dict[str, str]]:
        """Return equally plausible title variants when their policy-relevant fields agree."""
        tokens = lambda value: set("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())
        query_tokens = tokens(text)
        ranked = []
        for row in self.products.values():
            title_tokens = tokens(row.get("title", ""))
            if not title_tokens:
                continue
            overlap = len(title_tokens & query_tokens) / len(title_tokens)
            if overlap >= 0.8:
                ranked.append((overlap, len(title_tokens), row))
        if not ranked:
            return []
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, best_length, best = ranked[0]
        tied = [row for score, length, row in ranked
                if score == best_score and length == best_length]
        # An omitted color can leave multiple valid rows. Retain all of them only if
        # the policy facts used by the agent agree across variants.
        if any(any(row.get(field) != best.get(field) for field in stable_fields) for row in tied[1:]):
            return []
        return tied
