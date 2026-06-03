"""FieldExtractor: regex + LLM hybrid for structured invoice field extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass
class LineItem:
    description: str
    quantity: float | None
    unit_price: float | None
    total: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total": self.total,
        }


@dataclass
class InvoiceFields:
    invoice_number: str | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    po_number: str | None = None
    subtotal: float | None = None
    tax_rate: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    currency: str = "USD"
    line_items: list[LineItem] = field(default_factory=list)
    raw_text: str = ""
    extraction_method: str = "unknown"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_number": self.invoice_number,
            "vendor_name": self.vendor_name,
            "vendor_address": self.vendor_address,
            "invoice_date": self.invoice_date,
            "due_date": self.due_date,
            "po_number": self.po_number,
            "subtotal": self.subtotal,
            "tax_rate": self.tax_rate,
            "tax_amount": self.tax_amount,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "line_items": [li.to_dict() for li in self.line_items],
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
        }


class FieldExtractor:
    """Hybrid regex + LLM field extractor for invoice text.

    Attempts regex patterns first for speed and determinism, then
    uses an LLM result (provided as a JSON string) to fill any gaps
    and validate the regex findings.
    """

    # -----------------------------------------------------------------
    # Regex patterns
    # -----------------------------------------------------------------
    _INVOICE_NUM_RE = re.compile(
        r"(?:invoice|inv)[\s#:.\-]*([A-Z0-9\-]+)",
        re.IGNORECASE,
    )
    _PO_NUM_RE = re.compile(
        r"(?:purchase\s+order|p\.?o\.?)[\s#:.\-]*([A-Z0-9\-]+)",
        re.IGNORECASE,
    )
    _DATE_RE = re.compile(
        r"(?:invoice\s+date|date)[\s:]*([\d]{1,2}[/\-][\d]{1,2}[/\-][\d]{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    )
    _DUE_DATE_RE = re.compile(
        r"(?:due\s+date|payment\s+due)[\s:]*([\d]{1,2}[/\-][\d]{1,2}[/\-][\d]{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    )
    _TOTAL_RE = re.compile(
        r"(?:total\s+(?:amount\s+)?due|amount\s+due|grand\s+total|total)[\s:$]*([\d,]+\.\d{2})",
        re.IGNORECASE,
    )
    _SUBTOTAL_RE = re.compile(
        r"(?:subtotal|sub-total|sub\s+total)[\s:$]*([\d,]+\.\d{2})",
        re.IGNORECASE,
    )
    _TAX_RE = re.compile(
        r"(?:tax|vat|gst)[\s@%:(]*([\d]+(?:\.\d+)?)[\s%]*\)?[\s:$]*([\d,]+\.\d{2})?",
        re.IGNORECASE,
    )
    _CURRENCY_RE = re.compile(
        r"\b(USD|EUR|GBP|CAD|AUD|JPY)\b|([€£¥$])",
        re.IGNORECASE,
    )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def extract_with_regex(self, text: str) -> InvoiceFields:
        """Extract fields using only regex patterns. Fast, deterministic."""
        fields = InvoiceFields(raw_text=text, extraction_method="regex")

        if m := self._INVOICE_NUM_RE.search(text):
            fields.invoice_number = m.group(1).strip()

        if m := self._PO_NUM_RE.search(text):
            fields.po_number = m.group(1).strip()

        if m := self._DATE_RE.search(text):
            fields.invoice_date = m.group(1).strip()

        if m := self._DUE_DATE_RE.search(text):
            fields.due_date = m.group(1).strip()

        if m := self._TOTAL_RE.search(text):
            fields.total_amount = self._parse_amount(m.group(1))

        if m := self._SUBTOTAL_RE.search(text):
            fields.subtotal = self._parse_amount(m.group(1))

        if m := self._TAX_RE.search(text):
            try:
                fields.tax_rate = float(m.group(1))
            except (TypeError, ValueError):
                pass
            if m.group(2):
                fields.tax_amount = self._parse_amount(m.group(2))

        if m := self._CURRENCY_RE.search(text):
            symbol_map = {"€": "EUR", "£": "GBP", "¥": "JPY", "$": "USD"}
            raw = (m.group(1) or m.group(2) or "USD").upper()
            fields.currency = symbol_map.get(raw, raw)

        # Confidence: count non-None critical fields
        critical = [fields.invoice_number, fields.vendor_name, fields.total_amount, fields.invoice_date]
        fields.confidence = sum(1 for f in critical if f is not None) / len(critical)

        return fields

    def merge_llm_result(self, regex_fields: InvoiceFields, llm_json: str) -> InvoiceFields:
        """Merge LLM-extracted JSON into an existing InvoiceFields, filling gaps.

        The LLM result is authoritative for vendor info and line items.
        Numeric fields from regex are preferred when the LLM disagrees by less
        than 1 cent (floating-point noise); otherwise LLM wins.
        """
        try:
            data: dict[str, Any] = json.loads(llm_json)
        except json.JSONDecodeError:
            # Try to salvage JSON from inside a larger text block
            data = self._extract_json_from_text(llm_json)

        merged = InvoiceFields(
            raw_text=regex_fields.raw_text,
            extraction_method="hybrid",
        )

        # String fields: prefer LLM (richer context)
        merged.invoice_number = data.get("invoice_number") or regex_fields.invoice_number
        merged.vendor_name = data.get("vendor_name") or regex_fields.vendor_name
        merged.vendor_address = data.get("vendor_address") or regex_fields.vendor_address
        merged.invoice_date = data.get("invoice_date") or regex_fields.invoice_date
        merged.due_date = data.get("due_date") or regex_fields.due_date
        merged.po_number = data.get("po_number") or regex_fields.po_number
        merged.currency = data.get("currency") or regex_fields.currency or "USD"

        # Numeric fields: prefer non-None, use LLM as fallback
        merged.total_amount = self._merge_amount(
            regex_fields.total_amount, data.get("total_amount")
        )
        merged.subtotal = self._merge_amount(
            regex_fields.subtotal, data.get("subtotal")
        )
        merged.tax_amount = self._merge_amount(
            regex_fields.tax_amount, data.get("tax_amount")
        )
        merged.tax_rate = regex_fields.tax_rate or self._to_float(data.get("tax_rate"))

        # Line items: always from LLM (regex can't reliably parse tables)
        raw_items = data.get("line_items", [])
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                merged.line_items.append(
                    LineItem(
                        description=str(item.get("description", "")),
                        quantity=self._to_float(item.get("quantity")),
                        unit_price=self._to_float(item.get("unit_price")),
                        total=self._to_float(item.get("total")),
                    )
                )

        # Confidence: count non-None critical fields
        critical = [merged.invoice_number, merged.vendor_name, merged.total_amount, merged.invoice_date]
        merged.confidence = sum(1 for f in critical if f is not None) / len(critical)

        return merged

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    def validate_totals(self, fields: InvoiceFields) -> list[str]:
        """Check arithmetic consistency of extracted totals.

        Returns a list of discrepancy messages (empty means clean).
        """
        issues: list[str] = []

        if fields.total_amount is None:
            issues.append("Missing total_amount — cannot validate totals.")
            return issues

        # Check subtotal + tax = total
        if fields.subtotal is not None and fields.tax_amount is not None:
            computed = round(fields.subtotal + fields.tax_amount, 2)
            if abs(computed - fields.total_amount) > 0.02:
                issues.append(
                    f"Total mismatch: subtotal {fields.subtotal:.2f} + tax {fields.tax_amount:.2f} "
                    f"= {computed:.2f} but invoice total is {fields.total_amount:.2f}."
                )

        # Check tax_rate consistency
        if fields.tax_rate is not None and fields.subtotal is not None and fields.tax_amount is not None:
            expected_tax = round(fields.subtotal * fields.tax_rate / 100, 2)
            if abs(expected_tax - fields.tax_amount) > 0.05:  # 5-cent tolerance
                issues.append(
                    f"Tax rate inconsistency: {fields.tax_rate}% of {fields.subtotal:.2f} "
                    f"= {expected_tax:.2f} but tax_amount is {fields.tax_amount:.2f}."
                )

        # Check line items sum to subtotal
        if fields.line_items and fields.subtotal is not None:
            line_total = sum(
                (li.total or 0.0) for li in fields.line_items
            )
            if line_total > 0 and abs(line_total - fields.subtotal) > 0.02:
                issues.append(
                    f"Line items sum ({line_total:.2f}) does not match subtotal ({fields.subtotal:.2f})."
                )

        return issues

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_amount(raw: str) -> float | None:
        try:
            return float(raw.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return None

    @classmethod
    def _merge_amount(cls, regex_val: float | None, llm_val: Any) -> float | None:
        llm_float = cls._to_float(llm_val)
        if regex_val is None:
            return llm_float
        if llm_float is None:
            return regex_val
        # If they're close (< 1 cent), prefer regex (deterministic)
        if abs(regex_val - llm_float) < 0.01:
            return regex_val
        # LLM wins on disagreement (it has fuller context)
        return llm_float

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any]:
        """Salvage the first JSON object found in a larger text string."""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {}
