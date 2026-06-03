"""InvoiceAgent: Anthropic-powered invoice processing agent.

Uses claude-sonnet-4-6 with a manual tool-use loop to:
  1. Read invoice PDFs (or plain-text invoices)
  2. Read PO CSV files for matching
  3. Extract structured fields via LLM + regex hybrid
  4. Validate arithmetic totals
  5. Match invoices against POs
  6. Flag anomalies and discrepancies
  7. Write JSON results
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import anthropic

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from .extractor import FieldExtractor, InvoiceFields

MODEL = "claude-sonnet-4-6"

# System prompt that describes the agent's role and the JSON schema it must use
SYSTEM_PROMPT = """You are an expert invoice processing agent. When asked to analyse an invoice, you:

1. Call read_pdf_text or read_plain_text to obtain the invoice text.
2. Call read_csv if a PO (purchase-order) CSV list is supplied.
3. Extract ALL of the following fields from the invoice text into a single JSON object:
   {
     "invoice_number": "string or null",
     "vendor_name": "string or null",
     "vendor_address": "string or null",
     "invoice_date": "YYYY-MM-DD or original text or null",
     "due_date": "YYYY-MM-DD or original text or null",
     "po_number": "string or null",
     "subtotal": number_or_null,
     "tax_rate": number_or_null,
     "tax_amount": number_or_null,
     "total_amount": number_or_null,
     "currency": "USD",
     "line_items": [
       {"description": "...", "quantity": number_or_null, "unit_price": number_or_null, "total": number_or_null}
     ]
   }
4. Validate arithmetic (subtotal + tax = total, line items sum to subtotal).
5. If a PO list was provided, look up the PO number and compare vendor, amounts, and line items.
6. List every anomaly, discrepancy, or missing field.
7. Call write_file to persist the final JSON result.

Be thorough. Never skip a field that is present in the text. When a value is ambiguous, record both interpretations in the anomalies list.
"""


class InvoiceAgent:
    """Orchestrates multi-turn tool-use to process a single invoice."""

    def __init__(self, api_key: str | None = None) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.extractor = FieldExtractor()
        self._tools = self._build_tools()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        invoice_path: str | Path,
        po_csv_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Process one invoice and return the result dict.

        Args:
            invoice_path: Path to the invoice (PDF or text file).
            po_csv_path:  Optional path to a CSV of purchase orders.
            output_path:  Optional path where the JSON result will be written.
                          Defaults to ``<invoice_stem>_result.json`` in the same dir.

        Returns:
            A dictionary with keys: fields, validation_errors, po_match,
            anomalies, and raw_extraction.
        """
        invoice_path = Path(invoice_path)
        if output_path is None:
            output_path = invoice_path.with_name(invoice_path.stem + "_result.json")
        output_path = Path(output_path)

        # Build the initial user message
        parts: list[str] = [f"Process the invoice at: {invoice_path}"]
        if po_csv_path:
            parts.append(f"The PO list CSV is at: {po_csv_path}")
        parts.append(f"Write the final JSON result to: {output_path}")
        user_message = "\n".join(parts)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        result: dict[str, Any] = {}
        written_path: str | None = None

        # Agentic loop — run until end_turn or we have a written result
        for _iteration in range(20):  # safety cap
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self._tools,
                messages=messages,
            )

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract any JSON block from the final text
                for block in response.content:
                    if hasattr(block, "text"):
                        extracted = self._extract_json_from_text(block.text)
                        if extracted:
                            result = extracted
                break

            if response.stop_reason != "tool_use":
                break

            # Execute tool calls and gather results
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_output = self._dispatch_tool(block.name, block.input)

                # If the agent wrote the file, capture the JSON for return
                if block.name == "write_file" and not written_path:
                    written_path = block.input.get("path", "")
                    try:
                        result = json.loads(block.input.get("content", "{}"))
                    except json.JSONDecodeError:
                        pass

                # If we got invoice JSON back from an extraction tool, run
                # the hybrid extractor as a post-processing step
                if block.name in ("read_pdf_text", "read_plain_text"):
                    pdf_text = tool_output
                    regex_fields = self.extractor.extract_with_regex(pdf_text)
                    # Store on instance so we can merge later
                    self._last_regex_fields = regex_fields

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_output,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        # If the agent returned a valid result via write_file we already have
        # it; otherwise try to build one from the last text block.
        if not result and written_path and Path(written_path).exists():
            with open(written_path) as fh:
                try:
                    result = json.load(fh)
                except json.JSONDecodeError:
                    pass

        # Augment with our local validation if the LLM didn't include it
        result = self._ensure_validation_section(result)

        # Persist result if not already written
        if not written_path or not Path(written_path).exists():
            output_path.write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )

        return result

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tools() -> list[dict[str, Any]]:
        return [
            {
                "name": "read_pdf_text",
                "description": (
                    "Extract the full text content from a PDF invoice file. "
                    "Returns the extracted text as a string. "
                    "Use this before attempting field extraction."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative filesystem path to the PDF file.",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "read_csv",
                "description": (
                    "Read a CSV file and return its contents as a JSON array of objects. "
                    "Use this to load the purchase-order (PO) list for matching."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative filesystem path to the CSV file.",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": (
                    "Write text content to a file on the filesystem. "
                    "Use this to persist the final JSON result."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative filesystem path to write to.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The text content to write.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, inputs: dict[str, Any]) -> str:
        try:
            if name == "read_pdf_text":
                return self._read_pdf_text(inputs["path"])
            if name == "read_csv":
                return self._read_csv(inputs["path"])
            if name == "write_file":
                return self._write_file(inputs["path"], inputs["content"])
            return f"Unknown tool: {name}"
        except Exception as exc:  # noqa: BLE001
            return f"Error executing {name}: {exc}"

    @staticmethod
    def _read_pdf_text(path: str) -> str:
        """Extract text from a PDF using pypdf, or read as plain text."""
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"

        if p.suffix.lower() == ".pdf":
            if not HAS_PYPDF:
                return "pypdf is not installed. Install it with: pip install pypdf"
            try:
                reader = pypdf.PdfReader(str(p))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages).strip()
                return text if text else f"Could not extract text from {path}"
            except Exception as exc:  # noqa: BLE001
                return f"Error reading PDF {path}: {exc}"

        # Treat as plain text
        return p.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_csv(path: str) -> str:
        """Read a CSV and return a JSON array of row dicts."""
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            rows: list[dict[str, str]] = []
            with open(p, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append(dict(row))
            return json.dumps(rows, indent=2)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)})

    @staticmethod
    def _write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed."""
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as exc:  # noqa: BLE001
            return f"Error writing {path}: {exc}"

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _ensure_validation_section(self, result: dict[str, Any]) -> dict[str, Any]:
        """Add local validation results if the LLM didn't include them."""
        if "validation_errors" in result:
            return result

        # Try to build InvoiceFields from the result dict for local validation
        try:
            fields = InvoiceFields(
                invoice_number=result.get("invoice_number") or result.get("fields", {}).get("invoice_number"),
                vendor_name=result.get("vendor_name") or result.get("fields", {}).get("vendor_name"),
                invoice_date=result.get("invoice_date") or result.get("fields", {}).get("invoice_date"),
                total_amount=self._to_float(result.get("total_amount") or (result.get("fields") or {}).get("total_amount")),
                subtotal=self._to_float(result.get("subtotal") or (result.get("fields") or {}).get("subtotal")),
                tax_amount=self._to_float(result.get("tax_amount") or (result.get("fields") or {}).get("tax_amount")),
                tax_rate=self._to_float(result.get("tax_rate") or (result.get("fields") or {}).get("tax_rate")),
            )
            validation_errors = self.extractor.validate_totals(fields)
            if validation_errors:
                result.setdefault("anomalies", [])
                if isinstance(result["anomalies"], list):
                    result["anomalies"].extend(validation_errors)
                result["validation_errors"] = validation_errors
        except Exception:  # noqa: BLE001
            pass

        return result

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).replace(",", ""))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any]:
        """Salvage the first JSON object from a text block."""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                # Try with markdown fence stripped
                cleaned = re.sub(r"```(?:json)?|```", "", text[start:end])
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
        return {}
