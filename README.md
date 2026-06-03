# invoice-agent-ai

An AI-powered invoice processing agent built with the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) and Claude (`claude-sonnet-4-6`).

Capabilities:
- Extract structured fields from PDF and text invoices (vendor, date, amounts, line items, tax)
- Validate arithmetic totals (subtotal + tax = total, line items sum, tax-rate consistency)
- Match invoices against a CSV of purchase orders
- Flag discrepancies and anomalies
- Write results as JSON
- Process single invoices or entire directories in batch

---

## Installation

```bash
pip install invoice-agent-ai
```

Or, from source:

```bash
git clone https://github.com/example/invoice-agent-ai
cd invoice-agent-ai
pip install -e .
```

Requires Python 3.11+.

---

## Configuration

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or pass it via `--api-key` on every command.

---

## CLI Usage

### Process a single invoice

```bash
invoice-agent process invoice.pdf --po-list pos.csv --output result.json
```

| Option | Description |
|---|---|
| `INVOICE` | Path to the invoice file (PDF or plain text) |
| `--po-list PATH` | Path to a PO CSV file for matching (optional) |
| `--output PATH` | Where to write the JSON result (default: `<invoice>_result.json`) |
| `--api-key KEY` | Anthropic API key (defaults to `ANTHROPIC_API_KEY` env var) |
| `--quiet` / `-q` | Suppress rich output; print raw JSON to stdout |

### Batch-process a directory

```bash
invoice-agent batch invoices/ --po-list pos.csv --output-dir results/
```

| Option | Description |
|---|---|
| `INVOICES_DIR` | Directory containing invoice files |
| `--po-list PATH` | Shared PO CSV (optional) |
| `--output-dir DIR` | Directory for result files (default: same as each invoice) |
| `--fail-fast` | Stop on first error |

---

## PO CSV Format

The purchase-order CSV must have at minimum these columns (column names are case-insensitive):

```csv
po_number,vendor_name,amount,currency,description
PO-1001,Acme Corp,1500.00,USD,Office Supplies Q1
PO-1002,Widget LLC,4250.75,USD,Server Hardware
```

Any additional columns are passed to the LLM for context.

---

## Output JSON Schema

Each processed invoice produces a JSON file with the following top-level keys:

```json
{
  "invoice_number": "INV-2024-001",
  "vendor_name": "Acme Corp",
  "vendor_address": "123 Main St, Springfield, IL",
  "invoice_date": "2024-03-15",
  "due_date": "2024-04-14",
  "po_number": "PO-1001",
  "subtotal": 1388.89,
  "tax_rate": 8.0,
  "tax_amount": 111.11,
  "total_amount": 1500.00,
  "currency": "USD",
  "line_items": [
    {
      "description": "Widget Type A",
      "quantity": 10,
      "unit_price": 99.99,
      "total": 999.90
    }
  ],
  "validation_errors": [],
  "po_match": {
    "status": "matched",
    "po_number": "PO-1001",
    "notes": ["Amount matches PO within tolerance."]
  },
  "anomalies": []
}
```

### `po_match.status` values

| Value | Meaning |
|---|---|
| `matched` | PO found; vendor and amount agree |
| `partial` | PO found but one or more fields differ |
| `not_found` | No PO with the given number in the CSV |
| `no_po_number` | Invoice contains no PO number |
| `skipped` | No PO CSV was provided |

---

## Python API

```python
from invoice_agent import InvoiceAgent

agent = InvoiceAgent()  # reads ANTHROPIC_API_KEY from env

result = agent.process(
    invoice_path="invoice.pdf",
    po_csv_path="pos.csv",     # optional
    output_path="result.json",  # optional
)

print(result["vendor_name"])
print(result["total_amount"])
print(result["anomalies"])
```

### FieldExtractor (standalone)

The `FieldExtractor` can be used independently of the agent:

```python
from invoice_agent.extractor import FieldExtractor

extractor = FieldExtractor()

# Fast regex-only extraction
fields = extractor.extract_with_regex(invoice_text)

# Validate totals
errors = extractor.validate_totals(fields)

# Merge LLM JSON result into regex findings
merged = extractor.merge_llm_result(fields, llm_json_string)
```

---

## Architecture

```
invoice_agent/
  __init__.py       Re-exports InvoiceAgent and FieldExtractor
  agent.py          InvoiceAgent — Anthropic SDK agentic loop with tools:
                      read_pdf_text, read_csv, write_file
  extractor.py      FieldExtractor — regex + LLM hybrid extraction and
                      arithmetic validation
  cli.py            Click CLI — process and batch sub-commands
```

### How the agent loop works

1. The user message tells the agent which invoice file and PO CSV to process.
2. The agent calls `read_pdf_text` to extract raw text from the PDF.
3. Concurrently, `FieldExtractor.extract_with_regex` runs locally for a fast first pass.
4. If a PO CSV was specified the agent calls `read_csv`.
5. The LLM analyses the text, produces a structured JSON result, validates totals, and performs PO matching.
6. The LLM calls `write_file` to persist the result.
7. The Python layer merges the regex findings with the LLM JSON and runs a final local validation pass.

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy invoice_agent/
```

---

## License

MIT
