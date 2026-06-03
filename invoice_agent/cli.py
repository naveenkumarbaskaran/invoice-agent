"""CLI for invoice-agent.

Usage examples:

    # Process a single invoice
    invoice-agent process invoice.pdf --po-list pos.csv --output result.json

    # Batch-process all invoices in a directory
    invoice-agent batch invoices/ --po-list pos.csv --output-dir results/

    # Batch without a PO list
    invoice-agent batch invoices/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .agent import InvoiceAgent

console = Console()

# Extensions treated as processable invoice files
INVOICE_EXTENSIONS = {".pdf", ".txt", ".text", ".md"}


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="invoice-agent-ai")
def cli() -> None:
    """Invoice-Agent: AI-powered invoice extraction, validation, and PO matching."""


# ---------------------------------------------------------------------------
# process — single invoice
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("invoice", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--po-list",
    "po_csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to a CSV file listing purchase orders for matching.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to write the JSON result (default: <invoice>_result.json).",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var).",
    show_envvar=True,
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress rich output; print only the JSON result to stdout.",
)
def process(
    invoice: Path,
    po_csv: Path | None,
    output_path: Path | None,
    api_key: str | None,
    quiet: bool,
) -> None:
    """Process a single INVOICE file."""
    if not quiet:
        console.print(Panel.fit(
            f"[bold cyan]Processing invoice:[/] {invoice.name}",
            title="invoice-agent",
        ))

    agent = InvoiceAgent(api_key=api_key)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        disable=quiet,
        transient=True,
    ) as progress:
        task = progress.add_task("Running agent...", total=None)
        result = agent.process(
            invoice_path=invoice,
            po_csv_path=po_csv,
            output_path=output_path,
        )
        progress.update(task, completed=True)

    if quiet:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    _render_result(result, invoice.name)


# ---------------------------------------------------------------------------
# batch — directory of invoices
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("invoices_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--po-list",
    "po_csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to a CSV file listing purchase orders for matching.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for result JSON files (default: same dir as each invoice).",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var).",
    show_envvar=True,
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on the first error instead of continuing.",
)
def batch(
    invoices_dir: Path,
    po_csv: Path | None,
    output_dir: Path | None,
    api_key: str | None,
    fail_fast: bool,
) -> None:
    """Batch-process all invoice files in INVOICES_DIR."""
    invoice_files = sorted(
        f for f in invoices_dir.iterdir()
        if f.is_file() and f.suffix.lower() in INVOICE_EXTENSIONS
    )

    if not invoice_files:
        console.print(f"[yellow]No invoice files found in {invoices_dir}[/yellow]")
        sys.exit(0)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    agent = InvoiceAgent(api_key=api_key)
    errors: list[tuple[Path, str]] = []
    successes: list[tuple[Path, dict]] = []

    console.print(Panel.fit(
        f"[bold cyan]Batch processing {len(invoice_files)} invoice(s)[/] in [dim]{invoices_dir}[/]",
        title="invoice-agent batch",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing invoices...", total=len(invoice_files))

        for invoice_file in invoice_files:
            progress.update(task, description=f"[cyan]{invoice_file.name}")
            out_path: Path | None = None
            if output_dir:
                out_path = output_dir / (invoice_file.stem + "_result.json")
            try:
                result = agent.process(
                    invoice_path=invoice_file,
                    po_csv_path=po_csv,
                    output_path=out_path,
                )
                successes.append((invoice_file, result))
            except Exception as exc:  # noqa: BLE001
                errors.append((invoice_file, str(exc)))
                if fail_fast:
                    console.print(f"[red]Failed on {invoice_file.name}: {exc}[/red]")
                    progress.stop()
                    sys.exit(1)
            finally:
                progress.advance(task)

    # Summary
    _render_batch_summary(successes, errors)
    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_result(result: dict, filename: str) -> None:
    """Render a single invoice result to the console."""
    # Fields table
    fields_data: dict = result.get("fields", result)  # support flat or nested result
    table = Table(title=f"Extracted Fields — {filename}", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    display_keys = [
        "invoice_number", "vendor_name", "invoice_date", "due_date",
        "po_number", "subtotal", "tax_rate", "tax_amount", "total_amount", "currency",
    ]
    for key in display_keys:
        val = fields_data.get(key)
        if val is None:
            display = "[dim]—[/dim]"
        elif isinstance(val, float):
            display = f"{val:,.2f}"
        else:
            display = str(val)
        table.add_row(key.replace("_", " ").title(), display)

    console.print(table)

    # Line items
    items = fields_data.get("line_items", [])
    if items:
        li_table = Table(title="Line Items", show_header=True, header_style="bold blue")
        li_table.add_column("Description")
        li_table.add_column("Qty", justify="right")
        li_table.add_column("Unit Price", justify="right")
        li_table.add_column("Total", justify="right")
        for item in items:
            li_table.add_row(
                str(item.get("description", "")),
                _fmt_num(item.get("quantity")),
                _fmt_money(item.get("unit_price")),
                _fmt_money(item.get("total")),
            )
        console.print(li_table)

    # Validation errors
    validation_errors = result.get("validation_errors", [])
    if validation_errors:
        console.print("[bold red]Validation Errors:[/bold red]")
        for err in validation_errors:
            console.print(f"  [red]x[/red] {err}")
    else:
        console.print("[green]Totals validated successfully.[/green]")

    # PO match
    po_match = result.get("po_match")
    if po_match:
        status = po_match.get("status", "unknown")
        color = "green" if status == "matched" else "yellow" if status == "partial" else "red"
        console.print(f"PO Match: [{color}]{status}[/{color}]")
        notes = po_match.get("notes", [])
        for note in notes:
            console.print(f"  [dim]{note}[/dim]")
    elif result.get("po_number"):
        console.print("[yellow]No PO list provided; PO matching skipped.[/yellow]")

    # Anomalies
    anomalies = result.get("anomalies", [])
    if anomalies:
        console.print("[bold yellow]Anomalies / Flags:[/bold yellow]")
        for anomaly in anomalies:
            console.print(f"  [yellow]![/yellow] {anomaly}")
    else:
        console.print("[green]No anomalies detected.[/green]")

    # Output path hint
    out = result.get("output_path") or result.get("_output_path")
    if out:
        console.print(f"\n[dim]Result saved to: {out}[/dim]")


def _render_batch_summary(
    successes: list[tuple[Path, dict]],
    errors: list[tuple[Path, str]],
) -> None:
    console.rule("Batch Summary")
    console.print(f"[green]Processed:[/green] {len(successes)}  [red]Errors:[/red] {len(errors)}")

    if successes:
        t = Table(show_header=True, header_style="bold")
        t.add_column("File", style="cyan")
        t.add_column("Vendor")
        t.add_column("Invoice #")
        t.add_column("Total", justify="right")
        t.add_column("Anomalies", justify="right")
        for path, res in successes:
            fields = res.get("fields", res)
            anomaly_count = len(res.get("anomalies", []))
            color = "yellow" if anomaly_count else "white"
            t.add_row(
                path.name,
                str(fields.get("vendor_name") or "—"),
                str(fields.get("invoice_number") or "—"),
                _fmt_money(fields.get("total_amount")),
                f"[{color}]{anomaly_count}[/{color}]",
            )
        console.print(t)

    if errors:
        console.print("[bold red]Errors:[/bold red]")
        for path, msg in errors:
            console.print(f"  [red]x[/red] {path.name}: {msg}")


def _fmt_num(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):g}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _fmt_money(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.2f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
