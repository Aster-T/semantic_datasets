"""CLI entry point for semantic-datasets."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.registry import DownloaderRegistry

console = Console()


def _build_registry(data_dir: str, sources: tuple[str, ...]) -> DownloaderRegistry:
    """Build a registry, optionally limiting to specific sources."""
    registry = DownloaderRegistry(data_dir=data_dir)

    if not sources:
        registry.register_defaults()
    else:
        # Lazy-import only the requested sources
        from src.kaggle_downloader import KaggleDownloader
        from src.openml_downloader import OpenMLDownloader

        source_map = {
            "openml": lambda: OpenMLDownloader(data_dir=data_dir),
            "kaggle": lambda: KaggleDownloader(data_dir=data_dir),
        }
        for s in sources:
            if s not in source_map:
                raise click.BadParameter(
                    f"Unknown source '{s}'. Available: {', '.join(source_map)}"
                )
            registry.register(source_map[s]())

    return registry


@click.group()
def cli():
    """semantic-datasets: download datasets from OpenML, Kaggle and more."""


@cli.command()
@click.argument("query")
@click.option("-s", "--source", multiple=True, help="Limit to specific sources")
@click.option("-n", "--max-results", default=20, help="Max results per source")
@click.option("-d", "--data-dir", default="./data", help="Root data directory")
def search(query: str, source: tuple[str, ...], max_results: int, data_dir: str):
    """Search for datasets across all sources."""
    registry = _build_registry(data_dir, source)

    with console.status(f"Searching for '{query}'..."):
        results = registry.search(query, max_results=max_results)

    if not results:
        console.print("[yellow]No datasets found.[/yellow]")
        return

    table = Table(title=f"Search results for '{query}'")
    table.add_column("Source", style="cyan")
    table.add_column("ID", style="green")
    table.add_column("Name", style="bold")
    table.add_column("Description", max_width=50)

    for ds in results:
        table.add_row(ds.source, ds.dataset_id, ds.name, ds.description[:100])

    console.print(table)


@cli.command()
@click.argument("full_id")
@click.option("-d", "--data-dir", default="./data", help="Root data directory")
@click.option("-o", "--output", default=None, help="Override output directory")
def download(full_id: str, data_dir: str, output: str | None):
    """Download a dataset by full ID (e.g. openml/61, kaggle/zillow/zecon)."""
    registry = _build_registry(data_dir, ())
    dest = Path(output) if output else None

    with console.status(f"Downloading {full_id}..."):
        path = registry.download(full_id, dest_dir=dest)

    console.print(f"[green]Downloaded to:[/green] {path}")


@cli.command()
@click.argument("full_id")
@click.option("-d", "--data-dir", default="./data", help="Root data directory")
def info(full_id: str, data_dir: str):
    """Show metadata for a dataset."""
    registry = _build_registry(data_dir, ())

    with console.status(f"Fetching info for {full_id}..."):
        ds = registry.info(full_id)

    console.print(f"[bold]{ds.name}[/bold]  ({ds.full_id})")
    console.print(f"  Description: {ds.description[:300]}")
    if ds.tags:
        console.print(f"  Tags: {', '.join(ds.tags)}")
    if ds.url:
        console.print(f"  URL: {ds.url}")
    if ds.extra:
        for k, v in ds.extra.items():
            console.print(f"  {k}: {v}")


@cli.command(name="list")
@click.option("-s", "--source", required=True, help="Source to list from")
@click.option("-n", "--max-results", default=50, help="Max results")
@click.option("-d", "--data-dir", default="./data", help="Root data directory")
def list_datasets(source: str, max_results: int, data_dir: str):
    """List datasets from a specific source."""
    registry = _build_registry(data_dir, (source,))

    with console.status(f"Listing datasets from {source}..."):
        results = registry.get(source).list_datasets(max_results=max_results)

    table = Table(title=f"Datasets from {source}")
    table.add_column("ID", style="green")
    table.add_column("Name", style="bold")
    table.add_column("Description", max_width=60)

    for ds in results:
        table.add_row(ds.dataset_id, ds.name, ds.description[:100])

    console.print(table)


if __name__ == "__main__":
    cli()
