"""
AI-COS CLI — start the gateway, inspect stats, manage memory, and more.

Commands:
  aicos start       Launch the OpenAI-compatible gateway
  aicos chat        Interactive chat with AI-COS
  aicos remember    Store a memory
  aicos forget      Delete a memory
  aicos search      Search memories
  aicos stats       Show usage statistics
  aicos config      Show current configuration
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="aicos",
    help="AI-COS: The operating system between applications and AI",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

BANNER = """
[bold cyan]  █████╗ ██╗      ██████╗ ██████╗ ███████╗
 ██╔══██╗██║     ██╔════╝██╔═══██╗██╔════╝
 ███████║██║     ██║     ██║   ██║███████╗
 ██╔══██║██║     ██║     ██║   ██║╚════██║
 ██║  ██║██║     ╚██████╗╚██████╔╝███████║
 ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═════╝ ╚══════╝[/bold cyan]

[dim]The operating system between applications and AI[/dim]
[dim]v0.2.0 · github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-[/dim]
"""


def _print_banner() -> None:
    console.print(Panel(BANNER, border_style="cyan", padding=(0, 2)))


# ── start ─────────────────────────────────────────────────────────────────────

@app.command()
def start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Gateway host"),
    port: int = typer.Option(4000, "--port", "-p", help="Gateway port"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Auto-reload on changes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help="Path to .env file"),
) -> None:
    """Start the AI-COS gateway server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install aicos[/red]")
        raise typer.Exit(1)

    if env_file:
        from dotenv import load_dotenv
        load_dotenv(env_file)

    _print_banner()

    from aicos.core.config import get_config
    cfg = get_config()

    # Show configuration table
    config_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value")

    providers = cfg.available_providers()
    config_table.add_row("Gateway", f"http://{host}:{port}")
    config_table.add_row("Providers", ", ".join(providers) if providers else "[red]None configured[/red]")
    config_table.add_row("Router Strategy", cfg.router_strategy)
    config_table.add_row("Semantic Cache", "[green]enabled[/green]" if cfg.cache_enabled else "[red]disabled[/red]")
    config_table.add_row("Memory", "[green]enabled[/green]" if cfg.memory_enabled else "[red]disabled[/red]")
    config_table.add_row("Context Compression", "[green]enabled[/green]" if cfg.context_compression_enabled else "[red]disabled[/red]")
    config_table.add_row("Max Context Tokens", str(cfg.max_context_tokens))
    config_table.add_row("DB Path", cfg.db_path)

    console.print(config_table)

    if not providers or providers == ["ollama"]:
        console.print(
            Panel(
                "[yellow]Warning:[/yellow] No cloud API keys configured.\n"
                "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY in your .env file.\n"
                "Ollama is available if running locally.",
                title="[yellow]Configuration Notice[/yellow]",
                border_style="yellow",
            )
        )

    console.print(f"\n[green]Starting AI-COS gateway on http://{host}:{port}[/green]")
    console.print("[dim]OpenAI-compatible: POST /v1/chat/completions[/dim]")
    console.print("[dim]Metrics:           GET  /metrics[/dim]")
    console.print("[dim]Health:            GET  /health[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    from aicos.api.routes import create_app
    gateway_app = create_app(cfg)

    uvicorn.run(
        gateway_app,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="debug" if verbose else "info",
        access_log=verbose,
    )


# ── chat ──────────────────────────────────────────────────────────────────────

@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send (interactive if omitted)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Force model"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response"),
    show_stats: bool = typer.Option(True, "--stats/--no-stats", help="Show cost/latency stats"),
) -> None:
    """Chat with AI-COS. Interactive mode if no message given."""
    from aicos import AI

    ai = AI()

    async def _do_chat(msg: str) -> None:
        console.print(f"\n[bold cyan]You:[/bold cyan] {msg}")

        if stream:
            console.print("[bold green]AI:[/bold green] ", end="")
            collected = []
            async for token in ai.astream(msg, system=system, model=model):
                console.print(token, end="", highlight=False)
                collected.append(token)
            console.print()
        else:
            with console.status("[dim]Thinking...[/dim]"):
                response = await ai.achat(msg, system=system, model=model)
            console.print(f"[bold green]AI:[/bold green] {response}")

        if show_stats:
            summary = ai.cost_summary
            stats_text = (
                f"[dim]tokens: {summary.get('total_tokens', 0)} · "
                f"cost: ${summary.get('cost_usd', 0):.6f} · "
                f"requests: {summary.get('requests', 0)}[/dim]"
            )
            console.print(stats_text)

    if message:
        asyncio.run(_do_chat(message))
    else:
        # Interactive mode
        console.print("[bold cyan]AI-COS Interactive Chat[/bold cyan] [dim](Ctrl+C to exit)[/dim]\n")
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                    break
                if user_input.startswith("/clear"):
                    ai.clear_history()
                    console.print("[dim]History cleared[/dim]")
                    continue
                asyncio.run(_do_chat(user_input))
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break


# ── remember ──────────────────────────────────────────────────────────────────

@app.command()
def remember(
    content: str = typer.Argument(..., help="Content to remember"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
) -> None:
    """Store a memory in AI-COS."""
    from aicos import AI
    ai = AI()

    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    async def _store() -> None:
        memory_id = await ai.aremember(content, tags=tag_list)
        console.print(f"[green]Memory stored[/green] with ID: [bold]{memory_id}[/bold]")
        if tag_list:
            console.print(f"Tags: {', '.join(tag_list)}")

    asyncio.run(_store())


# ── forget ────────────────────────────────────────────────────────────────────

@app.command()
def forget(
    memory_id: int = typer.Argument(..., help="Memory ID to delete"),
) -> None:
    """Delete a stored memory."""
    from aicos import AI
    ai = AI()

    async def _delete() -> None:
        deleted = await ai.aforget(memory_id)
        if deleted:
            console.print(f"[green]Memory {memory_id} deleted[/green]")
        else:
            console.print(f"[red]Memory {memory_id} not found[/red]")
            raise typer.Exit(1)

    asyncio.run(_delete())


# ── search ────────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    threshold: float = typer.Option(0.3, "--threshold", help="Minimum similarity score"),
) -> None:
    """Search stored memories by semantic similarity."""
    from aicos import AI
    ai = AI()

    async def _search() -> None:
        results = await ai.asearch_memory(query, top_k=top_k, threshold=threshold)
        if not results:
            console.print("[dim]No memories found[/dim]")
            return

        table = Table(
            title=f"Memory Search: '{query}'",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
        )
        table.add_column("ID", style="dim", width=6)
        table.add_column("Score", width=8)
        table.add_column("Content")
        table.add_column("Tags", style="dim")

        for mem in results:
            score_color = "green" if mem["score"] > 0.7 else "yellow" if mem["score"] > 0.4 else "red"
            table.add_row(
                str(mem["id"]),
                f"[{score_color}]{mem['score']:.3f}[/{score_color}]",
                mem["content"][:80] + ("..." if len(mem["content"]) > 80 else ""),
                ", ".join(mem["tags"]) if mem["tags"] else "",
            )

        console.print(table)

    asyncio.run(_search())


# ── stats ─────────────────────────────────────────────────────────────────────

@app.command()
def stats() -> None:
    """Show AI-COS usage statistics."""
    from aicos.analytics.metrics import get_metrics
    m = get_metrics()
    data = m.to_dict()

    console.print(Panel("[bold cyan]AI-COS Statistics[/bold cyan]", border_style="cyan"))

    # Requests table
    req_table = Table(show_header=True, header_style="bold", box=box.SIMPLE)
    req_table.add_column("Metric")
    req_table.add_column("Value", justify="right")

    req_table.add_row("Total Requests", str(int(data["requests"]["total"])))
    req_table.add_row("Errors", str(int(data["requests"]["errors"])))
    req_table.add_row("Input Tokens", f"{int(data['tokens']['input_total']):,}")
    req_table.add_row("Output Tokens", f"{int(data['tokens']['output_total']):,}")
    req_table.add_row("Total Cost", f"${data['cost']['total_usd']:.4f}")
    req_table.add_row("Cost Saved (cache)", f"${data['cost']['saved_usd']:.4f}")
    req_table.add_row("Cache Hit Rate", f"{data['cache']['hit_rate'] * 100:.1f}%")
    req_table.add_row("Compression Ratio", f"{data['compression']['ratio'] * 100:.1f}%")
    req_table.add_row("Tokens Saved", f"{int(data['compression']['tokens_saved']):,}")
    req_table.add_row("Mean Latency", f"{data['latency']['mean_ms']:.0f}ms")
    req_table.add_row("P95 Latency", f"{data['latency']['p95_ms']:.0f}ms")
    req_table.add_row("Memories Stored", str(int(data["memory"]["stored"])))
    req_table.add_row("Uptime", f"{data['uptime_seconds']:.0f}s")

    console.print(req_table)


# ── config ────────────────────────────────────────────────────────────────────

@app.command(name="config")
def show_config() -> None:
    """Show current AI-COS configuration (secrets masked)."""
    from aicos.core.config import get_config
    cfg = get_config()
    masked = cfg.mask_secrets()

    table = Table(
        title="AI-COS Configuration",
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    for key, value in sorted(masked.items()):
        display = str(value) if value is not None else "[dim]not set[/dim]"
        table.add_row(key, display)

    console.print(table)


if __name__ == "__main__":
    app()
