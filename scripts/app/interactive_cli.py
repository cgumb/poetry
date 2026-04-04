from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from poetry_gp.backends.blocked import run_blocked_step
from poetry_gp.backends.scoring import try_create_daemon_client

console = Console()


TEXT_CANDIDATES = ["text", "poem", "content", "body"]
TITLE_CANDIDATES = ["title", "poem_title", "name"]
POET_CANDIDATES = ["poet", "author", "poet_name"]


def pick_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poems", type=Path, default=Path("data/poems.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--session-file", type=Path, default=Path("data/ratings_session.json"))
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--optimize-hyperparameters", action="store_true")
    parser.add_argument("--optimizer-maxiter", type=int, default=50)
    parser.add_argument("--score-backend", type=str, default="auto",
                        choices=["python", "daemon", "auto"],
                        help="Scoring backend: python (serial), daemon (parallel), auto (try daemon, fall back to python)")
    parser.add_argument("--daemon-nprocs", type=int, default=4,
                        help="Number of MPI processes for daemon scoring")
    parser.add_argument("--daemon-launcher", type=str, default="mpirun",
                        help="MPI launcher for daemon (mpirun, srun, etc.)")
    return parser.parse_args()


def display_text(value: object, fallback: str) -> str:
    text = str(value).strip()
    return text if text else fallback


def show_poem(df: pd.DataFrame, idx: int, title_col: str, poet_col: str, text_col: str) -> None:
    row = df.iloc[idx]
    title = display_text(row[title_col], "[untitled poem]")
    poet = display_text(row[poet_col], "[unknown poet]")

    # Create rich panel with poem
    header = Text()
    header.append(f"[{idx}] ", style="dim")
    header.append(title, style="bold cyan")
    header.append("\n")
    header.append(f"by {poet}", style="italic yellow")

    poem_text = str(row[text_col])[:4000]

    panel = Panel(
        f"{header}\n\n{poem_text}",
        title="📜 Poem",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2)
    )
    console.print(panel)


def save_session(path: Path, current_idx: int | None, rated_indices: list[int], ratings: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_index": current_idx,
        "rated_indices": rated_indices,
        "ratings": ratings,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_session(path: Path) -> tuple[int | None, list[int], list[float]]:
    if not path.exists():
        return None, [], []
    payload = json.loads(path.read_text())
    current_index = payload.get("current_index")
    rated_indices = [int(x) for x in payload.get("rated_indices", [])]
    ratings = [float(x) for x in payload.get("ratings", [])]
    if len(rated_indices) != len(ratings):
        raise ValueError(f"Session file {path} has mismatched rated_indices and ratings lengths")
    return current_index, rated_indices, ratings


def find_rating_files(data_dir: Path) -> list[Path]:
    """Find all rating session files in the data directory."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("ratings_*.json"))


def extract_username(rating_file: Path) -> str:
    """Extract username from rating file path (e.g., ratings_alice.json -> alice)."""
    name = rating_file.stem  # e.g., "ratings_alice"
    if name.startswith("ratings_"):
        return name[8:]  # Remove "ratings_" prefix
    return name


def select_or_create_user(data_dir: Path) -> Path:
    """Select existing user or create new one, return session file path."""
    rating_files = find_rating_files(data_dir)

    if len(rating_files) == 0:
        # No existing ratings, prompt for username
        console.print("[yellow]No existing rating sessions found.[/yellow]")
        username = Prompt.ask("👤 [bold cyan]Enter your username[/bold cyan]", default="user")
        return data_dir / f"ratings_{username}.json"

    if len(rating_files) == 1:
        # Only one session exists, use it automatically
        username = extract_username(rating_files[0])
        console.print(f"[green]✓[/green] Loading existing session for user: [bold]{username}[/bold]")
        return rating_files[0]

    # Multiple sessions exist, let user choose
    table = Table(title="👥 Existing Rating Sessions", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Username", style="cyan")
    table.add_column("Ratings", justify="right", style="green")

    for i, rf in enumerate(rating_files, start=1):
        username = extract_username(rf)
        session_data = json.loads(rf.read_text())
        n_ratings = len(session_data.get("rated_indices", []))
        table.add_row(str(i), username, str(n_ratings))

    table.add_row(str(len(rating_files) + 1), "[italic]Create new session[/italic]", "[dim]—[/dim]")

    console.print(table)

    while True:
        selection = Prompt.ask(f"Select session", default="1")
        try:
            choice = int(selection)
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
            continue

        if 1 <= choice <= len(rating_files):
            selected_file = rating_files[choice - 1]
            username = extract_username(selected_file)
            console.print(f"[green]✓[/green] Loading session for user: [bold]{username}[/bold]")
            return selected_file
        elif choice == len(rating_files) + 1:
            username = Prompt.ask("👤 [bold cyan]Enter username for new session[/bold cyan]", default="user")
            return data_dir / f"ratings_{username}.json"
        else:
            console.print(f"[red]Please select a number between 1 and {len(rating_files) + 1}.[/red]")


def print_rated_summary(df: pd.DataFrame, rated_indices: list[int], ratings: list[float], title_col: str, poet_col: str) -> None:
    if not rated_indices:
        console.print("[dim]No poems rated yet.[/dim]")
        return

    table = Table(title="⭐ Rated Poems", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=8)
    table.add_column("Rating", justify="center", width=10)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Poet", style="yellow", no_wrap=False)

    for idx, rating in zip(rated_indices, ratings):
        row = df.iloc[idx]
        title = display_text(row[title_col], "[untitled poem]")
        poet = display_text(row[poet_col], "[unknown poet]")

        # Color-code rating
        if rating > 0:
            rating_str = f"[green]{rating:+.1f}[/green]"
        elif rating < 0:
            rating_str = f"[red]{rating:+.1f}[/red]"
        else:
            rating_str = f"[dim]{rating:+.1f}[/dim]"

        table.add_row(str(idx), rating_str, title[:60], poet[:30])

    console.print(table)


def search_poems(df: pd.DataFrame, query: str, title_col: str, poet_col: str, text_col: str, limit: int = 10) -> list[int]:
    q = query.strip()
    if not q:
        return []
    title_mask = df[title_col].astype(str).str.contains(q, case=False, regex=False)
    poet_mask = df[poet_col].astype(str).str.contains(q, case=False, regex=False)
    text_mask = df[text_col].astype(str).str.contains(q, case=False, regex=False)
    mask = title_mask | poet_mask | text_mask
    matches = df.index[mask].tolist()[:limit]
    return [int(i) for i in matches]


def prompt_search(df: pd.DataFrame, title_col: str, poet_col: str, text_col: str) -> int | None:
    query = Prompt.ask("🔍 [bold cyan]Search title/poet/text[/bold cyan]")
    matches = search_poems(df, query, title_col, poet_col, text_col)
    if not matches:
        console.print("[yellow]No matches found.[/yellow]")
        return None

    table = Table(title="🔎 Search Results", box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Index", style="dim", width=8)
    table.add_column("Title", style="cyan", no_wrap=False)
    table.add_column("Poet", style="yellow", no_wrap=False)

    for i, idx in enumerate(matches, start=1):
        row = df.iloc[idx]
        title = display_text(row[title_col], "[untitled poem]")
        poet = display_text(row[poet_col], "[unknown poet]")
        table.add_row(str(i), str(idx), title[:50], poet[:30])

    console.print(table)

    selection = Prompt.ask("Choose a match number (blank to cancel)", default="")
    if not selection:
        return None
    try:
        chosen = int(selection)
    except ValueError:
        console.print("[red]Invalid selection.[/red]")
        return None
    if not (1 <= chosen <= len(matches)):
        console.print("[red]Selection out of range.[/red]")
        return None
    return matches[chosen - 1]


def main() -> None:
    args = parse_args()

    # Welcome banner
    banner = Panel(
        "[bold cyan]Poetry Exploration with Gaussian Process Recommendations[/bold cyan]\n\n"
        "Rate poems you encounter, then use [green bold]exploit[/green bold] to find poems you'll likely enjoy,\n"
        "or [magenta bold]explore[/magenta bold] to discover poems with uncertain appeal.\n\n"
        "[dim]An interactive demonstration of preference modeling and active learning.[/dim]",
        title="📚 Welcome to Poetry GP",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print(banner)

    # Load data
    with console.status("[bold cyan]Loading corpus...", spinner="dots"):
        poems = pd.read_parquet(args.poems)
        embeddings = np.load(args.embeddings, mmap_mode="r")

    if len(poems) != embeddings.shape[0]:
        raise ValueError("poem and embedding row counts do not match")

    console.print(f"[green]✓[/green] Loaded {len(poems):,} poems")

    cols = list(poems.columns)
    text_col = pick_column(cols, TEXT_CANDIDATES, cols[-1])
    title_col = pick_column(cols, TITLE_CANDIDATES, cols[0])
    poet_col = pick_column(cols, POET_CANDIDATES, cols[0])

    # Multi-user session management
    console.print()
    data_dir = args.session_file.parent
    session_file = select_or_create_user(data_dir)
    saved_current, rated_indices, ratings = load_session(session_file)
    rng = np.random.default_rng(args.seed)
    if saved_current is not None and 0 <= saved_current < len(poems):
        current_idx = int(saved_current)
        console.print(f"[green]✓[/green] Resumed session with {len(rated_indices)} ratings")
    else:
        current_idx = int(rng.integers(len(poems)))
        console.print(f"[cyan]Starting new session[/cyan]")

    current_length_scale = float(args.length_scale)
    current_variance = float(args.variance)
    current_noise = float(args.noise)

    # Initialize persistent daemon for scoring if needed
    daemon_client = None
    if args.score_backend in ("daemon", "auto"):
        with console.status("[bold cyan]Starting scoring daemon...", spinner="dots"):
            daemon_client = try_create_daemon_client(
                nprocs=args.daemon_nprocs,
                launcher=args.daemon_launcher,
                verbose=True
            )
        if daemon_client is not None:
            console.print(f"[green]✓[/green] Daemon started with {args.daemon_nprocs} processes")
        elif args.score_backend == "daemon":
            console.print("[red]✗[/red] Failed to start daemon (required for daemon mode)")
            return
        else:  # auto mode
            console.print("[yellow]⚠[/yellow] Daemon unavailable, will use Python scoring")

    # Create help panel
    help_panel = Panel(
        "[green bold]l[/green bold]ike   [dim]|[/dim]   "
        "[yellow bold]n[/yellow bold]eutral   [dim]|[/dim]   "
        "[red bold]d[/red bold]islike   [dim]|[/dim]   "
        "[cyan bold]e[/cyan bold]xploit   [dim]|[/dim]   "
        "[magenta bold]x[/magenta bold]plore   [dim]|[/dim]   "
        "[blue bold]s[/blue bold]earch   [dim]|[/dim]   "
        "[yellow bold]r[/yellow bold]ated   [dim]|[/dim]   "
        "[red bold]q[/red bold]uit",
        title="🎮 Commands",
        border_style="dim",
        box=box.ROUNDED
    )

    try:
        while True:
        console.print()
        show_poem(poems, current_idx, title_col, poet_col, text_col)
        console.print(help_panel)
        cmd = Prompt.ask("Choose action", default="").strip().lower()

        if cmd == "q":
            save_session(session_file, current_idx, rated_indices, ratings)
            console.print(f"[green]✓[/green] Saved session to [bold]{session_file}[/bold]")
            break

        if cmd in {"l", "n", "d"}:
            if current_idx not in rated_indices:
                rated_indices.append(current_idx)
                rating_value = {"l": 1.0, "n": 0.0, "d": -1.0}[cmd]
                ratings.append(rating_value)
                save_session(session_file, current_idx, rated_indices, ratings)

                # Color-coded feedback
                if rating_value > 0:
                    console.print(f"[green]✓ Recorded rating {rating_value:+.1f} for poem {current_idx}[/green]")
                elif rating_value < 0:
                    console.print(f"[red]✓ Recorded rating {rating_value:+.1f} for poem {current_idx}[/red]")
                else:
                    console.print(f"[yellow]✓ Recorded rating {rating_value:+.1f} for poem {current_idx}[/yellow]")
            else:
                console.print("[dim]Poem already rated.[/dim]")
            continue

        if cmd == "r":
            print_rated_summary(poems, rated_indices, ratings, title_col, poet_col)
            continue

        if cmd == "s":
            selected = prompt_search(poems, title_col, poet_col, text_col)
            if selected is not None:
                current_idx = selected
                save_session(session_file, current_idx, rated_indices, ratings)
            continue

        if cmd not in {"e", "x"}:
            console.print("[red]Unknown command.[/red]")
            continue

        if not rated_indices:
            console.print("[yellow]Rate at least one poem first.[/yellow]")
            continue
        # Show progress indicator
        with console.status("[bold cyan]Computing GP posterior...", spinner="dots"):
            result = run_blocked_step(
                embeddings,
                np.array(rated_indices, dtype=np.int64),
                np.array(ratings, dtype=np.float64),
                length_scale=current_length_scale,
                variance=current_variance,
                noise=current_noise,
                optimize_hyperparameters=args.optimize_hyperparameters,
                optimizer_maxiter=args.optimizer_maxiter,
                score_backend=args.score_backend,
                daemon_client=daemon_client,
                daemon_nprocs=args.daemon_nprocs,
                daemon_launcher=args.daemon_launcher,
            )

        if args.optimize_hyperparameters:
            current_length_scale = float(result.state.length_scale)
            current_variance = float(result.state.variance)
            current_noise = float(result.state.noise)
        current_idx = result.exploit_index if cmd == "e" else result.explore_index
        save_session(session_file, current_idx, rated_indices, ratings)

        # Create timing and parameter info panels
        timing_table = Table(show_header=False, box=None, padding=(0, 1))
        timing_table.add_column("Metric", style="cyan")
        timing_table.add_column("Value", style="green", justify="right")
        timing_table.add_row("Fit", f"{result.profile.fit_seconds:.4f}s")
        timing_table.add_row("Optimize", f"{result.profile.optimize_seconds:.4f}s")
        timing_table.add_row("Score", f"{result.profile.score_seconds:.4f}s")
        timing_table.add_row("Total", f"{result.profile.total_seconds:.4f}s")

        param_table = Table(show_header=False, box=None, padding=(0, 1))
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value", style="yellow", justify="right")
        param_table.add_row("Length scale", f"{result.state.length_scale:.4g}")
        param_table.add_row("Variance", f"{result.state.variance:.4g}")
        param_table.add_row("Noise", f"{result.state.noise:.4g}")
        if result.state.log_marginal_likelihood is not None:
            param_table.add_row("Log marginal ℒ", f"{result.state.log_marginal_likelihood:.6f}")

        console.print()
        console.print(Panel(timing_table, title="⏱️  Timing", border_style="green", box=box.ROUNDED))
        console.print(Panel(param_table, title="⚙️  Kernel Parameters", border_style="yellow", box=box.ROUNDED))

    finally:
        # Clean up daemon on exit
        if daemon_client is not None:
            console.print("[cyan]Shutting down daemon...[/cyan]")
            daemon_client.shutdown()
            console.print("[green]✓[/green] Daemon stopped")


if __name__ == "__main__":
    main()
