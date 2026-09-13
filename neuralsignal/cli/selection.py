"""Interactive path selection; scripts must supply explicit paths."""
from pathlib import Path
import sys
import yaml

from neuralsignal.config import load_config
from neuralsignal.console import color


def choose(entries, title, hint):
    if not entries:
        raise SystemExit(f"No entries found for {title}. {hint}")
    print(color(title, "cyan"))
    for number, (path, detail, available) in enumerate(entries, 1):
        print(f"  {color(str(number), 'yellow')}. {color(str(path), 'green' if available else 'dim')}  {color(detail, 'cyan' if available else 'dim')}")
    if not any(available for _, _, available in entries):
        raise SystemExit(f"No selectable entries. {hint}")
    if not sys.stdin.isatty():
        raise SystemExit(f"Interactive selection requires a terminal. {hint}")
    while True:
        try:
            answer = input("Choose a number (q to cancel): ").strip()
        except EOFError:
            raise SystemExit("Selection cancelled.") from None
        if answer.lower() in {"q", "quit"}:
            raise SystemExit("Selection cancelled.")
        if answer.isdigit() and 1 <= int(answer) <= len(entries):
            path, _, available = entries[int(answer) - 1]
            if available:
                return str(path)
        print(color("Choose a numbered, available entry, or q to cancel.", "yellow"))


def choose_config(kind):
    entries = []
    for path in sorted({*Path("configs").rglob("*.yaml"), *Path("configs").rglob("*.yml")}):
        try:
            config = load_config(path)
            if "remote_collect" in config:
                category = "remote launch"
            elif "runpod" in config:
                category = "pod manifest"
            elif "extraction" in config or "materialize" in (config.get("features") or {}):
                category = "feature collection"
            elif "dataset" in config and ("label_column" in config["dataset"] or "path" in config["dataset"]):
                category = "training"
            else:
                category = "other config"
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            category = "unreadable config"
        eligible = category == "training" if kind == "training" else category in (
            {"feature collection", "remote launch"} if kind == "remote" else {"feature collection"})
        entries.append((path, category + ("" if eligible else " (not selectable here)"), eligible))
    return choose(entries, f"Choose a {kind} config from configs/", "Pass a compatible YAML config path explicitly.")


def choose_run():
    root = Path("runs")
    paths = set(root.iterdir()) if root.exists() else set()
    # Keep container directories visible and expose their nested feature runs.
    paths.update(path.parent for path in root.rglob("manifest.json"))
    paths.update(path.parent for path in root.rglob("features") if path.is_dir())
    entries = []
    for path in sorted(paths):
        if path == root:
            continue
        available = path.is_dir() and ((path / "manifest.json").is_file() or (path / "features").is_dir())
        detail = "feature run" if available else "folder" if path.is_dir() else "file"
        if path == root / "s1" or root / "s1" in path.parents:
            available, detail = False, "S1 training outputs"
        entries.append((path, detail + ("" if available else " (not a feature run)"), available))
    return choose(entries, "Choose a feature run from runs/", "Pass --run PATH to a local feature run or Parquet file.")
