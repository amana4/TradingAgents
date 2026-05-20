"""Discover and score investment leads from seed domains."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.lead_graph import LeadDiscoveryGraph

DEFAULT_OPTIONS_PATH = Path("investment_options.txt")


def load_options(path: Path) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number} must use key=value format")
        key, value = line.split("=", 1)
        options[key.strip().lower()] = value.strip()
    return options


def option(options: Dict[str, str], *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = options.get(name)
        if value:
            return value
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and score investment leads from themes.")
    parser.add_argument("--config", default=str(DEFAULT_OPTIONS_PATH), help="Path to key=value options file.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Options file not found: {config_path}")

    options = load_options(config_path)
    raw_domains = option(options, "themes", "domains")
    investor_profile = option(options, "investor_profile", "goal", "strategy", "offer")
    output = option(options, "output", default="investment_ideas.csv")
    limit = int(option(options, "limit", default="25"))

    if not raw_domains:
        parser.error(f"{config_path} must define themes=... or domains=...")
    if not investor_profile:
        parser.error(f"{config_path} must define investor_profile=... or goal=...")

    domains = [domain.strip() for domain in raw_domains.split(",") if domain.strip()]
    print(f"Loaded options from {config_path}")
    print(f"Searching {len(domains)} themes; writing up to {limit} ideas to {output}")

    graph = LeadDiscoveryGraph(
        config=DEFAULT_CONFIG.copy(),
        show_progress=not args.no_progress,
    )
    leads = graph.discover(domains=domains, investor_profile=investor_profile, limit=limit)

    fieldnames = [
        "company",
        "website",
        "domain",
        "score",
        "priority",
        "why_interesting",
        "investor_signal",
        "catalysts",
        "research_angle",
        "source_urls",
    ]
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for lead in leads:
            row = lead.model_dump()
            row["catalysts"] = "; ".join(row.get("catalysts", []))
            row["source_urls"] = "; ".join(row.get("source_urls", []))
            writer.writerow(row)

    print(f"Wrote {len(leads)} leads to {output}")


if __name__ == "__main__":
    main()
