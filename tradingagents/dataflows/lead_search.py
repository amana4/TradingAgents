"""Lead discovery data sources.

The first implementation uses DuckDuckGo's HTML endpoint because it requires
no API key and keeps the workflow useful out of the box. Paid sources such as
Crunchbase, Apollo, People Data Labs, or LinkedIn exports can be added here
without touching the lead graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests
from parsel import Selector

from tradingagents.agents.schemas import LeadCandidate


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


_DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"
_DEFAULT_TIMEOUT = 10


def discover_companies_for_domain(domain: str, limit: int = 25) -> List[LeadCandidate]:
    """Discover company candidates for a seed market/domain.

    Results are intentionally candidates, not final leads. The lead graph's LLM
    scoring step decides whether each company is a real fit for the user's offer.
    """
    candidates: List[LeadCandidate] = []
    seen = set()

    for query in _queries_for_domain(domain):
        for result in search_web(query, limit=limit):
            candidate = _result_to_candidate(result, domain)
            if not candidate:
                continue
            key = (candidate.company.lower(), candidate.website or "")
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates

    return candidates


def search_web(query: str, limit: int = 10) -> List[SearchResult]:
    """Search the web and parse organic results from DuckDuckGo HTML."""
    response = requests.get(
        _DUCKDUCKGO_HTML_URL,
        params={"q": query},
        headers={"User-Agent": "TradingAgents lead discovery"},
        timeout=_DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    selector = Selector(text=response.text)
    results: List[SearchResult] = []

    for row in selector.css(".result"):
        title = " ".join(row.css(".result__title a::text").getall()).strip()
        href = row.css(".result__title a::attr(href)").get()
        snippet = " ".join(row.css(".result__snippet ::text").getall()).strip()
        url = _normalize_duckduckgo_url(href)

        if not title or not url:
            continue
        results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break

    return results


def _queries_for_domain(domain: str) -> Iterable[str]:
    domain = domain.strip()
    yield f'{domain} startups companies'
    yield f'{domain} companies hiring'
    yield f'{domain} funding announcement startup'
    yield f'{domain} conference exhibitors companies'


def _result_to_candidate(result: SearchResult, domain: str) -> Optional[LeadCandidate]:
    if _is_low_signal_url(result.url):
        return None

    company = _company_from_title(result.title)
    if not company:
        company = _company_from_url(result.url)
    if not company:
        return None

    return LeadCandidate(
        company=company,
        website=result.url,
        domain=domain,
        description=result.snippet,
        evidence_urls=[result.url],
    )


def _normalize_duckduckgo_url(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg", [None])[0]
        return unquote(uddg) if uddg else None
    return href


def _company_from_title(title: str) -> str:
    title = title.strip()
    for separator in (" | ", " - ", " — ", " – ", ":"):
        if separator in title:
            title = title.split(separator, 1)[0].strip()
            break
    return title[:120]


def _company_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split(".", 1)[0].replace("-", " ").title()


def _is_low_signal_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    low_signal_hosts = (
        "google.com",
        "duckduckgo.com",
        "youtube.com",
        "facebook.com",
        "instagram.com",
        "x.com",
        "twitter.com",
    )
    return any(host == item or host.endswith(f".{item}") for item in low_signal_hosts)
