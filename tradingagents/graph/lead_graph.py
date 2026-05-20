"""Lead discovery workflow.

This graph is deliberately separate from ``TradingAgentsGraph`` so the existing
ticker-analysis workflow remains unchanged while sharing the same LLM provider
configuration and structured-output style.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

from tqdm import tqdm

from tradingagents.agents.schemas import LeadCandidate, ScoredLead
from tradingagents.dataflows.lead_search import discover_companies_for_domain
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients import create_llm_client


CandidateSource = Callable[[str, int], List[LeadCandidate]]
T = TypeVar("T")


class LeadDiscoveryGraph:
    """Discover, score, and rank investment leads from seed domains."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        candidate_source: CandidateSource = discover_companies_for_domain,
        callbacks: Optional[List] = None,
        show_progress: bool = False,
    ):
        self.config = config or DEFAULT_CONFIG
        self.candidate_source = candidate_source
        self.callbacks = callbacks or []
        self.show_progress = show_progress

        llm_kwargs = self._get_provider_kwargs()
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_llm = deep_client.get_llm()
        self.quick_llm = quick_client.get_llm()

    def discover(self, domains: Iterable[str], investor_profile: str, limit: int = 25) -> List[ScoredLead]:
        """Discover and score leads for the user's investor profile."""
        candidates = self.discover_candidates(domains, per_domain_limit=limit)
        return self.score_candidates(candidates, investor_profile=investor_profile, limit=limit)

    def discover_candidates(
        self,
        domains: Iterable[str],
        per_domain_limit: int = 25,
    ) -> List[LeadCandidate]:
        """Collect unique company candidates from all seed domains."""
        candidates: List[LeadCandidate] = []
        seen = set()

        domain_list = [domain.strip() for domain in domains if domain.strip()]
        for domain in self._progress(domain_list, desc="Discovering themes"):
            for candidate in self.candidate_source(domain, per_domain_limit):
                key = (candidate.company.lower(), candidate.website or "")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)

        return candidates

    def score_candidates(
        self,
        candidates: Iterable[LeadCandidate],
        investor_profile: str,
        limit: int = 25,
    ) -> List[ScoredLead]:
        """Score candidates with the deep model and return the best leads."""
        scorer = self.deep_llm.with_structured_output(ScoredLead)
        scored: List[ScoredLead] = []
        candidate_list = list(candidates)

        for candidate in self._progress(candidate_list, desc="Scoring candidates"):
            result = scorer.invoke([
                {
                    "role": "system",
                    "content": (
                        "You score investment research leads for an individual investor. "
                        "Be specific, skeptical, and evidence-based. Use only the "
                        "provided candidate context and source URLs. This is not "
                        "financial advice; produce research ideas and risks to inspect."
                    ),
                },
                {
                    "role": "user",
                    "content": self._score_prompt(candidate, investor_profile),
                },
            ])
            if isinstance(result, ScoredLead):
                scored.append(result)
            else:
                scored.append(ScoredLead.model_validate(result))

        return sorted(scored, key=lambda lead: lead.score, reverse=True)[:limit]

    def _progress(self, items: List[T], desc: str):
        if not self.show_progress:
            return items
        return tqdm(items, desc=desc, unit="item")

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level
        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    @staticmethod
    def _score_prompt(candidate: LeadCandidate, investor_profile: str) -> str:
        return (
            "Investor profile / goal:\n"
            f"{investor_profile}\n\n"
            "Candidate company:\n"
            f"Company: {candidate.company}\n"
            f"Website/source: {candidate.website or 'Unknown'}\n"
            f"Seed domain: {candidate.domain}\n"
            f"Description/evidence: {candidate.description or 'No snippet available'}\n"
            f"Evidence URLs: {', '.join(candidate.evidence_urls) or 'None'}\n\n"
            "Score this investment lead from 0-100. Identify why it may be "
            "interesting, the strongest investor signal, catalysts to watch, "
            "source URLs, and a practical next research angle. Prefer public "
            "companies and clearly note when a company appears private."
        )
