from unittest.mock import MagicMock, patch

from tradingagents.agents.schemas import LeadCandidate, ScoredLead
from tradingagents.graph.lead_graph import LeadDiscoveryGraph


class FakeStructuredScorer:
    def invoke(self, messages):
        content = messages[-1]["content"]
        company = content.split("Company: ", 1)[1].split("\n", 1)[0]
        return ScoredLead(
            company=company,
            website="https://example.com",
            domain="robotics",
            score=87 if company == "Acme Robotics" else 52,
            priority="High" if company == "Acme Robotics" else "Medium",
            why_interesting="Shows a plausible fit for the investor profile.",
            investor_signal="Domain match",
            catalysts=["Domain match"],
            research_angle="Check whether the company is public and review filings.",
            source_urls=["https://example.com"],
        )


class FakeLLM:
    def with_structured_output(self, schema):
        assert schema is ScoredLead
        return FakeStructuredScorer()


def test_lead_discovery_scores_and_sorts_candidates():
    client = MagicMock()
    client.get_llm.return_value = FakeLLM()

    def candidate_source(domain, limit):
        return [
            LeadCandidate(company="Beta Space", website="https://beta.example", domain=domain),
            LeadCandidate(company="Acme Robotics", website="https://acme.example", domain=domain),
        ]

    config = {
        "llm_provider": "google",
        "deep_think_llm": "gemini-flash-latest",
        "quick_think_llm": "gemini-flash-latest",
        "backend_url": None,
        "google_thinking_level": None,
    }

    with patch("tradingagents.graph.lead_graph.create_llm_client", return_value=client):
        graph = LeadDiscoveryGraph(config=config, candidate_source=candidate_source)
        leads = graph.discover(
            ["robotics"],
            investor_profile="Individual investor seeking robotics public equities",
            limit=2,
        )

    assert [lead.company for lead in leads] == ["Acme Robotics", "Beta Space"]
    assert leads[0].score == 87


def test_lead_discovery_deduplicates_candidates():
    client = MagicMock()
    client.get_llm.return_value = FakeLLM()

    def candidate_source(domain, limit):
        return [
            LeadCandidate(company="Acme Robotics", website="https://acme.example", domain=domain),
            LeadCandidate(company="Acme Robotics", website="https://acme.example", domain=domain),
        ]

    config = {
        "llm_provider": "google",
        "deep_think_llm": "gemini-flash-latest",
        "quick_think_llm": "gemini-flash-latest",
        "backend_url": None,
        "google_thinking_level": None,
    }

    with patch("tradingagents.graph.lead_graph.create_llm_client", return_value=client):
        graph = LeadDiscoveryGraph(config=config, candidate_source=candidate_source)
        candidates = graph.discover_candidates(["robotics"])

    assert len(candidates) == 1
