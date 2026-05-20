from tradingagents.dataflows.lead_search import SearchResult, _result_to_candidate


def test_result_to_candidate_uses_title_and_evidence_url():
    result = SearchResult(
        title="Acme Robotics - Warehouse automation platform",
        url="https://acme.example",
        snippet="Acme Robotics builds warehouse automation systems.",
    )

    candidate = _result_to_candidate(result, "robotics automation")

    assert candidate.company == "Acme Robotics"
    assert candidate.domain == "robotics automation"
    assert candidate.evidence_urls == ["https://acme.example"]


def test_result_to_candidate_filters_low_signal_hosts():
    result = SearchResult(
        title="Video result",
        url="https://youtube.com/watch?v=123",
        snippet="",
    )

    assert _result_to_candidate(result, "space tech") is None
