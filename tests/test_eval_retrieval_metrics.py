from maison_concierge.eval.retrieval import _mrr, _recall_at_k


def test_recall_at_k_full_hit():
    assert _recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_recall_at_k_partial():
    assert _recall_at_k(["a", "z", "c"], ["a", "b"], k=3) == 0.5


def test_recall_at_k_no_hit():
    assert _recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


def test_recall_at_k_with_empty_expected():
    assert _recall_at_k(["a", "b"], [], k=3) == 0.0


def test_mrr_first_position():
    assert _mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_third_position():
    assert _mrr(["x", "y", "a"], ["a"]) == 1 / 3


def test_mrr_no_match():
    assert _mrr(["x", "y"], ["a"]) == 0.0
