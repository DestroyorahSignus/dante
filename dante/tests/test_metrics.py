"""Unit tests for the retrieval metrics (§5.1) — self-contained, no artifacts/GPU."""
import math

from dante.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1.0 / 3.0


def test_reciprocal_rank_no_relevant():
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_reciprocal_rank_takes_first_hit():
    # b at rank 2 and a at rank 3 are both relevant → MRR uses the first hit.
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_recall_at_k_partial():
    # 2 of 4 relevant found in top-3.
    assert recall_at_k(["a", "b", "x", "c"], {"a", "b", "c", "d"}, k=3) == 0.5


def test_recall_at_k_full():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_ndcg_perfect_order_is_one():
    # Ideal order (3,2,1) → nDCG == 1.0.
    ranked = ["a", "b", "c"]
    rel = {"a": 3, "b": 2, "c": 1}
    assert math.isclose(ndcg_at_k(ranked, rel, k=3), 1.0, rel_tol=1e-9)


def test_ndcg_reversed_order_less_than_one():
    rel = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["c", "b", "a"], rel, k=3) < 1.0


def test_ndcg_known_value():
    # ranked [b(2), a(3)] over k=2.
    # DCG = 2/log2(2) + 3/log2(3) = 2/1 + 3/1.5849625 = 2 + 1.892789 = 3.892789
    # IDCG (ideal 3,2) = 3/1 + 2/1.5849625 = 3 + 1.261859 = 4.261859
    rel = {"a": 3, "b": 2}
    expected = (2 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 2 / math.log2(3))
    assert math.isclose(ndcg_at_k(["b", "a"], rel, k=2), expected, rel_tol=1e-9)


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["x", "y"], {"x": 0, "y": 0}, k=2) == 0.0
