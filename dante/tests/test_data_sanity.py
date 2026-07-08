"""Data-sanity checks over the prepared artifacts (§3.2 / §5).

Reads artifacts from ``$DANTE_ARTIFACTS`` (or ``./artifacts``). If they are absent
(the normal case on a dev box — they live on the Modal volume), every test SKIPS.
When present, it asserts the leakage-free, graded-eval invariants the ablation
depends on. The md5 ``hash(query_id) % 10`` split is recomputed here (matching
modal_train.py::prepare_data), not assumed.
"""
import hashlib
import json
import os

import pytest

ARTIFACTS = os.environ.get("DANTE_ARTIFACTS", "./artifacts")
DATA = os.path.join(ARTIFACTS, "data")
POS_GRADE = 2  # Exact + Substitute


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"artifact not found: {path} (set DANTE_ARTIFACTS to enable)")
    return path


def _load_json(name):
    with open(_require(os.path.join(DATA, name))) as f:
        return json.load(f)


def _md5_is_test(qid: str) -> bool:
    """Recompute the prepare_data split (md5 hash, NOT int %)."""
    h = int(hashlib.md5(str(qid).encode("utf-8")).hexdigest(), 16)
    return (h % 10) == 0


def test_qrels_has_all_four_grades():
    qrels = _load_json("qrels.json")
    grades = {g for pids in qrels.values() for g in pids.values()}
    assert grades == {0, 1, 2, 3}, f"expected all 4 grades, got {sorted(grades)}"


def test_grade_distribution_sane():
    qrels = _load_json("qrels.json")
    from collections import Counter

    counts = Counter(g for pids in qrels.values() for g in pids.values())
    total = sum(counts.values())
    assert total > 0
    # Exact (3) should be the plurality per ESCI distribution (~65%).
    assert counts[3] / total > 0.30, f"Exact share unexpectedly low: {counts}"


def test_catalog_covers_test_gold():
    import pandas as pd

    qrels = _load_json("qrels.json")
    catalog = pd.read_parquet(_require(os.path.join(DATA, "catalog.parquet")))
    cat_ids = set(catalog["product_id"].astype(str))
    gold = {pid for pids in qrels.values()
            for pid, g in pids.items() if g >= POS_GRADE}
    covered = sum(1 for pid in gold if pid in cat_ids)
    coverage = covered / len(gold) if gold else 1.0
    assert coverage >= 0.99, f"catalog covers only {coverage:.4f} of test-gold positives"


def test_split_is_query_disjoint_no_leakage():
    """Recompute the md5 split over queries.json: all test queries must hash to test."""
    queries = _load_json("queries.json")
    mislabeled = [qid for qid in queries if not _md5_is_test(qid)]
    # queries.json holds the TEST split, so every id must recompute to is_test=True.
    assert not mislabeled, f"{len(mislabeled)} test queries do not hash to the test bucket"


def test_stats_reports_zero_leakage():
    stats = _load_json("stats.json")
    assert stats.get("leakage", 1) == 0, f"stats.json reports leakage={stats.get('leakage')}"


def test_all_test_ids_hash_to_test_bucket():
    """Every id in the (test-only) queries/qrels files must md5-hash to the test bucket.

    NOTE: this does NOT check the ~10% split ratio — queries.json is already the
    test-only slice, so the fraction here is ~1.0 by construction. The real 10%
    ratio is asserted at prepare_data time (stats.json)."""
    qrels = _load_json("qrels.json")
    queries = _load_json("queries.json")
    # queries.json = test queries only, so we can't recompute the full ratio from it
    # alone; instead sanity-check the hash gives ~10% on the union of ids we DO see
    # plus assert the test set is non-trivial.
    all_test_ids = set(queries) | set(qrels)
    assert all_test_ids, "no test queries found"
    in_test = sum(1 for qid in all_test_ids if _md5_is_test(qid))
    frac = in_test / len(all_test_ids)
    # These are all test ids → should be ~1.0; the real 10% ratio is asserted at
    # prepare_data time (stats.json). Here we just confirm the hash is consistent.
    assert frac > 0.95, f"test ids inconsistent with md5 split: {frac:.3f}"
