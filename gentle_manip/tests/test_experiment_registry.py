"""Tests for the global experiment-ID registry (gentle_manip.utils.experiment_registry)."""
import random

import pytest

import gentle_manip.utils.experiment_registry as R


@pytest.fixture
def tbl(tmp_path):
    return tmp_path / "experiments.csv"


def _mk_run(base, algo, task, eid):
    d = base / algo / task / eid
    (d / "checkpoint").mkdir(parents=True)
    return d


def test_new_id_shape_and_uniqueness(tbl):
    rng = random.Random(0)
    ids = {R.new_id(table_path=tbl, base="nope", rng=rng) for _ in range(20)}
    # each freshly minted (no registration between) can repeat since nothing is taken; instead
    # test the shape + that registering makes it taken.
    for i in ids:
        assert R._looks_like_id(i) and len(i) == 5 and i.isalpha() and i.islower()


def test_id_avoids_taken(tbl, tmp_path):
    # register one id, then a new id must differ from it
    first = R.new_id(table_path=tbl, base="nope", rng=random.Random(1))
    R.add_entry(first, "dppo", "t", "t", tmp_path / first, table_path=tbl)
    for _ in range(50):
        nxt = R.new_id(table_path=tbl, base="nope")
        assert nxt != first


def test_add_entry_idempotent_update(tbl, tmp_path):
    R.add_entry("aaaaa", "serl", "lift", "exp", tmp_path / "r", table_path=tbl)
    R.add_entry("aaaaa", "serl", "lift", "exp", tmp_path / "r", status="done", table_path=tbl)
    rows = R.load_table(tbl)
    assert len(rows) == 1
    assert rows[0]["status"] == "done"
    assert rows[0]["algo"] == "serl" and rows[0]["task"] == "lift"


def test_set_status(tbl, tmp_path):
    R.add_entry("bbbbb", "dppo", "t", "e", tmp_path / "r", table_path=tbl)
    R.set_status("bbbbb", "finished", table_path=tbl)
    assert R.load_table(tbl)[0]["status"] == "finished"


def test_looks_like_id():
    assert R._looks_like_id("abcde")
    assert not R._looks_like_id("abcd")       # too short
    assert not R._looks_like_id("abcdef")     # too long
    assert not R._looks_like_id("ABCDE")      # uppercase
    assert not R._looks_like_id("abc1e")      # digit
    assert not R._looks_like_id("2026-07-05_12-00-00")


def test_reconcile_with_repo_base(tbl, tmp_path, monkeypatch):
    # point _REPO at tmp so base="logs" resolves under tmp_path
    monkeypatch.setattr(R, "_REPO", tmp_path)
    base = tmp_path / "logs"
    kept = _mk_run(base, "dppo-finetune", "taskA", "kkkkk")
    R.add_entry("kkkkk", "dppo-finetune", "taskA", "expA", kept, table_path=tbl)
    R.add_entry("ddddd", "serl", "taskB", "expB", base / "serl" / "taskB" / "ddddd", table_path=tbl)
    _mk_run(base, "dppo-pretrain", "taskC", "ooooo")

    res = R.reconcile(table_path=tbl, base="logs")
    ids = {r["id"] for r in R.load_table(tbl)}
    assert "kkkkk" in ids                 # kept (dir exists)
    assert "ddddd" not in ids and "ddddd" in res["dropped"]   # dropped (dir gone)
    assert "ooooo" in ids and "ooooo" in res["added"]         # back-filled
    # inferred algo/task from path for the orphan
    orphan = next(r for r in R.load_table(tbl) if r["id"] == "ooooo")
    assert orphan["algo"] == "dppo-pretrain" and orphan["task"] == "taskC"


def test_format_table_empty(tbl):
    assert "no experiments" in R.format_table(tbl)
