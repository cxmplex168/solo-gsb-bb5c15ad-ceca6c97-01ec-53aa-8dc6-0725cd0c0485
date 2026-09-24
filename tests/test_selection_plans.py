"""End-to-end tests for the read-only selection-plan simulation."""
from __future__ import annotations

import itertools
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-selection-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "selection.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dsplan-{name}-{next(_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


def _cand(candidate_id: str, epsilon: float, delta: float = 0.0,
          value: int = 1) -> dict:
    return {
        "candidate_id": candidate_id,
        "epsilon": epsilon,
        "delta": delta,
        "value": value,
    }


async def _plan(client, dataset_id: str, candidates: list[dict]):
    return await client.post(
        f"/datasets/{dataset_id}/selection-plans",
        json={"candidates": candidates},
    )


def _expected_plan(
    committed: list[tuple[float, float]],
    candidates: list[dict],
    cap: float,
    delta_cap: float,
) -> tuple[list[str], int, float]:
    """Independent brute-force recomputation via rdp.evaluate_entries."""
    best_key = None
    best = ([], 0, 0.0)
    n = len(candidates)
    for mask in range(1 << n):
        picked = [c for i, c in enumerate(candidates) if mask & (1 << i)]
        entries = committed + [(c["epsilon"], c["delta"]) for c in picked]
        _, eps, _ = rdp.evaluate_entries(entries, delta_cap)
        eps = max(0.0, eps)
        if eps > cap + settings.epsilon_tolerance:
            continue
        ids = sorted(c["candidate_id"] for c in picked)
        value = sum(c["value"] for c in picked)
        key = (-value, eps, ids)
        if best_key is None or key < best_key:
            best_key = key
            best = (ids, value, eps)
    return best


# ----------------------------------------------------------------- behaviour --


@pytest.mark.asyncio
async def test_plan_all_candidates_fit(client):
    ds = _new_dataset_id("allfit")
    await _register(client, ds, eps=1.0, delta=1e-6)
    candidates = [
        _cand("c1", 0.1, 1e-8, 2),
        _cand("c2", 0.2, 1e-8, 3),
        _cand("c3", 0.05, 0.0, 1),
    ]
    r = await _plan(client, ds, candidates)
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == ds
    assert body["candidate_count"] == 3
    assert body["selected_candidate_ids"] == ["c1", "c2", "c3"]
    assert body["unselected_candidate_ids"] == []
    assert body["total_value"] == 6
    assert body["epsilon_cap"] == 1.0
    assert body["epsilon_committed"] == 0.0
    _, want_eps, _ = rdp.evaluate_entries(
        [(0.1, 1e-8), (0.2, 1e-8), (0.05, 0.0)], 1e-6
    )
    assert body["projected_epsilon"] == pytest.approx(want_eps)
    assert body["epsilon_remaining"] == pytest.approx(1.0 - want_eps)
    assert body["rdp_best_order"] > 0
    datetime.fromisoformat(body["generated_at"])


@pytest.mark.asyncio
async def test_plan_picks_highest_value_subset(client):
    ds = _new_dataset_id("maxvalue")
    cap, delta_cap = 1.0, 1e-6
    await _register(client, ds, eps=cap, delta=delta_cap)
    # {c1,c2} (value 8) beats {c3} (value 5); c3 pairs with nothing and all
    # three together exceed the cap.
    candidates = [
        _cand("c1", 0.35, 1e-8, 4),
        _cand("c2", 0.35, 1e-8, 4),
        _cand("c3", 0.7, 1e-8, 5),
    ]
    r = await _plan(client, ds, candidates)
    assert r.status_code == 200
    body = r.json()
    want_ids, want_value, want_eps = _expected_plan(
        [], candidates, cap, delta_cap
    )
    assert want_ids == ["c1", "c2"]  # sanity: fixture shaped as intended
    assert body["selected_candidate_ids"] == want_ids
    assert body["unselected_candidate_ids"] == ["c3"]
    assert body["total_value"] == want_value == 8
    assert body["projected_epsilon"] == pytest.approx(want_eps)
    assert body["epsilon_remaining"] == pytest.approx(cap - want_eps)


@pytest.mark.asyncio
async def test_plan_tie_breaks_by_smaller_projected_epsilon(client):
    ds = _new_dataset_id("tie-eps")
    await _register(client, ds, eps=0.25, delta=1e-6)
    # Same value, only one fits: the cheaper candidate wins.
    candidates = [_cand("hi", 0.2, 0.0, 5), _cand("lo", 0.1, 0.0, 5)]
    r = await _plan(client, ds, candidates)
    assert r.status_code == 200
    body = r.json()
    assert body["selected_candidate_ids"] == ["lo"]
    assert body["unselected_candidate_ids"] == ["hi"]
    assert body["total_value"] == 5
    assert body["projected_epsilon"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_plan_tie_breaks_by_lexicographic_id_list(client):
    ds = _new_dataset_id("tie-ids")
    await _register(client, ds, eps=0.15, delta=1e-6)
    # Identical epsilon/value; only one fits: ascending id order decides.
    candidates = [_cand("b", 0.1, 0.0, 5), _cand("a", 0.1, 0.0, 5)]
    body = (await _plan(client, ds, candidates)).json()
    assert body["selected_candidate_ids"] == ["a"]
    assert body["unselected_candidate_ids"] == ["b"]

    # Multi-element lists compare lexicographically too: ["a","c"] < ["b"].
    ds2 = _new_dataset_id("tie-ids2")
    await _register(client, ds2, eps=0.1, delta=1e-6)
    candidates2 = [
        _cand("b", 0.1, 0.0, 4),
        _cand("a", 0.05, 0.0, 2),
        _cand("c", 0.05, 0.0, 2),
    ]
    body2 = (await _plan(client, ds2, candidates2)).json()
    assert body2["selected_candidate_ids"] == ["a", "c"]
    assert body2["unselected_candidate_ids"] == ["b"]
    assert body2["total_value"] == 4


@pytest.mark.asyncio
async def test_plan_empty_when_nothing_fits_reports_occupancy(client):
    ds = _new_dataset_id("empty")
    await _register(client, ds, eps=0.5, delta=1e-6)
    granted = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-1", "query_type": "histogram",
              "epsilon": 0.4, "delta": 0.0},
    )
    assert granted.json()["allowed"] is True

    r = await _plan(client, ds, [_cand("too-big", 0.2, 0.0, 9)])
    assert r.status_code == 200
    body = r.json()
    assert body["selected_candidate_ids"] == []
    assert body["unselected_candidate_ids"] == ["too-big"]
    assert body["total_value"] == 0
    # Empty plan: projection is exactly the current occupancy.
    assert body["epsilon_committed"] == pytest.approx(0.4)
    assert body["projected_epsilon"] == pytest.approx(0.4)
    assert body["epsilon_remaining"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_plan_counts_granted_requests_and_active_reservations(client):
    ds = _new_dataset_id("committed")
    cap, delta_cap = 1.0, 1e-6
    await _register(client, ds, eps=cap, delta=delta_cap)
    await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-1", "query_type": "gaussian_mean",
              "epsilon": 0.15, "delta": 1e-7},
    )
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "res-1", "epsilon": 0.1, "delta": 1e-8,
              "ttl_seconds": 3600},
    )
    candidates = [
        _cand("c1", 0.2, 1e-8, 3),
        _cand("c2", 0.3, 1e-9, 5),
        _cand("c3", 0.1, 0.0, 2),
        _cand("c4", 0.25, 1e-7, 4),
        _cand("c5", 0.05, 1e-6, 1),
        _cand("c6", 0.4, 1e-8, 6),
    ]
    body = (await _plan(client, ds, candidates)).json()
    committed = [(0.15, 1e-7), (0.1, 1e-8)]
    want_ids, want_value, want_eps = _expected_plan(
        committed, candidates, cap, delta_cap
    )
    _, want_committed, _ = rdp.evaluate_entries(committed, delta_cap)
    assert body["epsilon_committed"] == pytest.approx(max(0.0, want_committed))
    assert body["selected_candidate_ids"] == want_ids
    assert body["total_value"] == want_value
    assert body["projected_epsilon"] == pytest.approx(want_eps)
    assert body["epsilon_remaining"] == pytest.approx(cap - want_eps)
    assert sorted(
        body["selected_candidate_ids"] + body["unselected_candidate_ids"]
    ) == sorted(c["candidate_id"] for c in candidates)


@pytest.mark.asyncio
async def test_plan_ignores_reversed_requests(client):
    ds = _new_dataset_id("reversed")
    await _register(client, ds, eps=0.5, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-1", "query_type": "sql_count",
              "epsilon": 0.4, "delta": 0.0},
    )
    # While req-1 occupies 0.4, a 0.2 candidate cannot fit.
    before = (await _plan(client, ds, [_cand("c1", 0.2, 0.0, 7)])).json()
    assert before["selected_candidate_ids"] == []
    assert before["epsilon_committed"] == pytest.approx(0.4)

    rev = await client.post(
        f"/datasets/{ds}/requests/req-1/reversals",
        # reversal_id is globally unique across datasets: namespace it.
        json={"reversal_id": f"rev-{ds}", "reason": "mistaken submission"},
    )
    assert rev.status_code == 201

    # After the reversal the budget is free again and the plan recomputes.
    after = (await _plan(client, ds, [_cand("c1", 0.2, 0.0, 7)])).json()
    assert after["selected_candidate_ids"] == ["c1"]
    assert after["total_value"] == 7
    assert after["epsilon_committed"] == pytest.approx(0.0)
    assert after["projected_epsilon"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_plan_ignores_expired_reservations_without_sweeping(client):
    ds = _new_dataset_id("expired")
    await _register(client, ds, eps=0.5, delta=1e-6)
    created = await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "res-1", "epsilon": 0.4, "delta": 0.0,
              "ttl_seconds": 10},
    )
    assert created.json()["allowed"] is True
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    async with db.connect() as conn:
        await db.backdate_reservation_expiry(
            conn, dataset_id=ds, reservation_id="res-1",
            expires_at=past.isoformat(),
        )
        await conn.commit()

    # The expired reservation no longer counts against the plan...
    body = (await _plan(client, ds, [_cand("c1", 0.4, 0.0, 3)])).json()
    assert body["epsilon_committed"] == pytest.approx(0.0)
    assert body["selected_candidate_ids"] == ["c1"]

    # ...but the simulation is read-only: the row is still stored as
    # "granted" (the lazy expiry sweep only runs on the normal read paths).
    async with db.connect() as conn:
        row = await db.get_reservation(conn, ds, "res-1")
        assert row["status"] == "granted"
        assert row["expired_at"] is None

    got = await client.get(f"/datasets/{ds}/reservations/res-1")
    assert got.json()["status"] == "expired"


@pytest.mark.asyncio
async def test_plan_is_read_only_and_repeatable(client):
    ds = _new_dataset_id("readonly")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-1", "query_type": "histogram",
              "epsilon": 0.2, "delta": 1e-8},
    )
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "res-1", "epsilon": 0.1, "delta": 1e-8,
              "ttl_seconds": 3600},
    )
    candidates = [_cand("c1", 0.2, 1e-8, 3), _cand("c2", 0.3, 1e-8, 4)]

    first = (await _plan(client, ds, candidates)).json()
    second = (await _plan(client, ds, candidates)).json()
    # generated_at may move; the plan content must not.
    for key in (
        "epsilon_cap", "epsilon_committed", "candidate_count",
        "selected_candidate_ids", "unselected_candidate_ids", "total_value",
        "projected_epsilon", "epsilon_remaining", "rdp_best_order",
    ):
        assert first[key] == second[key]

    # Nothing was created or mutated: no new requests, no reservations
    # consumed, no reversals, budget view unchanged.
    reqs = (await client.get(f"/datasets/{ds}/requests")).json()
    assert reqs["count"] == 1
    reservations = (await client.get(f"/datasets/{ds}/reservations")).json()
    assert [r["status"] for r in reservations["reservations"]] == ["granted"]
    reversals = (await client.get(f"/datasets/{ds}/reversals")).json()
    assert reversals["count"] == 0
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["granted_requests"] == 1
    assert budget["active_reservations"] == 1
    assert budget["epsilon_consumed"] == pytest.approx(
        first["epsilon_committed"] - budget["epsilon_reserved"]
    )


@pytest.mark.asyncio
async def test_plan_recomputes_against_latest_ledger(client):
    ds = _new_dataset_id("recompute")
    await _register(client, ds, eps=1.0, delta=1e-6)
    candidates = [_cand("a", 0.6, 0.0, 10), _cand("b", 0.3, 0.0, 4)]

    first = (await _plan(client, ds, candidates)).json()
    assert first["selected_candidate_ids"] == ["a", "b"]
    assert first["total_value"] == 14

    # A newly granted request shrinks what the next plan can pick.
    await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "req-1", "query_type": "sql_count",
              "epsilon": 0.3, "delta": 0.0},
    )
    second = (await _plan(client, ds, candidates)).json()
    assert second["selected_candidate_ids"] == ["a"]
    assert second["total_value"] == 10
    assert second["epsilon_committed"] == pytest.approx(0.3)
    assert second["projected_epsilon"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_plan_pure_dp_exact_composition(client):
    ds = _new_dataset_id("pure")
    await _register(client, ds, eps=1.0, delta=1e-6)
    # All-pure candidates compose exactly: any two fit (0.8), three do not.
    candidates = [
        _cand("c1", 0.4, 0.0, 3),
        _cand("c2", 0.4, 0.0, 3),
        _cand("c3", 0.4, 0.0, 3),
    ]
    body = (await _plan(client, ds, candidates)).json()
    assert body["selected_candidate_ids"] == ["c1", "c2"]
    assert body["total_value"] == 6
    # Exact basic composition, no RDP conversion slack.
    assert body["projected_epsilon"] == pytest.approx(0.8)
    assert body["epsilon_remaining"] == pytest.approx(0.2)
    assert body["rdp_best_order"] == rdp.RDP_ORDERS[0]


@pytest.mark.asyncio
async def test_plan_twelve_candidates_accepted(client):
    ds = _new_dataset_id("twelve")
    await _register(client, ds, eps=1.0, delta=1e-6)
    candidates = [
        _cand(f"c{i:02d}", 0.05 * (i + 1), 1e-8, i + 1) for i in range(12)
    ]
    r = await _plan(client, ds, candidates)
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_count"] == 12
    want_ids, want_value, want_eps = _expected_plan([], candidates, 1.0, 1e-6)
    assert body["selected_candidate_ids"] == want_ids
    assert body["total_value"] == want_value
    assert body["projected_epsilon"] == pytest.approx(want_eps)


# ---------------------------------------------------------------- validation --


@pytest.mark.asyncio
async def test_plan_missing_dataset_404(client):
    r = await _plan(client, "no-such-dataset", [_cand("c1", 0.1, 0.0, 1)])
    assert r.status_code == 404
    assert r.json()["error"] == "dataset_not_found"


@pytest.mark.asyncio
async def test_plan_candidate_count_bounds(client):
    ds = _new_dataset_id("bounds")
    await _register(client, ds)
    assert (await _plan(client, ds, [])).status_code == 422
    thirteen = [_cand(f"c{i}", 0.01, 0.0, 1) for i in range(13)]
    assert (await _plan(client, ds, thirteen)).status_code == 422


@pytest.mark.asyncio
async def test_plan_duplicate_candidate_id_422(client):
    ds = _new_dataset_id("dup")
    await _register(client, ds)
    candidates = [_cand("c1", 0.1, 0.0, 1), _cand("c1", 0.2, 0.0, 2)]
    assert (await _plan(client, ds, candidates)).status_code == 422


@pytest.mark.asyncio
async def test_plan_invalid_candidate_fields_422(client):
    ds = _new_dataset_id("invalid")
    await _register(client, ds)
    bad_payloads = [
        [_cand("c1", 0.1, 0.0, 0)],            # value must be >= 1
        [_cand("c1", 0.1, 0.0, -2)],
        [_cand("c1", 0.1, 0.0, 2.5)],          # value must be an integer
        [_cand("c1", 0.0, 0.0, 1)],            # epsilon must be > 0
        [_cand("c1", -0.1, 0.0, 1)],
        [_cand("c1", 0.1, -0.5, 1)],           # delta must be >= 0
        [_cand("c1", 0.1, 1.0, 1)],            # delta must be < 1
        [_cand("bad id!", 0.1, 0.0, 1)],       # illegal id characters
        [_cand("", 0.1, 0.0, 1)],              # empty id
        [{"candidate_id": "c1", "epsilon": 0.1, "delta": 0.0}],  # no value
    ]
    for payload in bad_payloads:
        r = await _plan(client, ds, payload)
        assert r.status_code == 422, payload
