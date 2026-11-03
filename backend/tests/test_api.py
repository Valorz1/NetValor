"""HTTP API behaviour: health, docs, CORS, calculations, validation and guards."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.api import calc as calc_routes
from tests.conftest import SAMPLE_PROFILE, ClientFactory, register_and_login


def test_api_health(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_database_and_lessons(client: TestClient) -> None:
    response = client.get("/api/health/ready")

    assert response.json() == {"status": "ok", "database": True, "lessons": 8}


def test_docs_are_available_in_development(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_docs_can_be_disabled(make_client: ClientFactory) -> None:
    client = make_client(docs_enabled=False)

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_cors_allows_local_frontend(client: TestClient) -> None:
    response = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-credentials" not in response.headers


def test_cors_rejects_unknown_origins(client: TestClient) -> None:
    response = client.options(
        "/api/health",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )

    assert "access-control-allow-origin" not in response.headers


def test_security_headers_and_request_id(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert len(response.headers["x-request-id"]) == 16


def test_oversized_request_body_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login",
        content=b'{"user_id": "' + b"a" * 70_000 + b'"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413


def test_oversized_chunked_body_is_rejected(client: TestClient) -> None:
    def chunks():  # no Content-Length header
        for _ in range(100):
            yield b"a" * 1024

    response = client.post(
        "/api/auth/login", content=chunks(), headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 413


def test_validation_errors_do_not_echo_submitted_values(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register", json={"user_id": "Not Valid!", "password": "short-secret"}
    )

    assert response.status_code == 422
    assert "short-secret" not in response.text
    assert "Not Valid!" not in response.text
    assert {tuple(error["loc"]) for error in response.json()["detail"]} >= {("body", "user_id")}


def test_unexpected_errors_return_generic_500(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client()

    def explode(*_args: object) -> None:
        raise RuntimeError("internal path /secret/db")

    monkeypatch.setattr(calc_routes, "debt_payoff", explode)
    response = TestClient(client.app, raise_server_exceptions=False).post(
        "/api/calc/debt-payoff", json={"balance": 1, "apr": 0.1, "monthly_payment": 1}
    )

    assert response.status_code == 500
    assert "secret" not in response.text
    assert "Reference:" in response.json()["detail"]


# --- Calculations --------------------------------------------------------------


def test_debt_payoff_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/calc/debt-payoff", json={"balance": 2450, "apr": 0.249, "monthly_payment": 350}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["months"] == 8
    assert Decimal(result["total_interest"]) == Decimal("225.48")
    assert result["pays_off"] is True
    assert result["reason"] is None


def test_debt_payoff_reports_why_it_never_pays_off(client: TestClient) -> None:
    response = client.post(
        "/api/calc/debt-payoff", json={"balance": 1000, "apr": 0.12, "monthly_payment": 10}
    )

    assert response.json()["reason"] == "payment_does_not_cover_interest"


def test_compare_extra_payment_endpoint_includes_payoff_dates(client: TestClient) -> None:
    response = client.post(
        "/api/calc/compare-extra",
        json={"balance": 2450, "apr": 0.249, "monthly_payment": 350, "extra": 120},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["months_saved"] == 2
    assert result["baseline_payoff_date"] is not None
    assert result["with_extra_payoff_date"] is not None
    assert any("daily interest" in note for note in result["assumptions"])


def test_emergency_fund_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/calc/emergency-fund", json={"saved": 3200, "monthly_essentials": 2000}
    )

    assert response.status_code == 200
    result = response.json()
    assert Decimal(result["months_covered"]) == Decimal("1.60")
    assert Decimal(result["amount_left"]) == Decimal("2800.00")


def test_goal_timeline_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/calc/goal-timeline", json={"current": 500, "target": 2000, "monthly_saving": 200}
    )

    assert response.status_code == 200
    assert response.json()["months"] == 8
    assert response.json()["target_date"] is not None


@pytest.mark.parametrize(
    ("path", "body", "field"),
    [
        ("/api/calc/debt-payoff", {"balance": -1, "apr": 0.249, "monthly_payment": 350}, "balance"),
        ("/api/calc/debt-payoff", {"balance": 1, "apr": 24.9, "monthly_payment": 350}, "apr"),
        ("/api/calc/debt-payoff", {"balance": 1e12, "apr": 0.1, "monthly_payment": 1}, "balance"),
        ("/api/calc/debt-payoff", {"balance": "NaN", "apr": 0.1, "monthly_payment": 1}, "balance"),
        (
            "/api/calc/debt-payoff",
            {"balance": "Infinity", "apr": 0.1, "monthly_payment": 1},
            "balance",
        ),
        ("/api/calc/debt-payoff", {"balance": 1, "apr": 0.1, "monthly_payment": 1, "x": 1}, "x"),
        ("/api/calc/emergency-fund", {"saved": 100, "monthly_essentials": 0}, "monthly_essentials"),
        (
            "/api/calc/emergency-fund",
            {"saved": 1, "monthly_essentials": 1, "target_months": 99},
            "target_months",
        ),
        (
            "/api/calc/compare-extra",
            {"balance": 1, "apr": 0.1, "monthly_payment": 1, "extra": -5},
            "extra",
        ),
    ],
)
def test_invalid_finance_input_returns_clear_422(
    client: TestClient, path: str, body: dict[str, object], field: str
) -> None:
    response = client.post(path, json=body)

    assert response.status_code == 422
    assert field in response.text


def test_nan_json_literal_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/calc/debt-payoff",
        content=b'{"balance": NaN, "apr": 0.1, "monthly_payment": 1}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_finance_value_error_message_is_returned(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_calculation(*_args: object, **_kwargs: object) -> None:
        raise ValueError("monthly payment must be greater than interest.")

    monkeypatch.setattr(calc_routes, "debt_payoff", reject_calculation)
    response = client.post(
        "/api/calc/debt-payoff", json={"balance": 1000, "apr": 0.12, "monthly_payment": 10}
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "monthly payment must be greater than interest."}


# --- Profiles and memory (authenticated) ------------------------------------------


def test_memory_create_edit_delete_flow(client: TestClient) -> None:
    headers = register_and_login(client)
    assert client.put("/api/profile/sam", json=SAMPLE_PROFILE, headers=headers).status_code == 200

    profile = client.get("/api/profile/sam", headers=headers).json()
    assert profile["monthly_take_home_income"] == "2800"
    assert profile["debts"][0]["balance"] == "2450"
    assert profile["goals"][0]["target_date"] == "2027-06-01"

    created = client.post(
        "/api/memory/sam", json={"content": "  Wants a simple monthly check-in "}, headers=headers
    )
    assert created.status_code == 201
    assert created.json()["content"] == "Wants a simple monthly check-in"
    note_id = created.json()["note_id"]

    listed = client.get("/api/memory/sam", headers=headers)
    assert [note["note_id"] for note in listed.json()["notes"]] == [note_id]

    edited = client.patch(
        f"/api/memory/sam/{note_id}", json={"content": "Prefers a short check-in"}, headers=headers
    )
    assert edited.status_code == 200
    assert edited.json()["content"] == "Prefers a short check-in"

    deleted = client.delete(f"/api/memory/sam/{note_id}", headers=headers)
    assert deleted.json() == {"deleted": True}
    assert client.get("/api/memory/sam", headers=headers).json()["notes"] == []
    assert client.delete(f"/api/memory/sam/{note_id}", headers=headers).status_code == 404


def test_clear_all_memory(client: TestClient) -> None:
    headers = register_and_login(client)
    for text in ("one", "two"):
        client.post("/api/memory/sam", json={"content": text}, headers=headers)

    response = client.delete("/api/memory/sam", headers=headers)

    assert response.json() == {"deleted": 2}
    assert client.get("/api/memory/sam", headers=headers).json()["notes"] == []


def test_profile_put_creates_profile_and_get_404_before(client: TestClient) -> None:
    headers = register_and_login(client)
    assert client.get("/api/profile/sam", headers=headers).status_code == 404

    saved = client.put("/api/profile/sam", json=SAMPLE_PROFILE, headers=headers)

    assert saved.status_code == 200
    assert client.get("/api/profile/sam", headers=headers).json()["name"] == "Sam"


@pytest.mark.parametrize(
    "change",
    [
        {"name": ""},
        {"name": "x" * 81},
        {"savings": -1},
        {"savings": "1e30"},
        {"debts": [{"name": "Card", "balance": 1, "apr": 2, "monthly_payment": 1}]},
        {"debts": [SAMPLE_PROFILE["debts"][0]] * 21},
        {"unexpected": "field"},
    ],
)
def test_profile_validation(client: TestClient, change: dict[str, object]) -> None:
    headers = register_and_login(client)

    response = client.put("/api/profile/sam", json={**SAMPLE_PROFILE, **change}, headers=headers)

    assert response.status_code == 422


@pytest.mark.parametrize("content", ["", "   ", "x" * 501])
def test_note_validation(client: TestClient, content: str) -> None:
    headers = register_and_login(client)

    assert (
        client.post("/api/memory/sam", json={"content": content}, headers=headers).status_code
        == 422
    )
