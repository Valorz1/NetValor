from decimal import Decimal

from fastapi.testclient import TestClient

from app import main
from app.memory import JsonProfileStore


def test_api_health() -> None:
    with TestClient(main.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_are_available() -> None:
    with TestClient(main.app) as client:
        response = client.get("/docs")

    assert response.status_code == 200


def test_cors_allows_local_frontend() -> None:
    with TestClient(main.app) as client:
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_debt_payoff_endpoint() -> None:
    with TestClient(main.app) as client:
        response = client.post(
            "/api/calc/debt-payoff",
            json={"balance": 2450, "apr": 0.249, "monthly_payment": 350},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["months"] == 8
    assert Decimal(result["total_interest"]) == Decimal("225.48")
    assert result["pays_off"] is True


def test_compare_extra_payment_endpoint_includes_payoff_dates() -> None:
    with TestClient(main.app) as client:
        response = client.post(
            "/api/calc/compare-extra",
            json={
                "balance": 2450,
                "apr": 0.249,
                "monthly_payment": 350,
                "extra": 120,
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["months_saved"] == 2
    assert result["baseline_payoff_date"] is not None
    assert result["with_extra_payoff_date"] is not None


def test_emergency_fund_endpoint() -> None:
    with TestClient(main.app) as client:
        response = client.post(
            "/api/calc/emergency-fund",
            json={"saved": 3200, "monthly_essentials": 2000},
        )

    assert response.status_code == 200
    result = response.json()
    assert Decimal(result["months_covered"]) == Decimal("1.60")
    assert Decimal(result["amount_left"]) == Decimal("2800.00")


def test_invalid_finance_input_returns_clear_422() -> None:
    with TestClient(main.app) as client:
        response = client.post(
            "/api/calc/debt-payoff",
            json={"balance": -1, "apr": 0.249, "monthly_payment": 350},
        )

    assert response.status_code == 422
    assert "balance" in response.text


def test_finance_value_error_returns_clear_422() -> None:
    with TestClient(main.app) as client:
        response = client.post(
            "/api/calc/emergency-fund",
            json={"saved": 100, "monthly_essentials": 0},
        )

    assert response.status_code == 422
    assert "monthly_essentials" in response.text


def test_finance_value_error_message_is_returned(monkeypatch) -> None:
    def reject_calculation(*_args, **_kwargs):
        raise ValueError("monthly payment must be greater than interest.")

    monkeypatch.setattr(main, "debt_payoff", reject_calculation)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/calc/debt-payoff",
            json={"balance": 1000, "apr": 0.12, "monthly_payment": 10},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "monthly payment must be greater than interest."
    }


def test_memory_create_edit_delete_flow(tmp_path) -> None:
    original_store = main.profile_store
    main.profile_store = JsonProfileStore(tmp_path)
    try:
        with TestClient(main.app) as client:
            profile_response = client.get("/api/profile/alex")
            assert profile_response.status_code == 200
            profile = profile_response.json()
            assert profile["monthly_take_home_income"] == "3600"
            assert profile["monthly_essentials"] == "2000"
            assert profile["savings"] == "3200"
            assert profile["debts"][0]["balance"] == "2450"
            assert profile["goals"][0]["target_date"] == "2027-06-01"

            created = client.post(
                "/api/memory/alex",
                json={"content": "Wants a simple monthly check-in"},
            )
            assert created.status_code == 201
            note_id = created.json()["note_id"]

            listed = client.get("/api/memory/alex")
            assert any(note["note_id"] == note_id for note in listed.json()["notes"])

            edited = client.patch(
                f"/api/memory/alex/{note_id}",
                json={"content": "Prefers a short monthly check-in"},
            )
            assert edited.status_code == 200
            assert edited.json()["content"] == "Prefers a short monthly check-in"

            deleted = client.delete(f"/api/memory/alex/{note_id}")
            assert deleted.status_code == 200
            assert deleted.json() == {"deleted": True}

            listed_after_delete = client.get("/api/memory/alex")
            assert all(
                note["note_id"] != note_id
                for note in listed_after_delete.json()["notes"]
            )
    finally:
        main.profile_store = original_store


def test_profile_put_creates_profile(tmp_path) -> None:
    original_store = main.profile_store
    main.profile_store = JsonProfileStore(tmp_path)
    profile = {
        "name": "Sam",
        "monthly_take_home_income": 2800,
        "monthly_essentials": 1700,
        "savings": 500,
        "debts": [],
        "goals": [],
    }
    try:
        with TestClient(main.app) as client:
            saved = client.put("/api/profile/sam", json=profile)
            loaded = client.get("/api/profile/sam")

        assert saved.status_code == 200
        assert saved.json()["name"] == "Sam"
        assert loaded.status_code == 200
        assert loaded.json()["monthly_take_home_income"] == "2800"
    finally:
        main.profile_store = original_store
