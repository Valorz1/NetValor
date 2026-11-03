"""Authentication and authorisation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import PASSWORD, SAMPLE_PROFILE, ClientFactory, register_and_login

USER_ROUTES = [
    ("GET", "/api/profile/{user}", None),
    ("PUT", "/api/profile/{user}", SAMPLE_PROFILE),
    ("GET", "/api/memory/{user}", None),
    ("POST", "/api/memory/{user}", {"content": "note"}),
    ("DELETE", "/api/memory/{user}", None),
    ("PATCH", "/api/memory/{user}/some-note", {"content": "note"}),
    ("DELETE", "/api/memory/{user}/some-note", None),
    ("POST", "/api/chat/{user}", {"message": "hello"}),
    ("GET", "/api/conversations/{user}", None),
    ("GET", "/api/conversations/{user}/some-conversation", None),
    ("DELETE", "/api/conversations/{user}/some-conversation", None),
]


@pytest.mark.parametrize(("method", "path", "body"), USER_ROUTES)
def test_user_routes_require_authentication(
    client: TestClient, method: str, path: str, body: object
) -> None:
    response = client.request(method, path.format(user="sam"), json=body)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(("method", "path", "body"), USER_ROUTES)
def test_users_cannot_access_each_others_data(
    client: TestClient, method: str, path: str, body: object
) -> None:
    victim = register_and_login(client, "alex")
    client.put("/api/profile/alex", json=SAMPLE_PROFILE, headers=victim)
    attacker = register_and_login(client, "mallory")

    response = client.request(method, path.format(user="alex"), json=body, headers=attacker)

    assert response.status_code == 403
    assert "2450" not in response.text


def test_invalid_and_malformed_tokens_are_rejected(client: TestClient) -> None:
    for header in ("Bearer not-a-real-token", "Bearer " + "x" * 300, "Basic abc", "Bearer"):
        response = client.get("/api/memory/sam", headers={"Authorization": header})
        assert response.status_code == 401


def test_register_login_me_logout(client: TestClient) -> None:
    headers = register_and_login(client)

    assert client.get("/api/auth/me", headers=headers).json() == {"user_id": "sam"}
    assert client.post("/api/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_login_response_shape(client: TestClient) -> None:
    client.post("/api/auth/register", json={"user_id": "sam", "password": PASSWORD})

    body = client.post("/api/auth/login", json={"user_id": "sam", "password": PASSWORD}).json()

    assert body["token_type"] == "bearer"
    assert body["user_id"] == "sam"
    assert len(body["access_token"]) >= 40
    assert body["expires_at"]


def test_wrong_password_and_unknown_user_look_identical(client: TestClient) -> None:
    client.post("/api/auth/register", json={"user_id": "sam", "password": PASSWORD})

    wrong = client.post("/api/auth/login", json={"user_id": "sam", "password": "nope nope nope"})
    unknown = client.post(
        "/api/auth/login", json={"user_id": "nobody", "password": "nope nope nope"}
    )

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_duplicate_registration_conflicts(client: TestClient) -> None:
    register_and_login(client)

    response = client.post("/api/auth/register", json={"user_id": "sam", "password": PASSWORD})

    assert response.status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"user_id": "sam", "password": "short"},
        {"user_id": "sam", "password": "aaaaaaaaaaaaaaaa"},
        {"user_id": "sam", "password": "x" * 129},
        {"user_id": "Sam", "password": PASSWORD},
        {"user_id": "sammy-the-user", "password": "SAMMY-THE-USER"},
        {"user_id": "sam", "password": PASSWORD, "is_admin": True},
    ],
)
def test_registration_validation(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/api/auth/register", json=body).status_code == 422


def test_registration_can_be_disabled(make_client: ClientFactory) -> None:
    client = make_client(allow_registration=False)

    response = client.post("/api/auth/register", json={"user_id": "sam", "password": PASSWORD})

    assert response.status_code == 403


def test_login_is_rate_limited(make_client: ClientFactory) -> None:
    client = make_client(auth_rate_limit_per_minute=3)
    attempts = [
        client.post("/api/auth/login", json={"user_id": "sam", "password": "guess guess guess"})
        for _ in range(4)
    ]

    assert [response.status_code for response in attempts] == [401, 401, 401, 429]
    assert int(attempts[-1].headers["retry-after"]) >= 1


def test_delete_account_requires_password_and_removes_data(client: TestClient) -> None:
    headers = register_and_login(client)
    client.put("/api/profile/sam", json=SAMPLE_PROFILE, headers=headers)

    wrong = client.post(
        "/api/auth/delete-account", json={"password": "not my password"}, headers=headers
    )
    assert wrong.status_code == 401
    assert client.get("/api/profile/sam", headers=headers).status_code == 200

    deleted = client.post("/api/auth/delete-account", json={"password": PASSWORD}, headers=headers)
    assert deleted.status_code == 204
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    assert (
        client.post("/api/auth/login", json={"user_id": "sam", "password": PASSWORD}).status_code
        == 401
    )


def test_demo_user_is_not_seeded_by_default(client: TestClient) -> None:
    response = client.post("/api/auth/login", json={"user_id": "alex", "password": PASSWORD})

    assert response.status_code == 401


def test_demo_user_requires_an_explicit_password(make_client: ClientFactory) -> None:
    client = make_client(seed_demo_user=True, demo_user_password="too short")

    # The account is skipped, so registering the ID is still possible.
    response = client.post("/api/auth/register", json={"user_id": "alex", "password": PASSWORD})

    assert response.status_code == 201


def test_demo_user_seeded_with_configured_password(make_client: ClientFactory) -> None:
    client = make_client(seed_demo_user=True, demo_user_password="demo password 123")

    login = client.post(
        "/api/auth/login", json={"user_id": "alex", "password": "demo password 123"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    assert client.get("/api/profile/alex", headers=headers).json()["name"] == "Alex"
    assert len(client.get("/api/memory/alex", headers=headers).json()["notes"]) == 1
