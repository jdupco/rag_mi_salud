from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hola",
                "conversation_id": "test_conversation_id",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0


def test_chat_requires_message():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": "test_conversation_id",
            },
        )

    assert response.status_code == 422


def test_chat_rejects_empty_message():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "",
                "conversation_id": "test_conversation_id",
            },
        )

    assert response.status_code == 422