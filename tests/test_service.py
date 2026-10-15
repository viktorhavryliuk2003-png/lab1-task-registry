from __future__ import annotations

import os
from fastapi.testclient import TestClient
import pytest

from app.main import app, STORAGE_PATH

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_storage():
    """Перед кожним тестом очищуємо тестове сховище."""
    if STORAGE_PATH.exists():
        STORAGE_PATH.unlink()
    yield
    if STORAGE_PATH.exists():
        STORAGE_PATH.unlink()


def test_get_tasks_initial_empty():
    """Сценарій: початковий список задач порожній і повертає JSON-масив."""
    response = client.get("/tasks")
    assert response.status_code == 200
    assert response.json() == []


def test_manual_task_creation_success():
    """Сценарій: успішне ручне створення задачі."""
    payload = {
        "title": "Провести ретроспективу спринту",
        "description": "Обговорити результати тижня",
        "assignee": "Ірина",
        "due_date": "2026-10-12"
    }
    response = client.post("/tasks", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == payload["title"]
    assert data["assignee"] == "Ірина"
    assert data["due_date"] == "2026-10-12"

    # Перевіряємо, що запис з'явився у списку
    list_resp = client.get("/tasks")
    assert len(list_resp.json()) == 1
    assert list_resp.json()[0]["id"] == data["id"]


def test_manual_task_invalid_data_rejected():
    """Сценарій: хібно сформований ручний вхід відхиляється з кодом 422, база не змінюється."""
    payload = {
        "title": "Х",  # занадто коротка назва (менше 3 символів)
        "assignee": "Максим",
        "due_date": "неправильна-дата"  # не відповідає YYYY-MM-DD
    }
    response = client.post("/tasks", json=payload)
    assert response.status_code == 422

    # Перевіряємо, що сховище залишилося порожнім
    list_resp = client.get("/tasks")
    assert list_resp.json() == []


def test_ai_endpoint_without_api_key_returns_service_unavailable():
    """Сценарій: запит до AI-маршруту без налаштованого ключа дає 503 (збій провайдера / fallback)."""
    # Гарантуємо відсутність ключів в середовищі для цього тесту
    old_or = os.environ.pop("OPENROUTER_API_KEY", None)
    old_ai = os.environ.pop("OPENAI_API_KEY", None)
    try:
        response = client.post("/tasks/from-text", json={"text": "Підготувати демо; відповідальний: Ірина; строк: 2026-10-09"})
        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "provider_unavailable"
    finally:
        if old_or:
            os.environ["OPENROUTER_API_KEY"] = old_or
        if old_ai:
            os.environ["OPENAI_API_KEY"] = old_ai