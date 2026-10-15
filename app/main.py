from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Task Registry API (Lab 1)",
    description="Локальний HTTP-сервіс реєстру задач із перевіркою AI-пропозицій",
    version="1.0.0"
)

# Шлях до локального сховища
STORAGE_PATH = Path("data") / "tasks.json"

def load_storage() -> list[dict[str, Any]]:
    if not STORAGE_PATH.exists():
        return []
    try:
        return json.loads(STORAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_storage(tasks: list[dict[str, Any]]) -> None:
    STORAGE_PATH.parent.mkdir(exist_ok=True, parents=True)
    STORAGE_PATH.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


# Pydantic моделі контракту
class TaskCreateManual(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    assignee: str = Field(..., min_length=1)
    due_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")

class TaskFromTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@app.get("/tasks", response_model=list[dict[str, Any]])
def get_tasks():
    """Перегляд списку всіх задач (ручний шлях, не залежить від LLM)."""
    return load_storage()


@app.post("/tasks", status_code=status.HTTP_201_CREATED, response_model=dict[str, Any])
def create_task_manual(payload: TaskCreateManual):
    """Ручне створення задачі з повним набором полів."""
    tasks = load_storage()
    new_id = f"task-{len(tasks) + 1}-{abs(hash(payload.title)) % 10000}"
    
    task_record = {
        "id": new_id,
        "title": payload.title.strip(),
        "description": payload.description,
        "assignee": payload.assignee.strip(),
        "due_date": payload.due_date,
    }
    tasks.append(task_record)
    save_storage(tasks)
    return task_record


@app.post("/tasks/from-text", status_code=status.HTTP_201_CREATED, response_model=dict[str, Any])
def create_task_from_text(payload: TaskFromTextRequest):
    """Створення задачі з тексту за допомогою LLM з жорстким перевірочним конвеєром у коді."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Текст не може бути порожнім")

    # 1. Виклик LLM через OpenAI SDK (OpenRouter або OpenAI)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    model_name = os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "provider_unavailable", "reason": "Ключ API не налаштовано в середовищі"}
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=20.0, max_retries=0)
        
        system_prompt = (
            "Виділи із тексту задачі поля у форматі JSON: "
            "{\"title\": string, \"description\": string|null, \"assignee\": string|null, \"due_date\": string|null (YYYY-MM-DD)}. "
            "КРИТИЧНЕ ПРАВИЛО: Ніколи не вигадуй дані. Якщо відповідального або дедлайну немає у тексті, поверни null."
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw_content = response.choices[0].message.content
        proposal = json.loads(raw_content)
    except Exception as err:
        # У разі збою провайдера чи таймауту повертаємо статус збою без зміни сховища
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "provider_unavailable", "reason": type(err).__name__}
        )

    # 2. Детермінована валідація та перевірка підтвердження текстом (Grounding)
    title = proposal.get("title")
    description = proposal.get("description")
    assignee = proposal.get("assignee")
    due_date = proposal.get("due_date")

    # Перевірка наявності обов'язкових полів або конфліктів у тексті
    dates_in_text = DATE_RE.findall(text)
    if len(set(dates_in_text)) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "needs_clarification", "reason": "contradictory_dates_in_text"}
        )

    if not title or not assignee or not due_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "needs_clarification", "reason": "missing_required_fields_or_clarification_needed"}
        )

    # Перевірка формату дати та буквального підтвердження у вхідному тексті (захист від галюцинацій)
    if not DATE_RE.fullmatch(due_date):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "rejected_ungrounded", "reason": "invalid_date_format"}
        )

    if assignee not in text or due_date not in text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "rejected_ungrounded", "reason": "hallucinated_field_not_found_in_text"}
        )

    # 3. Збереження у сховище, якщо всі правила пройдено
    tasks = load_storage()
    new_id = f"task-ai-{len(tasks) + 1}"
    task_record = {
        "id": new_id,
        "title": str(title).strip(),
        "description": description,
        "assignee": str(assignee).strip(),
        "due_date": due_date,
    }
    tasks.append(task_record)
    save_storage(tasks)
    return task_record