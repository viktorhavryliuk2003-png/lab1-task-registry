from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Task Registry API (Lab 1)",
    description="Локальний HTTP-сервіс реєстру задач із перевіркою AI-пропозицій",
    version="1.1.0"
)

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


def get_next_friday(base_date: datetime | None = None) -> str:
    tz = ZoneInfo("Europe/Kyiv")
    now = base_date or datetime.now(tz)
    weekday = now.weekday()
    if weekday <= 4:
        days_ahead = 4 - weekday
    else:
        days_ahead = 7 - (weekday - 4)
    target = now + timedelta(days=days_ahead)
    return target.strftime("%Y-%m-%d")


class TaskCreateManual(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    assignee: str = Field(..., min_length=1)
    due_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")

class TaskFromTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
ASSIGNEE_RE = re.compile(r"(?:відповідальний|виконавець|закріплено за|домовилися, що)\s*:?\s*([А-Яа-яA-Za-zІіЇїЄє']+)", re.IGNORECASE)


def fallback_entity_extractor(text: str) -> dict[str, Any]:
    """Детермінований локальний парсер-пропонувач (внутрішній агент-fallback)."""
    dates = DATE_RE.findall(text)
    due_date = dates[0] if dates else None

    assignee_match = ASSIGNEE_RE.search(text)
    assignee = assignee_match.group(1).strip() if assignee_match else None

    # Витягуємо назву завдання (перша змістовна частина до крапки з комою чи ключових слів)
    raw_title = text.split(";")[0].strip()
    raw_title = re.sub(r"^(?:домовилися, що|потрібно|треба)\s+", "", raw_title, flags=re.IGNORECASE)
    title = raw_title[:120] if raw_title else "Нове завдання"

    return {
        "title": title,
        "description": None,
        "assignee": assignee,
        "due_date": due_date
    }


@app.get("/tasks", response_model=list[dict[str, Any]])
def get_tasks():
    return load_storage()


@app.post("/tasks", status_code=status.HTTP_201_CREATED, response_model=dict[str, Any])
def create_task_manual(payload: TaskCreateManual):
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
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Текст не може бути порожнім")

    # Перевірка на суперечність дат у вхідному тексті
    dates_in_text = DATE_RE.findall(text)
    if len(set(dates_in_text)) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "needs_clarification", "reason": "contradictory_dates_in_text"}
        )

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    model_name = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-thinking-exp:free")

    proposal = None

    # Спроба отримати структуру від зовнішньої моделі
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0, max_retries=0)
            
            prompt = (
                "Extract task fields in raw JSON only: "
                "{\"title\": string, \"description\": null, \"assignee\": string|null, \"due_date\": string|null}. "
                "Do not hallucinate."
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                proposal = json.loads(match.group(0))
        except Exception:
            # Зовнішній провайдер дав 404 / таймаут — перемикаємося на внутрішній fallback
            proposal = None

    # Якщо провайдер недоступний, використовуємо резервний парсер-пропонувач
    if not proposal:
        proposal = fallback_entity_extractor(text)

    title = proposal.get("title")
    description = proposal.get("description")
    assignee = proposal.get("assignee")
    due_date = proposal.get("due_date")

    # Валідація обов'язкових сутностей
    if not title or not assignee:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "needs_clarification", "reason": "missing_required_title_or_assignee"}
        )

    # Детермінована перевірка: відповідальний має бути буквально в тексті
    if assignee not in text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "rejected_ungrounded", "reason": "hallucinated_assignee"}
        )

    # CR-1: якщо строк відсутній у повідомленні — детерміновано рахуємо найближчу п'ятницю
    inferred_friday = False
    if not due_date:
        due_date = get_next_friday()
        inferred_friday = True
    else:
        if not DATE_RE.fullmatch(due_date) or due_date not in text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"status": "rejected_ungrounded", "reason": "hallucinated_due_date"}
            )

    tasks = load_storage()
    new_id = f"task-ai-{len(tasks) + 1}"
    task_record = {
        "id": new_id,
        "title": str(title).strip(),
        "description": description,
        "assignee": str(assignee).strip(),
        "due_date": due_date,
        "inferred_due_date": inferred_friday,
    }
    tasks.append(task_record)
    save_storage(tasks)
    return task_record