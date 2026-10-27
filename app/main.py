from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Завантажуємо налаштування середовища
load_dotenv(override=True)

app = FastAPI(
    title="Task Registry API (Lab 1)",
    description="HTTP-сервіс реєстру задач на базі нейромережі через Groq (OpenAI-сумісний API)",
    version="2.5.0"
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
ASSIGNEE_RE = re.compile(
    r"(?:відповідальний|виконавець|закріплено за|домовилися, що)\s*:?\s*([А-Яа-яA-Za-zІіЇїЄє']+)",
    re.IGNORECASE
)


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

    # 1. ПЕРЕВІРКА НА СУПЕРЕЧНІСТЬ ДАТ (Вимога M06 -> 422)
    dates_in_text = DATE_RE.findall(text)
    if len(set(dates_in_text)) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "needs_clarification", "reason": "contradictory_dates_in_text"}
        )

    # 2. ПЕРЕВІРКА НА СУПЕРЕЧНІСТЬ ВІДПОВІДАЛЬНИХ (Вимога M07 -> 422)
    assignees_found = [a.strip() for a in ASSIGNEE_RE.findall(text)]
    if len(set(assignees_found)) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "needs_clarification", "reason": "contradictory_assignees_in_text"}
        )

    # Зчитування API-ключа (перевірка сумісна з тестами, які видаляють OPENAI_API_KEY або OPENROUTER_API_KEY)
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
    )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "provider_unavailable", "reason": "Ключ API не налаштовано в середовищі"}
        )

    # Маршрутизація провайдера
    if api_key.startswith("sk-or-v1-"):
        base_url = "https://openrouter.ai/api/v1"
        model_name = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    else:
        # За замовчуванням швидка хмара Groq
        base_url = "https://api.groq.com/openai/v1"
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    proposal = None
    today_iso = datetime.now(ZoneInfo("Europe/Kyiv")).strftime("%Y-%m-%d")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=15.0, max_retries=1)

        print(f"\n[AI NEURAL NET]: Відправляємо задачу в нейромережу {model_name}...")

        prompt = (
            f"Reference date: today is {today_iso}.\n"
            "Task: Extract task details from user text and return ONLY raw valid JSON with keys:\n"
            "\"title\": string (short clean name of the action/task in the EXACT language of the user input, do not translate into English),\n"
            "\"description\": string or null (additional details, context or notes if mentioned in the text, otherwise null),\n"
            "\"assignee\": string or null (person name responsible),\n"
            "\"due_date\": string or null (format YYYY-MM-DD. Convert words like 'до кінця грудня' to 2026-12-31, 'до 29 жовтня' to 2026-10-29. If no deadline mentioned, return null).\n"
            "Output strictly JSON without markdown."
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )

        raw = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            proposal = json.loads(match.group(0))
            print(f"[AI NEURAL NET SUCCESS]: Відповідь успішно згенерована нейромережею:\n{json.dumps(proposal, ensure_ascii=False, indent=2)}\n")

    except Exception as e:
        print(f"\n[AI NEURAL NET ERROR]: {e}\n")
        proposal = None

    if not proposal:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "provider_unavailable", "reason": "Нейромережа не відповіла на запит"}
        )

    title = proposal.get("title")
    description = proposal.get("description")
    assignee = proposal.get("assignee")
    due_date = proposal.get("due_date")

    # Валідація обов'язкових полів
    if not title or not assignee:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "needs_clarification", "reason": "missing_required_title_or_assignee"}
        )

    # Детермінований захист (grounding): виконавець обов'язково має бути присутній у вихідному тексті
    if assignee not in text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"status": "rejected_ungrounded", "reason": "hallucinated_assignee"}
        )

    inferred_friday = False
    if not due_date:
        due_date = get_next_friday()
        inferred_friday = True
    else:
        if not DATE_RE.fullmatch(str(due_date)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"status": "rejected_ungrounded", "reason": "invalid_date_format"}
            )
        if dates_in_text and due_date not in text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"status": "rejected_ungrounded", "reason": "hallucinated_due_date"}
            )

    tasks = load_storage()
    new_id = f"task-ai-{len(tasks) + 1}"
    task_record = {
        "id": new_id,
        "title": str(title).strip(),
        "description": str(description).strip() if description else None,
        "assignee": str(assignee).strip(),
        "due_date": due_date,
        "inferred_due_date": inferred_friday,
    }
    tasks.append(task_record)
    save_storage(tasks)
    return task_record