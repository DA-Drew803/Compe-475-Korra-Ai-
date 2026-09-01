from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from app.graph import graph

DATA_DIR = Path(os.getenv("KORRA_DATA_DIR", "/data"))
CHAT_DB = DATA_DIR / "korra_chat.db"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("korra-api")

app = FastAPI(title="Korra Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ThreadResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    response: str


class NoteItem(BaseModel):
    id: int
    category: str
    note: str


class NotesResponse(BaseModel):
    notes: List[NoteItem]


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CHAT_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/threads", response_model=ThreadResponse)
def create_thread() -> ThreadResponse:
    thread_id = str(uuid.uuid4())
    with sqlite3.connect(CHAT_DB) as conn:
        conn.execute("INSERT INTO threads(thread_id) VALUES (?)", (thread_id,))
        conn.commit()
    return ThreadResponse(thread_id=thread_id)


def load_history(thread_id: str) -> list[BaseMessage]:
    with sqlite3.connect(CHAT_DB) as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()

    messages: list[BaseMessage] = []
    for role, content in rows:
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


def save_message(thread_id: str, role: str, content: str) -> None:
    with sqlite3.connect(CHAT_DB) as conn:
        conn.execute(
            "INSERT INTO conversations(thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        )
        conn.commit()


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    with sqlite3.connect(CHAT_DB) as conn:
        exists = conn.execute(
            "SELECT 1 FROM threads WHERE thread_id = ?",
            (payload.thread_id,),
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    history = load_history(payload.thread_id)
    user_message = HumanMessage(content=payload.message)
    result = graph.invoke({"messages": [*history, user_message]})
    messages = result["messages"]

    final_text = "I was unable to generate a response."
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            final_text = message.content
            break

    save_message(payload.thread_id, "human", payload.message)
    save_message(payload.thread_id, "ai", final_text)
    return ChatResponse(thread_id=payload.thread_id, response=final_text)


@app.get("/notes", response_model=NotesResponse)
def get_notes() -> NotesResponse:
    db_path = DATA_DIR / "korra_memory.db"
    if not db_path.exists():
        return NotesResponse(notes=[])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, category, note FROM notes ORDER BY id DESC"
        ).fetchall()

    notes = [NoteItem(id=row[0], category=row[1], note=row[2]) for row in rows]
    return NotesResponse(notes=notes)
