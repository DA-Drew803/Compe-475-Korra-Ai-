from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ==========================================
# Module 13 - Section 1
# Database Search Agent (Memory/Load-Store Unit)
# ==========================================

# Make the local tools folder importable even when graph.py is loaded directly
CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_MODEL = "gpt-4o-mini"
DB_PATH = Path("/app/data/database_search_agent.db")

SYSTEM_PROMPT = """
You are Korra-DB, a specialized database search and memory agent.

ROLE
You are not a general-purpose coding or analysis assistant.
You are optimized for memory-style operations, similar to a processor's load-store unit.
Your main job is to store, retrieve, search, and query structured information efficiently.

SPECIALIZED CAPABILITIES
You have access to database tools for:
1. Conversation history storage
2. Conversation history retrieval
3. Student record lookup
4. Course catalog search
5. General database search queries

BEHAVIORAL PRIORITIES
- Prioritize accurate data access over long explanations.
- Use tools whenever the user asks to store, retrieve, search, query, or look up information.
- Keep responses concise and database-focused.
- Avoid unnecessary reasoning, speculation, or broad decision-making.
- Do not act like a coding tutor unless the request is directly about stored data.

LOAD-STORE / MEMORY ANALOGY
Treat requests like memory operations:
- STORE: save new information to persistent storage
- LOAD: retrieve exact stored information
- SEARCH: locate matching records
- QUERY: filter structured data efficiently

TOOL SELECTION GUIDELINES
- Use save_conversation_memory when the user asks to save, remember, or store conversation information.
- Use get_conversation_history when the user asks to retrieve prior conversations or memory.
- Use query_student_records for student-related structured lookups like name, ID, major, or GPA.
- Use search_course_catalog for course-related lookups like course code, title, or description.
- Use general_database_search for broader keyword searches across stored tables.

RESPONSE RULES
- Prefer direct factual answers from the database.
- When returning records, present them clearly and compactly.
- If no matching data is found, say so plainly.
- Do not invent records or query results.
- If a request cannot be satisfied from available data, explain that briefly.

SAFETY
- Respect privacy and only return data present in the database.
- Do not fabricate sensitive data.
- Do not claim persistence succeeded unless the database tool returned success.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("korra-db")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    log.error("Missing required API key in environment")
    sys.exit("Missing OPENAI_API_KEY in your environment (.env).")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def init_database() -> None:
    """Initialize the local SQLite database with memory, student, and course tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                major TEXT NOT NULL,
                gpa REAL NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS course_catalog (
                course_code TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                units INTEGER NOT NULL
            )
            """
        )

        # Seed sample student data
        conn.execute(
            """
            INSERT OR IGNORE INTO students (student_id, name, major, gpa)
            VALUES
                ('S1001', 'Elias Rivera', 'Computer Engineering', 3.72),
                ('S1002', 'Maya Chen', 'Electrical Engineering', 3.91),
                ('S1003', 'Jordan Patel', 'Computer Science', 3.65),
                ('S1004', 'Sophia Nguyen', 'Mechanical Engineering', 3.84)
            """
        )

        # Seed sample course data
        conn.execute(
            """
            INSERT OR IGNORE INTO course_catalog (course_code, title, description, units)
            VALUES
                ('COMPE475', 'AI Agents', 'Design and implementation of intelligent software agents.', 3),
                ('EE410', 'Signals and Systems', 'Continuous and discrete-time signal analysis and systems.', 3),
                ('COMPE271', 'Computer Architecture', 'Processor organization, memory hierarchy, and performance.', 3),
                ('CS330', 'Database Systems', 'Relational databases, SQL, indexing, and query processing.', 3)
            """
        )

        conn.commit()


@tool
def save_conversation_memory(topic: str, content: str) -> str:
    """Store conversation information for later retrieval."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO conversation_memory(topic, content) VALUES (?, ?)",
            (topic, content),
        )
        conn.commit()
    return f"Stored conversation memory under topic '{topic}'."


@tool
def get_conversation_history(topic: str = "") -> str:
    """Retrieve prior conversation history. If topic is blank, return recent memories."""
    with sqlite3.connect(DB_PATH) as conn:
        if topic.strip():
            rows = conn.execute(
                """
                SELECT id, topic, content, created_at
                FROM conversation_memory
                WHERE topic LIKE ?
                ORDER BY created_at DESC
                """,
                (f"%{topic}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, topic, content, created_at
                FROM conversation_memory
                ORDER BY created_at DESC
                LIMIT 10
                """
            ).fetchall()

    if not rows:
        return "No conversation history found."

    result = [
        {
            "id": row[0],
            "topic": row[1],
            "content": row[2],
            "created_at": row[3],
        }
        for row in rows
    ]
    return json.dumps(result, indent=2)


@tool
def query_student_records(
    student_id: str = "",
    name: str = "",
    major: str = "",
) -> str:
    """Query structured student records by student ID, name, or major."""
    query = """
        SELECT student_id, name, major, gpa
        FROM students
        WHERE 1=1
    """
    params: list[str] = []

    if student_id.strip():
        query += " AND student_id LIKE ?"
        params.append(f"%{student_id}%")
    if name.strip():
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if major.strip():
        query += " AND major LIKE ?"
        params.append(f"%{major}%")

    query += " ORDER BY name"

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return "No matching student records found."

    result = [
        {
            "student_id": row[0],
            "name": row[1],
            "major": row[2],
            "gpa": row[3],
        }
        for row in rows
    ]
    return json.dumps(result, indent=2)


@tool
def search_course_catalog(keyword: str = "", course_code: str = "") -> str:
    """Search the course catalog by keyword or course code."""
    query = """
        SELECT course_code, title, description, units
        FROM course_catalog
        WHERE 1=1
    """
    params: list[str] = []

    if course_code.strip():
        query += " AND course_code LIKE ?"
        params.append(f"%{course_code}%")

    if keyword.strip():
        query += " AND (title LIKE ? OR description LIKE ?)"
        params.append(f"%{keyword}%")
        params.append(f"%{keyword}%")

    query += " ORDER BY course_code"

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return "No matching course records found."

    result = [
        {
            "course_code": row[0],
            "title": row[1],
            "description": row[2],
            "units": row[3],
        }
        for row in rows
    ]
    return json.dumps(result, indent=2)


@tool
def general_database_search(keyword: str) -> str:
    """Search across conversation memory, student records, and course catalog."""
    with sqlite3.connect(DB_PATH) as conn:
        memory_rows = conn.execute(
            """
            SELECT topic, content, created_at
            FROM conversation_memory
            WHERE topic LIKE ? OR content LIKE ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()

        student_rows = conn.execute(
            """
            SELECT student_id, name, major, gpa
            FROM students
            WHERE student_id LIKE ? OR name LIKE ? OR major LIKE ?
            ORDER BY name
            """,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()

        course_rows = conn.execute(
            """
            SELECT course_code, title, description, units
            FROM course_catalog
            WHERE course_code LIKE ? OR title LIKE ? OR description LIKE ?
            ORDER BY course_code
            """,
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()

    result = {
        "conversation_memory": [
            {"topic": row[0], "content": row[1], "created_at": row[2]}
            for row in memory_rows
        ],
        "students": [
            {"student_id": row[0], "name": row[1], "major": row[2], "gpa": row[3]}
            for row in student_rows
        ],
        "courses": [
            {"course_code": row[0], "title": row[1], "description": row[2], "units": row[3]}
            for row in course_rows
        ],
    }

    if not result["conversation_memory"] and not result["students"] and not result["courses"]:
        return f"No database matches found for keyword '{keyword}'."

    return json.dumps(result, indent=2)


def initialize_korra() -> StateGraph:
    init_database()

    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)

    tools_list = [
        save_conversation_memory,
        get_conversation_history,
        query_student_records,
        search_course_catalog,
        general_database_search,
    ]
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools=tools_list)

    def database_agent(state: State) -> dict[str, list[BaseMessage]]:
        messages_with_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages_with_prompt)
        return {"messages": [response]}

    def route_tools(state: State) -> str:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            return "tools"
        return END

    graph_builder = StateGraph(State)
    graph_builder.add_node("database_agent", database_agent)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "database_agent")
    graph_builder.add_conditional_edges(
        "database_agent",
        route_tools,
        {
            "tools": "tools",
            END: END,
        },
    )
    graph_builder.add_edge("tools", "database_agent")

    return graph_builder.compile()


graph = initialize_korra()
log.info("Database Search Agent initialized successfully.")
log.info("Specialized tools: conversation memory, student queries, course catalog, general DB search.")


if __name__ == "__main__":
    print("Module 13 Section 1 Database Search Agent loaded successfully.")
    print(f"SQLite database path: {DB_PATH}")