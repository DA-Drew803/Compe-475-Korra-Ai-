from __future__ import annotations

import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ==============================
# Module 10 - Section 3
# Korra Full Stack Containerized
# ==============================

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR / "tools"
DATA_DIR = Path(os.getenv("KORRA_DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "korra_memory.db"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from file_stats_tool import analyze_file_statistics

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "2"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("korra")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

load_dotenv()


@tool
def save_note_to_database(note: str, category: str = "general") -> str:
    """Save a short note to the local SQLite database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                note TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO notes(category, note) VALUES (?, ?)",
            (category, note),
        )
        conn.commit()

    return f"Saved note to database under category '{category}'."


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def initialize_korra():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not openai_api_key or not tavily_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or TAVILY_API_KEY in environment")

    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.7)
    tavily_tool = TavilySearch(max_results=MAX_SEARCH_RESULTS, include_answer=True)

    tools_list = [tavily_tool, analyze_file_statistics, save_note_to_database]
    llm_with_tools = llm.bind_tools(tools_list)

    tavily_node = ToolNode(tools=[tavily_tool])
    file_stats_node = ToolNode(tools=[analyze_file_statistics])
    database_node = ToolNode(tools=[save_note_to_database])

    def korra(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tavily_tool_node(state: State):
        return tavily_node.invoke(state)

    def file_stats_tool_node(state: State):
        return file_stats_node.invoke(state)

    def database_tool_node(state: State):
        return database_node.invoke(state)

    def route_tools(state: State):
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return END

        tool_name = tool_calls[0]["name"]
        if tool_name == tavily_tool.name:
            return "tavily_tool"
        if tool_name == analyze_file_statistics.name:
            return "file_stats_tool"
        if tool_name == save_note_to_database.name:
            return "database_tool"
        return END

    graph_builder = StateGraph(State)
    graph_builder.add_node("korra", korra)
    graph_builder.add_node("tavily_tool", tavily_tool_node)
    graph_builder.add_node("file_stats_tool", file_stats_tool_node)
    graph_builder.add_node("database_tool", database_tool_node)

    graph_builder.add_edge(START, "korra")
    graph_builder.add_conditional_edges(
        "korra",
        route_tools,
        {
            "tavily_tool": "tavily_tool",
            "file_stats_tool": "file_stats_tool",
            "database_tool": "database_tool",
            END: END,
        },
    )
    graph_builder.add_edge("tavily_tool", "korra")
    graph_builder.add_edge("file_stats_tool", "korra")
    graph_builder.add_edge("database_tool", "korra")

    return graph_builder.compile()


graph = initialize_korra()
