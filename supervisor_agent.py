from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import deque
from typing import Annotated, Any, Deque

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from database_agent.graph import graph as database_search_agent
from code_agent.graph import graph as code_analysis_agent
from decision_agent.graph import graph as decision_routing_agent

import hazard_logger

# ==========================================
# Module 15 - Section 4
# Integrated Hazard Detection System
# Structural + Forwarding + Data Stall + Control Hazard
# ==========================================

DEFAULT_MODEL = "gpt-4o-mini"

BUBBLE_TICK_SECONDS = 0.5
SLOW_DATABASE_DELAY_SECONDS = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("korra-supervisor")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in your environment (.env).")

model = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)


class WorkerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


producer_cache: dict[str, dict[str, Any]] = {}
producer_lock = asyncio.Lock()

busy_workers: dict[str, str] = {}
worker_queues: dict[str, Deque[tuple[str, asyncio.Event]]] = {
    "database_search_agent": deque(),
    "code_analysis_agent": deque(),
    "decision_routing_agent": deque(),
}

hazard_lock = asyncio.Lock()

queued_downstream_workers: dict[str, list[str]] = {}


def _make_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def _latest_user_text(state: WorkerState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)

    if state.get("messages"):
        return str(state["messages"][-1].content)

    return ""


def _message_text(messages: list[BaseMessage]) -> str:
    return "\n".join(
        str(message.content)
        for message in messages
        if getattr(message, "content", None)
    )


def _looks_like_no_record(text: str) -> bool:
    lowered = text.lower()
    return (
        "no record" in lowered
        or "not found" in lowered
        or "no student found" in lowered
        or "could not find" in lowered
        or "does not exist" in lowered
        or "unknown student" in lowered
    )


async def acquire_worker(worker_name: str, request_id: str) -> None:
    event: asyncio.Event | None = None

    async with hazard_lock:
        worker_queues.setdefault(worker_name, deque())

        if worker_name in busy_workers:
            hazard_logger.structural_hazard(
                request_id=request_id,
                worker_name=worker_name,
                current_request=busy_workers.get(worker_name),
                queue_depth=len(worker_queues[worker_name]),
            )

            event = asyncio.Event()
            worker_queues[worker_name].append((request_id, event))

        else:
            busy_workers[worker_name] = request_id
            log.info("PIPELINE: request=%s acquired worker=%s", request_id, worker_name)
            return

    if event is not None:
        await event.wait()

        hazard_logger.resume(request_id, worker_name)

        async with hazard_lock:
            busy_workers[worker_name] = request_id
            log.info("PIPELINE: request=%s resumed worker=%s", request_id, worker_name)


async def release_worker(worker_name: str, request_id: str) -> None:
    async with hazard_lock:
        current_owner = busy_workers.get(worker_name)

        if current_owner == request_id:
            busy_workers.pop(worker_name, None)

        if worker_queues.get(worker_name):
            next_request_id, next_event = worker_queues[worker_name].popleft()
            log.info(
                "PIPELINE: worker=%s released by request=%s; resuming request=%s",
                worker_name,
                request_id,
                next_request_id,
            )
            next_event.set()
        else:
            log.info(
                "PIPELINE: worker=%s released by request=%s; no queued requests",
                worker_name,
                request_id,
            )


async def _cache_write(
    request_id: str,
    producer_name: str,
    messages: list[BaseMessage],
) -> None:
    async with producer_lock:
        producer_cache[request_id] = {
            "producer": producer_name,
            "ready": True,
            "messages": messages,
        }


async def _cache_read_if_ready(request_id: str) -> dict[str, Any] | None:
    async with producer_lock:
        cached = producer_cache.get(request_id)

        if cached and cached.get("ready"):
            return cached

    return None


async def wait_for_producer_ready(
    request_id: str,
    dependent_worker: str,
    producer_worker: str,
) -> dict[str, Any]:
    cached = await _cache_read_if_ready(request_id)

    if cached is not None:
        hazard_logger.forwarding(
            request_id=request_id,
            producer_name=producer_worker,
            consumer_name=dependent_worker,
        )
        return cached

    hazard_logger.stall(
        request_id=request_id,
        worker_name=dependent_worker,
        waiting_on=producer_worker,
    )

    bubble_count = 0

    while True:
        bubble_count += 1

        hazard_logger.bubble_cycle(
            request_id=request_id,
            worker_name=dependent_worker,
            cycle=bubble_count,
        )

        await asyncio.sleep(BUBBLE_TICK_SECONDS)

        cached = await _cache_read_if_ready(request_id)

        if cached is not None:
            hazard_logger.resume(request_id, dependent_worker)
            return cached


async def run_database_producer(
    state: WorkerState,
    request_id: str,
    slow: bool = True,
) -> None:
    worker_name = "database_search_agent"

    await acquire_worker(worker_name, request_id)

    try:
        log.info("PIPELINE: request=%s started producer=%s", request_id, worker_name)

        if slow:
            await asyncio.sleep(SLOW_DATABASE_DELAY_SECONDS)

        result = await database_search_agent.ainvoke(state)
        messages = result.get("messages", [])

        await _cache_write(request_id, worker_name, messages)

        log.info(
            "PIPELINE: request=%s producer=%s wrote ready cache",
            request_id,
            worker_name,
        )

    finally:
        await release_worker(worker_name, request_id)


async def run_code_analysis_after_database(
    request_id: str,
    original_user_text: str,
    database_context: str,
) -> list[BaseMessage]:
    worker_name = "code_analysis_agent"

    await acquire_worker(worker_name, request_id)

    try:
        code_prompt = HumanMessage(
            content=(
                "Use the Database Search result below as required context. "
                "Do not skip it and do not use stale assumptions.\n\n"
                f"Original user request:\n{original_user_text}\n\n"
                f"Database Search output now ready:\n{database_context}\n\n"
                "Now perform the requested Code Analysis using that database output."
            )
        )

        result = await code_analysis_agent.ainvoke({"messages": [code_prompt]})
        return result.get("messages", [])

    finally:
        await release_worker(worker_name, request_id)


async def run_decision_after_database(
    request_id: str,
    original_user_text: str,
    database_context: str,
) -> list[BaseMessage]:
    worker_name = "decision_routing_agent"

    await acquire_worker(worker_name, request_id)

    try:
        decision_prompt = HumanMessage(
            content=(
                "Use this Database Search output directly as forwarded input.\n\n"
                f"Original user request:\n{original_user_text}\n\n"
                f"Forwarded Database Search output:\n{database_context}\n\n"
                "Now make the requested decision or recommendation."
            )
        )

        result = await decision_routing_agent.ainvoke({"messages": [decision_prompt]})
        return result.get("messages", [])

    finally:
        await release_worker(worker_name, request_id)


async def integrated_hazard_pipeline_node(
    state: WorkerState,
) -> dict[str, list[BaseMessage]]:
    request_id = _make_request_id()
    user_text = _latest_user_text(state)

    producer_worker = "database_search_agent"
    code_worker = "code_analysis_agent"
    decision_worker = "decision_routing_agent"

    async with producer_lock:
        producer_cache.pop(request_id, None)

    queued_downstream_workers[request_id] = [
        code_worker,
        decision_worker,
    ]

    log.info(
        "PIPELINE: request=%s tentatively queued downstream workers=%s",
        request_id,
        queued_downstream_workers[request_id],
    )

    database_task = asyncio.create_task(
        run_database_producer(
            state=state,
            request_id=request_id,
            slow=True,
        )
    )

    producer_output = await wait_for_producer_ready(
        request_id=request_id,
        dependent_worker=code_worker,
        producer_worker=producer_worker,
    )

    await database_task

    database_context = _message_text(producer_output["messages"])

    if _looks_like_no_record(database_context):
        hazard_logger.control_hazard(
            request_id=request_id,
            reason="database returned no record found",
        )

        flushed_workers = queued_downstream_workers.get(
            request_id,
            [code_worker, decision_worker],
        )

        hazard_logger.flush(
            request_id=request_id,
            workers=flushed_workers,
        )

        queued_downstream_workers[request_id] = []

        hazard_logger.resume(
            request_id=request_id,
            worker_name="supervisor_reroute_help_path",
        )

        return {
            "messages": [
                AIMessage(
                    content=(
                        "No record was found for the requested student.\n\n"
                        "The system detected a control hazard because the Database "
                        "Search result invalidated the tentatively queued Code Analysis "
                        "and Decision/Routing workers. Those downstream workers were "
                        "flushed, and execution was rerouted to this help response.\n\n"
                        "Please verify the student ID and try again."
                    )
                )
            ]
        }

    code_messages = await run_code_analysis_after_database(
        request_id=request_id,
        original_user_text=user_text,
        database_context=database_context,
    )

    code_context = _message_text(code_messages)

    hazard_logger.forwarding(
        request_id=request_id,
        producer_name=code_worker,
        consumer_name=decision_worker,
    )

    decision_messages = await run_decision_after_database(
        request_id=request_id,
        original_user_text=user_text,
        database_context=database_context + "\n\n" + code_context,
    )

    decision_context = _message_text(decision_messages)

    queued_downstream_workers[request_id] = []

    final_text = (
        "Integrated hazard pipeline completed successfully.\n\n"
        "Database Search output:\n"
        f"{database_context}\n\n"
        "Code Analysis output:\n"
        f"{code_context}\n\n"
        "Decision/Routing output:\n"
        f"{decision_context}"
    )

    return {"messages": [AIMessage(content=final_text)]}


def route_after_pipeline(state: WorkerState) -> str:
    return END


builder = StateGraph(WorkerState)

builder.add_node("integrated_hazard_pipeline", integrated_hazard_pipeline_node)
builder.add_edge(START, "integrated_hazard_pipeline")
builder.add_conditional_edges(
    "integrated_hazard_pipeline",
    route_after_pipeline,
    {END: END},
)

graph = builder.compile(name="module15_section4_integrated_hazards")

log.info("Module 15 Section 4 Integrated Supervisor initialized successfully.")
log.info("Enabled hazards: structural, forwarding, data stall, control flush/reroute.")
log.info(
    "Required logs: [STRUCTURAL HAZARD], [FORWARDING], [STALL], "
    "[BUBBLE CYCLE N], [CONTROL HAZARD], [FLUSH], [RESUME]."
)


if __name__ == "__main__":
    print("Module 15 Section 4 Integrated Supervisor loaded successfully.")
    print("Hazards enabled:")
    print("- Structural Hazard")
    print("- Data Forwarding")
    print("- Data Hazard Stalling")
    print("- Control Hazard Flush/Reroute")