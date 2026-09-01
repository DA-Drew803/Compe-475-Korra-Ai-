from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import deque
from typing import Annotated, Deque

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool
from typing_extensions import TypedDict

from database_agent.graph import graph as database_search_agent
from code_agent.graph import graph as code_analysis_agent
from decision_agent.graph import graph as decision_routing_agent

import hazard_logger

# ==========================================
# Module 15 - Section 1
# Structural Hazard Detection & Resolution
# Full Supervisor Architecture
# ==========================================

DEFAULT_MODEL = "gpt-4o-mini"

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

# --------------------------------------------------
# Structural Hazard State
# --------------------------------------------------
busy_workers: dict[str, str] = {}
worker_queues: dict[str, Deque[tuple[str, asyncio.Event]]] = {
    "database_search_agent": deque()
}
hazard_lock = asyncio.Lock()


class WorkerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _make_request_id(state: WorkerState) -> str:
    """Create a short local request id for clear hazard logs."""
    return f"req-{uuid.uuid4().hex[:8]}"


async def acquire_worker(worker_name: str, request_id: str) -> None:
    """
    Pre-dispatch structural hazard detector.

    If the worker is free, mark it busy.
    If the worker is busy, queue this request and wait until resumed.
    """
    event: asyncio.Event | None = None

    async with hazard_lock:
        worker_queues.setdefault(worker_name, deque())

        if worker_name in busy_workers:
            hazard_logger.structural_hazard(request_id, worker_name)
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
            log.info("PIPELINE: request=%s resumed and acquired worker=%s", request_id, worker_name)


async def release_worker(worker_name: str, request_id: str) -> None:
    """Release a worker and resume the next queued request, if one exists."""
    async with hazard_lock:
        current_owner = busy_workers.get(worker_name)

        if current_owner == request_id:
            busy_workers.pop(worker_name, None)

        if worker_queues.get(worker_name):
            next_request_id, next_event = worker_queues[worker_name].popleft()
            log.info(
                "PIPELINE: worker=%s freed by request=%s; resuming queued request=%s",
                worker_name,
                request_id,
                next_request_id,
            )
            next_event.set()
        else:
            log.info("PIPELINE: worker=%s freed by request=%s; no queued requests", worker_name, request_id)


def build_structural_database_worker():
    """
    Wrap the existing Database Search Agent with explicit structural hazard
    detection and FIFO queueing.
    """

    async def structural_database_node(state: WorkerState) -> dict[str, list[BaseMessage]]:
        worker_name = "database_search_agent"
        request_id = _make_request_id(state)

        log.info(
            "PIPELINE: request=%s reached pre-dispatch check for worker=%s",
            request_id,
            worker_name,
        )

        await acquire_worker(worker_name, request_id)

        try:
            log.info("PIPELINE: request=%s running worker=%s", request_id, worker_name)

            # Demo delay: keeps Request A on the worker long enough for Request B
            # to hit the busy-worker check and produce the required screenshot logs.
            await asyncio.sleep(3)

            result = await database_search_agent.ainvoke(state)

            log.info("PIPELINE: request=%s completed worker=%s", request_id, worker_name)

            return {"messages": result["messages"]}

        finally:
            await release_worker(worker_name, request_id)

    def end_turn(state: WorkerState) -> str:
        return END

    builder = StateGraph(WorkerState)
    builder.add_node("database_search_agent", structural_database_node)
    builder.add_edge(START, "database_search_agent")
    builder.add_conditional_edges("database_search_agent", end_turn, {END: END})

    return builder.compile(name="database_search_agent")


structural_database_search_agent = build_structural_database_worker()

SUPERVISOR_PROMPT = """
You are Korra-Supervisor, a control-unit style pipeline coordinator for specialized worker agents.

ROLE
You coordinate specialized workers the way a processor control unit coordinates
multiple functional units in a pipeline. Your job is to read the user's request,
select the correct workers, invoke them in the proper order, preserve relevant
information from earlier workers, and produce the final response.

AVAILABLE WORKERS

1. Database Search Agent
Use for:
- student records
- GPA lookups
- grades
- course history
- class averages
- course catalog data
- skill level lookups
- student IDs
- student performance queries
- midterm and assignment score lookups
- any stored academic data retrieval

2. Code Analysis Agent
Use for:
- Python code analysis
- C code analysis
- RISC-V assembly analysis
- explaining code behavior
- identifying concepts covered by a code sample
- determining code complexity
- identifying prerequisite skills needed for code

3. Decision/Routing Agent
Use for:
- recommendations
- choosing between alternatives
- deciding readiness
- assigning difficulty levels
- course planning
- study group formation strategy
- deciding whether to simplify or keep an assignment
- final recommendation based on prior worker outputs

CORE ROUTING RULES
- Do not answer specialized requests directly when a worker should handle them.
- For single-domain requests, use only the best matching worker.
- For multi-domain requests, use multiple workers in sequence.
- Choose the minimum number of workers necessary.
- Do not call workers in parallel within a single request.
- Maintain clear control flow and correct ordering.

PIPELINE COORDINATION RULES
When a request requires multiple workers:
1. Invoke workers in sequence.
2. Preferred ordering:
   - Database Search -> Code Analysis -> Decision/Routing
3. Output from earlier workers must inform later workers.
4. Treat previous worker outputs as context for later workers.
5. After all needed workers respond, synthesize them into one coherent final answer.

DATA FORWARDING RULES
- When Database Search runs first, its findings must influence later interpretation.
- When Code Analysis runs before Decision/Routing, the code findings must influence the recommendation.
- The final answer must clearly reflect information from all workers used.
- Do not ignore prior worker outputs.

SYNTHESIS RULES
- Synthesize only when multiple workers were used.
- Do not simply paste worker responses together.
- Combine results into a concise, coherent answer.
- Explicitly connect database facts, code complexity, and final recommendation.
- Make the final result read like one unified answer, not three separate answers.

PASSTHROUGH RULE
- If exactly one worker is sufficient, use forward_message.
- Do not perform unnecessary synthesis for a single-worker request.

STRUCTURAL HAZARD RULE
- If multiple requests require the same worker, treat that as a shared-resource conflict.
- Requests must not interfere with each other.
- Keep state isolated per request.
- Shared workers may process one request at a time if necessary.
- If the Database Search Agent is already busy, queue the later request and resume it after the first request completes.

FINAL RESPONSE STYLE
- For single-worker tasks: use forward_message passthrough.
- For multi-worker tasks: produce one synthesized final answer.
- Be clear, concise, and practical.
- Explicitly ground the recommendation in the worker outputs.
"""

forward_message_tool = create_forward_message_tool("supervisor")

supervisor = create_supervisor(
    [
        structural_database_search_agent,
        code_analysis_agent,
        decision_routing_agent,
    ],
    model=model,
    tools=[forward_message_tool],
    prompt=SUPERVISOR_PROMPT,
    supervisor_name="supervisor",
)

graph = supervisor.compile(name="module15_section1_structural_hazard")

log.info("Module 15 Section 1 Supervisor initialized successfully.")
log.info("Integrated workers: database_search_agent, code_analysis_agent, decision_routing_agent")
log.info("Structural hazard detection enabled for Database Search Agent.")
log.info("FIFO queueing enabled for Database Search Agent.")
log.info("Required logs: [STRUCTURAL HAZARD] and [RESUME].")


if __name__ == "__main__":
    print("Module 15 Section 1 Supervisor loaded successfully.")
    print("Structural hazard detection enabled.")
    print("Database Search Agent uses busy-worker tracking and FIFO queueing.")
