from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections import deque
from typing import Annotated, Deque

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
# Module 15 - Section 2
# Data Hazard Detection & Forwarding
# Structural Hazard Detection retained from Section 1
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
# Section 1: Structural Hazard State
# --------------------------------------------------
busy_workers: dict[str, str] = {}
worker_queues: dict[str, Deque[tuple[str, asyncio.Event]]] = {
    "database_search_agent": deque()
}
hazard_lock = asyncio.Lock()

# --------------------------------------------------
# Section 2: Data Hazard / Forwarding State
# --------------------------------------------------
# result_cache stores producer worker outputs by request id.
# Example:
# result_cache["req-abc123"]["database_search_agent"] = "student GPA is 3.7"
# --------------------------------------------------
result_cache: dict[str, dict[str, str]] = {}
cache_lock = asyncio.Lock()


class WorkerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _make_request_id(state: WorkerState) -> str:
    """
    Create a stable request id from the original user input.

    Section 1 used a random id for screenshots. Section 2 needs a stable id
    so Database Search and Decision/Routing can locate the same cache entry.
    """
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            digest = hashlib.sha1(_message_text(message).encode("utf-8")).hexdigest()[:8]
            return f"req-{digest}"

    joined = "|".join(_message_text(m) for m in state.get("messages", []))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    return f"req-{digest}"


async def acquire_worker(worker_name: str, request_id: str) -> None:
    """
    Section 1 pre-dispatch structural hazard detector.
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
    """
    Section 1 worker release and FIFO resume.
    """
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
            log.info(
                "PIPELINE: worker=%s freed by request=%s; no queued requests",
                worker_name,
                request_id,
            )


async def cache_worker_output(request_id: str, worker_name: str, output_text: str) -> None:
    """
    Store a producer worker result for later forwarding.
    """
    async with cache_lock:
        result_cache.setdefault(request_id, {})
        result_cache[request_id][worker_name] = output_text
        log.info(
            "DATA CACHE: request=%s stored output from producer=%s",
            request_id,
            worker_name,
        )


async def get_cached_output(request_id: str, producer_name: str) -> str | None:
    """
    Look for an upstream producer result.
    """
    async with cache_lock:
        return result_cache.get(request_id, {}).get(producer_name)


def build_structural_database_worker():
    """
    Wrap Database Search with:
    1. structural hazard detection and queueing
    2. result caching for downstream forwarding
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

            # Demo delay retained from Section 1 for structural hazard screenshots.
            await asyncio.sleep(3)

            result = await database_search_agent.ainvoke(state)
            output_text = _message_text(result["messages"][-1])

            await cache_worker_output(request_id, worker_name, output_text)

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


def build_forwarding_decision_worker():
    """
    Wrap Decision/Routing with data hazard forwarding.

    If Database Search already produced output for this request, feed that output
    directly to Decision/Routing as context and log [FORWARDING].
    """

    async def forwarding_decision_node(state: WorkerState) -> dict[str, list[BaseMessage]]:
        consumer_name = "decision_agent"
        producer_name = "database_search_agent"
        request_id = _make_request_id(state)

        forwarded_output = await get_cached_output(request_id, producer_name)

        if forwarded_output is not None:
            hazard_logger.forwarding(
                request_id=request_id,
                producer_name=producer_name,
                consumer_name=consumer_name,
            )

            forwarding_context = SystemMessage(
                content=(
                    "FORWARDED UPSTREAM RESULT FROM Database Search Agent. "
                    "Use this result directly when making your recommendation.\\n\\n"
                    f"{forwarded_output}"
                )
            )

            forwarded_state: WorkerState = {
                "messages": [forwarding_context] + state["messages"]
            }

            log.info(
                "DATA FORWARDING: request=%s producer=%s consumer=%s cached_result=used",
                request_id,
                producer_name,
                consumer_name,
            )

            result = await decision_routing_agent.ainvoke(forwarded_state)
            return {"messages": result["messages"]}

        log.info(
            "DATA FORWARDING: request=%s no cached producer output found for consumer=%s",
            request_id,
            consumer_name,
        )

        result = await decision_routing_agent.ainvoke(state)
        return {"messages": result["messages"]}

    def end_turn(state: WorkerState) -> str:
        return END

    builder = StateGraph(WorkerState)
    builder.add_node("decision_agent", forwarding_decision_node)
    builder.add_edge(START, "decision_agent")
    builder.add_conditional_edges("decision_agent", end_turn, {END: END})

    return builder.compile(name="decision_agent")


structural_database_search_agent = build_structural_database_worker()
forwarding_decision_routing_agent = build_forwarding_decision_worker()

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

DATA HAZARD / FORWARDING RULE
- For a request that first needs database facts and then needs a recommendation,
  invoke Database Search Agent first and Decision/Routing Agent second.
- Do not call the supervisor again between those workers.
- The Database Search output is forwarded directly to Decision/Routing through
  the forwarding wrapper.

PIPELINE COORDINATION RULES
When a request requires multiple workers:
1. Invoke workers in sequence.
2. Preferred ordering:
   - Database Search -> Code Analysis -> Decision/Routing
3. Output from earlier workers must inform later workers.
4. Treat previous worker outputs as context for later workers.
5. After all needed workers respond, synthesize them into one coherent final answer.

DATA FORWARDING TEST CASE
If the user asks:
"Look up student 12345's GPA and tell me if they should take COMPE 475"
Use:
- Database Search Agent
- Decision/Routing Agent

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
        forwarding_decision_routing_agent,
    ],
    model=model,
    tools=[forward_message_tool],
    prompt=SUPERVISOR_PROMPT,
    supervisor_name="supervisor",
)

graph = supervisor.compile(name="module15_section2_data_hazard_forwarding")

log.info("Module 15 Section 2 Supervisor initialized successfully.")
log.info("Structural hazard detection retained for Database Search Agent.")
log.info("Data hazard forwarding enabled: database_search_agent -> decision_agent.")
log.info("Required log: [FORWARDING].")


if __name__ == "__main__":
    print("Module 15 Section 2 Supervisor loaded successfully.")
    print("Data forwarding enabled from Database Search Agent to Decision/Routing Agent.")
