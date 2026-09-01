from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage

# Your langgraph.json points to supervisor_agent.py, so this test imports from there.
from supervisor_agent import graph


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


async def run_request(label: str, prompt: str) -> None:
    print(f"\n===== Starting Request {label} =====")
    print(prompt)

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=prompt)
            ]
        }
    )

    print(f"\n===== Finished Request {label} =====")
    final_message = result["messages"][-1]
    print(final_message.content)


async def main() -> None:
    request_a = "Look up student ID 12345 GPA in the database."
    request_b = "Look up student ID 67890 course history in the database."

    # Start both requests at the same time.
    # This bypasses LangGraph Studio's one-run-at-a-time queue and lets your
    # structural hazard detector see both requests competing for the same worker.
    await asyncio.gather(
        run_request("A", request_a),
        run_request("B", request_b),
    )


if __name__ == "__main__":
    asyncio.run(main())
