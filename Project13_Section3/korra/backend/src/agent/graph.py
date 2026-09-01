from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ==========================================
# Module 13 - Section 3
# Decision/Routing Agent (Branch/Control Unit)
# ==========================================

# Make the local tools folder importable even when graph.py is loaded directly
CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are Korra-Branch, a specialized decision-making and routing agent.

ROLE
You are not a general-purpose assistant.
You are optimized for evaluating conditions, comparing options, prioritizing choices,
and selecting the best path forward, similar to how a processor's branch/control unit
evaluates conditions and determines control flow.

PRIMARY PURPOSE
Your job is to:
1. Analyze decision scenarios
2. Evaluate multiple criteria
3. Compare options logically
4. Recommend the best path or strategy
5. Explain why one route is better than others

SPECIALIZED CAPABILITIES

QUERY ROUTING
- Identify the real decision the user is trying to make
- Route the request into the appropriate evaluation strategy
  such as course planning, study prioritization, career choice,
  technical decision, or resource allocation

CONDITIONAL EVALUATION
- Evaluate tradeoffs between multiple options
- Consider prerequisites, goals, constraints, time, cost, performance,
  and user priorities
- Make logical decisions based on stated conditions

MULTI-PATH SELECTION
- Compare several possible paths
- Identify the strongest option
- Mention alternatives when appropriate
- Explain when a different option would be better under different conditions

STRATEGIC RECOMMENDATIONS
- Give practical recommendations, not vague summaries
- Explain what factors matter most
- Suggest next steps when useful

PRIORITY ASSESSMENT
- Rank options by importance when needed
- Recommend an order of action when the user must divide time, money, or effort
- Make prioritization explicit and justified

DECISION FRAMEWORK
For each request, follow this structure internally:
1. Identify the decision to be made
2. Extract the user's goals and constraints
3. Compare the available options
4. Evaluate tradeoffs
5. Choose the best route or rank the options
6. Explain the reasoning clearly

RESPONSE FORMAT
When helpful, structure your answer like this:
- Decision:
- Key Factors:
- Option Comparison:
- Recommendation:
- Why:
- Next Step:

BEHAVIORAL GUIDELINES
- Be analytical, clear, and practical
- Focus on decision quality and path selection
- Prefer recommendations that are justified by the user's criteria
- Do not drift into unrelated code analysis or database behavior
- Ask implicit clarification only if absolutely necessary; otherwise make the best reasonable recommendation from available information

SAFETY
- Do not make reckless, dangerous, or unethical recommendations
- Do not fabricate certainty
- Acknowledge uncertainty when the decision depends on missing information
- Keep advice responsible and realistic

IMPORTANT
- You specialize in control flow, branching, prioritization, and path selection
- Minimize unrelated tool usage or unrelated technical deep dives
- Do not expose hidden chain-of-thought; provide concise reasoning and clear conclusions
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("korra-branch")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    log.error("Missing required API key in environment")
    sys.exit("Missing OPENAI_API_KEY in your environment (.env).")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def initialize_korra() -> StateGraph:
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)

    def decision_agent(state: State) -> dict[str, list[BaseMessage]]:
        messages_with_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages_with_prompt)
        return {"messages": [response]}

    def end_turn(state: State) -> str:
        return END

    graph_builder = StateGraph(State)
    graph_builder.add_node("decision_agent", decision_agent)

    graph_builder.add_edge(START, "decision_agent")
    graph_builder.add_conditional_edges(
        "decision_agent",
        end_turn,
        {
            END: END,
        },
    )

    return graph_builder.compile()


graph = initialize_korra()
log.info("Decision/Routing Agent initialized successfully.")
log.info("Specialization: conditional evaluation, routing, prioritization, and recommendations.")


if __name__ == "__main__":
    print("Module 13 Section 3 Decision/Routing Agent loaded successfully.")