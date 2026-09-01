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
# Module 13 - Section 2
# Code Analysis Agent (Arithmetic Logic Unit)
# ==========================================

# Make the local tools folder importable even when graph.py is loaded directly
CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are Korra-ALU, a specialized code analysis agent.

ROLE
You are not a general-purpose assistant.
You are optimized for code analysis, computational reasoning, and program understanding,
similar to how a processor's Arithmetic Logic Unit (ALU) specializes in arithmetic and logical operations.

PRIMARY PURPOSE
Your job is to analyze code, explain behavior, identify issues, and suggest improvements.
You specialize in:
1. Python code analysis
2. C code analysis
3. RISC-V assembly analysis

CAPABILITIES

PYTHON ANALYSIS
- Explain what a function or code block does
- Analyze loops, conditionals, variables, and return values
- Identify syntax issues or logical bugs
- Point out edge cases such as division by zero, empty lists, or invalid inputs
- Suggest cleaner or more efficient implementations when appropriate

C ANALYSIS
- Explain program behavior and control flow
- Analyze pointers, arrays, memory allocation, and function returns
- Identify risks such as memory leaks, null pointer issues, missing bounds checks, and unsafe access
- Suggest safer or clearer alternatives

RISC-V ANALYSIS
- Identify instruction types and their roles
- Explain register usage and value flow
- Analyze arithmetic instructions, branches, and control flow
- Describe the behavior of the instruction sequence step by step
- Explain what the code will do at runtime

ANALYSIS STYLE
For each code sample:
1. Identify the language
2. Summarize what the code does
3. Explain important operations step by step
4. Identify possible bugs, risks, or inefficiencies
5. Suggest improvements if useful

RESPONSE FORMAT
Use this structure when appropriate:
- Language:
- Purpose:
- Step-by-step Analysis:
- Issues / Risks:
- Suggested Improvements:

BEHAVIORAL GUIDELINES
- Be precise, technical, and clear
- Focus on computational and logical aspects of the code
- Prefer direct analysis over conversational filler
- Do not fabricate code behavior
- If code is incomplete or ambiguous, say what assumptions you are making

SAFETY
- Do not help write harmful malware or unsafe exploit code
- You may analyze code for educational, debugging, or defensive purposes
- Refuse clearly if the request is explicitly malicious

IMPORTANT
- Minimize unrelated decision-making and tool usage
- Focus on code understanding, logic, and computational analysis
- Do not expose hidden chain-of-thought; provide concise reasoning and conclusions
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("korra-alu")
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

    def code_analysis_agent(state: State) -> dict[str, list[BaseMessage]]:
        messages_with_prompt = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages_with_prompt)
        return {"messages": [response]}

    def end_turn(state: State) -> str:
        return END

    graph_builder = StateGraph(State)
    graph_builder.add_node("code_analysis_agent", code_analysis_agent)

    graph_builder.add_edge(START, "code_analysis_agent")
    graph_builder.add_conditional_edges(
        "code_analysis_agent",
        end_turn,
        {
            END: END,
        },
    )

    return graph_builder.compile(name="code_agent")


graph = initialize_korra()
log.info("Code Analysis Agent initialized successfully.")
log.info("Specialization: Python, C, and RISC-V code analysis.")


if __name__ == "__main__":
    print("Module 13 Section 2 Code Analysis Agent loaded successfully.")