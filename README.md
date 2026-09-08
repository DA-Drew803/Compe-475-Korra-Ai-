# Korra AI

Korra is an AI assistant developed for **COMPE 475** using Python, LangChain, and LangGraph. The project demonstrates how a language model can reason about a user request, call an external tool when needed, process the tool’s result, and return a final response.

## Workflow

Korra follows this graph structure:

```text
User Input → LLM Node → Tool Node → LLM Node → Final Response
```

The graph uses conditional routing:

1. The user sends a message.
2. The LLM determines whether a tool is required.
3. If needed, the request is routed to the tool node.
4. The tool executes and returns its result.
5. The result is passed back to the LLM.
6. The LLM creates the final response.
7. The graph routes to `END`.

## Features

* Conversational AI interface
* Tool-calling support
* Conditional graph routing
* State-based message management
* Multi-step LLM and tool interaction
* LangGraph Studio compatibility
* Environment-variable support for API keys

## Technologies

* Python
* LangChain
* LangGraph
* Large Language Model API
* LangGraph Studio


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DA-Drew803/Compe-475-Korra-Ai-.git
cd Compe-475-Korra-Ai-
```

### 2. Create a Conda environment

```bash
conda create --name korra python=3.11
conda activate korra
```

### 3. Install the dependencies

```bash
pip install -e .
```

### 4. Configure environment variables

Create a `.env` file in the project directory and add the API key required by the model:

```env
OPENAI_API_KEY=your_api_key_here
```


## Running the Project

Start the LangGraph development server:

```bash
langgraph dev
```

After the server starts, open the provided LangGraph Studio URL in your browser. Enter a message to test Korra’s LLM reasoning, tool execution, and final response.

## Example Interaction

```text
User: Ask a question that requires one of Korra's tools.

Korra:
1. Receives the question
2. Determines that a tool is needed
3. Calls the appropriate tool
4. Processes the tool result
5. Returns a final response
```

## Graph Architecture

```text
START
  │
  ▼
LLM Node
  │
  ├── Tool required ──► Tool Node ──► LLM Node
  │                                      │
  └── No tool required ──────────────────┤
                                         ▼
                                        END
```

## Purpose

The purpose of this project is to demonstrate the design of an agentic AI workflow. Unlike a basic chatbot that produces only one response, Korra can decide when to use a tool, execute that tool, analyze its output, and continue processing before responding to the user.


