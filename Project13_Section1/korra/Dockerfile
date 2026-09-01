FROM python:3.11-slim

WORKDIR /app

# Install system build tools (gcc for compiling C tool)
RUN apt-get update && apt-get install -y \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*
	
# Copy project source code
COPY . .


# Compile the C tool for Linux
RUN gcc -Wall -Wextra -std=c11 -O2 -o src/agent/tools/file_stats src/agent/tools/file_stats.c


# Install Python dependencies
RUN pip install --upgrade pip && \
	pip install -e . && \
	pip install langgraph-cli langchain-openai langchain-tavily langchain-mcp-adapters "langgraph-cli[inmem]"


# Expose LangGraph dev server port
EXPOSE 2024

# Start LangGraph dev server
CMD ["langgraph", "dev", "--host", "0.0.0.0"]
