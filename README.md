An  AI chatbot architecture built with **LangGraph** that simulates hardware pipeline hazards (Structural, Data, and Control) inside an LLM multi-agent system. The system coordinates tasks using a central Supervisor agent, three specialized Worker agents, and a dedicated Hazard Layer.

---

## 🛠️ Architecture Overview

The system mirrors a classical CPU pipeline architecture translated into LLM agent interactions. The graph topology can be visualized and debugged inside **LangGraph Studio**.

* **Central Supervisor:** Orchestrates incoming user requests and schedules worker tasks.
* **Worker Agents (x3):** Dedicated nodes handling specific domain tasks (e.g., Database Search, Core Chat, Reasoning).
* **Hazard Layer:** A pipeline monitoring layer residing in `graph.py` that intercepts agent transitions to inspect, log, and resolve scheduling conflicts before execution.

---

## Live Demo & Simulation Walkthrough

### 1. Structural Hazard Demo
A structural hazard occurs when two parallel operations compete for a single, non-shareable hardware resource. 
* **Scenario:** Triggering two simultaneous `Database Search` requests.
* **Behavior:** The Hazard Layer detects the resource collision, enforces a queueing decision, and executes them sequentially.
* **Expected Log Output:**
  ```text
  [STRUCTURAL HAZARD] Resource conflict detected on Database_Search_Node.
  [INFO] Queueing second request; delaying execution cycle.
  ```

### 2. Data Hazard Demo
A data hazard happens when an instruction depends on the result of a previous instruction that has not yet completed. This simulation demonstrates two methods of handling data dependencies:

* **Forwarding Case (Optimal):** The result is passed directly between workers as soon as the data is ready, minimizing delay.
* **Stalling Case (Bubble Cycle):** If a worker requests data that is not yet ready, the pipeline must wait.
* **Expected Log Output:**
  ```text
  [STALL] Data dependency unresolved for Worker_3.
  [BUBBLE CYCLE 1] Waiting for upstream agent payload...
  [BUBBLE CYCLE 2] Waiting...
  [RESUME] Data payload received. Forwarding to execution.
  ```

### 3. Control Hazard Demo
A control hazard arises from the pipeline making a decision based on a conditional branch before the outcome is definitively known (akin to branch misprediction in a CPU pipeline).
* **Scenario:** The chatbot runs a canonical "no record found" scenario. The Supervisor predicts a data match and pre-fetches subsequent steps, but the worker returns a null value.
* **Behavior:** The system catches the mismatch, flushes the incorrectly speculative steps, and restarts execution along the correct path.
* **Expected Log Output:**
  ```text
  [CONTROL HAZARD] Branch misprediction: expected record payload, received NULL.
  [FLUSH] Purging speculatively queued worker tasks from state.
  [RESUME] Fetching fallback error handling routine.
  ```

---
