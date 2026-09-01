
# ============================================================
# INTERNAL HELPERS
# ============================================================

def _emit(tag: str, message: str = "") -> None:
    """
    Write a tagged log line to stdout with an immediate flush.

    The line begins directly with the bracketed tag so graders can scan for
    exact tag strings without stripping a timestamp prefix. Flushing on
    every line is required so streaming screenshots and live demo
    recordings show partial state mid-execution rather than buffering the
    entire trace until the program exits.

    Args:
        tag: The bracketed log tag, e.g. "[STRUCTURAL HAZARD]"
        message: Optional context appended after the tag

    Returns:
        None
    """
    line = f"{tag} {message}".rstrip()
    print(line, flush=True)


# ============================================================
# HAZARD LOG TAG FUNCTIONS
# ============================================================

def structural_hazard(detail: str) -> None:
    """
    Emit a [STRUCTURAL HAZARD] event.

    Use when two requests want the same worker at the same cycle and the
    second must be queued.

    Args:
        detail: Human-readable context about which request is queued and why

    Returns:
        None
    """
    _emit("[STRUCTURAL HAZARD]", detail)


def forwarding(detail: str) -> None:
    """
    Emit a [FORWARDING] event.

    Use when a completed worker output is passed directly to a dependent
    worker without a supervisor round-trip.

    Args:
        detail: Human-readable context naming the producer and consumer workers

    Returns:
        None
    """
    _emit("[FORWARDING]", detail)


def stall(detail: str) -> None:
    """
    Emit a [STALL] event.

    Use when a dependent worker cannot proceed because its producer is not
    done. Follow this with one bubble_cycle() call per pause tick, then a
    resume() call when the dependency clears.

    Args:
        detail: Human-readable context naming the waiting worker and its dependency

    Returns:
        None
    """
    _emit("[STALL]", detail)


def bubble_cycle(n: int) -> None:
    """
    Emit a [BUBBLE CYCLE N] event.

    Call once per stall tick so the bubble is visible in the log. A
    time.sleep alone is not sufficient – the rubric requires these lines.

    Args:
        n: Cycle number (1, 2, 3, ...) emitted on each tick of the stall

    Returns:
        None
    """
    _emit(f"[BUBBLE CYCLE {n}]")


def control_hazard(detail: str) -> None:
    """
    Emit a [CONTROL HAZARD] event.

    Use when a worker output invalidates already-queued downstream work.
    Follow this with a flush() call and then a resume() call describing the
    rerouted path.

    Args:
        detail: Human-readable context naming the invalidating result and request

    Returns:
        None
    """
    _emit("[CONTROL HAZARD]", detail)


def flush(workers_discarded: list[str]) -> None:
    """
    Emit a [FLUSH] event listing which queued workers were discarded.

    Args:
        workers_discarded: List of worker names that were flushed from the queue

    Returns:
        None
    """
    _emit("[FLUSH]", f"Discarding queued: {', '.join(workers_discarded)}")


def resume(detail: str) -> None:
    """
    Emit a [RESUME] event.

    Use after a stall releases, a queue slot frees up, or a flush completes
    and rerouting begins.

    Args:
        detail: Human-readable context describing what is resuming

    Returns:
        None
    """
    _emit("[RESUME]", detail)


# ============================================================
# ENTRY POINT
# ============================================================

# Run this file directly to produce a sample trace matching the example log
# in the project specification. Use this to verify your environment captures
# and displays the tags correctly before integrating into graph.py.
if __name__ == "__main__":
    structural_hazard("Request B queued -- Database Search Agent in use by Request A")
    resume("Request B starting -- Database Search Agent free")
    forwarding("Database Search output -> Decision/Routing (Request B), supervisor bypassed")
    stall("Decision/Routing waiting on Database Search (Request C)")
    bubble_cycle(1)
    bubble_cycle(2)
    resume("Decision/Routing input ready (Request C)")
    control_hazard("Database Search returned 'no record found' (Request D)")
    flush(["Code Analysis", "Decision/Routing"])
    resume("Rerouting Request D to error/help response path")
