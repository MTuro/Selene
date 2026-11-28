"""Measure warm Perception latency on an already running local Ollama instance.

Set PYTHONPATH to the revision under test and use an isolated ISAAC_HOME.
The first of six measurements for each prompt is discarded. Results are milliseconds.
"""

from statistics import median
from time import perf_counter

from langchain_core.messages import HumanMessage

from isaac.core.state import make_initial_state
from isaac.nodes.perception import perception_node

for request in [
    "can you hear me?",
    "what is your name?",
    "explain what Python is.",
    "explain neural networks in two sentences.",
]:
    times = []
    for _ in range(6):
        state = make_initial_state()
        state["voice_session"] = True
        state["messages"] = [HumanMessage(content="Selene, " + request)]
        t = perf_counter()
        result = perception_node(state)
        times.append((perf_counter() - t) * 1000)
    print(request, result["task_mode"], round(median(times[1:]), 3))
