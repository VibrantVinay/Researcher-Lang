from langgraph.graph import StateGraph, START, END

# Define the routing logic for the Critic
def route_research(state: ResearchState):
    # CIRCUIT BREAKER: Never let an LLM spend your money forever
    if state["critique_count"] >= state["max_revisions"]:
        return "synthesis"
    
    if state["is_satisfactory"]:
        return "synthesis"
    else:
        return "scraper" # Loop backward!

# Build Graph
workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("scraper", scraper_node)
workflow.add_node("critic", critic_node)
workflow.add_node("synthesis", synthesis_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "scraper")
workflow.add_edge("scraper", "critic")

# The Fork in the Road
workflow.add_conditional_edges(
    "critic",
    route_research,
    {
        "scraper": "scraper",
        "synthesis": "synthesis"
    }
)

workflow.add_edge("synthesis", END)

app = workflow.compile()
