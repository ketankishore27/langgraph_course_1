from langgraph.graph import StateGraph, MessagesState
from state import customState
from langgraph.graph import START, END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv(override=True)
from nodes import reasoning_node, tool_node_handler, tool_node


def conditional_branching(state: customState):

    if state["messages"][-1].tool_calls:
        return "tool_invoke"

    return "end_process"


def create_graph():

    graph = StateGraph(customState)
    graph.add_node("reason", reasoning_node)
    graph.add_node("act", tool_node_handler)

    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason", conditional_branching, {"tool_invoke": "act", "end_process": END}
    )

    graph.add_edge("act", "reason")
    flow = graph.compile()
    flow.get_graph().draw_mermaid_png(
        output_file_path="react_agent/simple_React_Agent.png"
    )
    return flow


if __name__ == "__main__":
    flow = create_graph()
    result = flow.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the age of Donald Trump in 2025. use web search to find the current age. Triple it and return the result"
                )
            ]
        }
    )
