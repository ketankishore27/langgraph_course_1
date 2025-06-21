from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import MessageGraph, END, START
from typing import List
from tool_executor import executor_node
from dotenv import load_dotenv
from chains import chain_responder_initial, chain_revisor, chain_summary
load_dotenv(override=True)

builder = MessageGraph()
builder.add_node("first_responder", chain_responder_initial)
builder.add_node("revisor_node", chain_revisor)
builder.add_node("tool_node", executor_node)
builder.add_node("summary_node", chain_summary)

builder.add_edge(START, "first_responder")
builder.add_edge("first_responder", "tool_node")
builder.add_edge("tool_node", "revisor_node")

def should_summarize_or_continue(state: List[BaseMessage]) -> str:
    if sum([isinstance(state_msg, ToolMessage) for state_msg in state]) < 2:
        return "should_continue"

    return "should_summarize"

builder.add_conditional_edges("revisor_node", should_summarize_or_continue, 
                              {"should_continue": "tool_node", "should_summarize": END})

# builder.add_edge("summary_node", END)

flow = builder.compile()
flow.get_graph().draw_mermaid_png(output_file_path = "flow.png")

response = flow.invoke(
    "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital"
)

print(response)





