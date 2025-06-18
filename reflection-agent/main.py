import os

work_dir = os.path.join(os.getcwd(), "reflection-agent")
os.chdir(work_dir)
print(f"Current working: {work_dir}")

from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from typing import Sequence, List
from langgraph.graph import MessageGraph, END, START
from chains import generation_chain, reflection_chain


def generation_chain_processing(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    return generation_chain.invoke({"messages": state})


def reflection_chain_processing(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    response = reflection_chain.invoke({"messages": state})
    return [HumanMessage(content=response.content)]


def should_reflect_or_end(state: Sequence[BaseMessage]) -> str:
    if len(state) > 6:
        return "should end"

    return "should reflect"


def create_graph():
    flow = MessageGraph()
    flow.add_node("generation_node", generation_chain_processing)
    flow.add_node("reflection_node", reflection_chain_processing)
    flow.add_edge(START, "generation_node")
    flow.add_conditional_edges(
        "generation_node",
        should_reflect_or_end,
        {"should end": END, "should reflect": "reflection_node"},
    )

    flow.add_edge("reflection_node", "generation_node")
    builder = flow.compile()
    builder.get_graph().draw_mermaid_png(output_file_path="flow.png")
    return builder


if __name__ == "__main__":
    agent_graph = create_graph()
    inputs = HumanMessage(
        content="""
    Make this tweet better: "
    @langchainAI - newly Tool Calling feature is seroiusly underrated.

    After a long wait, it's here - making the implementation of agents across different models with function calling - super easy.

    made a video convering their newest blog post
    """
    )
    response = agent_graph.invoke(inputs)
    print(f"Final Result: \n{response[-1].content}")
