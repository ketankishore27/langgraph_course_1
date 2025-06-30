from langgraph.graph import StateGraph, START, END
from state import customGraph
from nodes.generation_node import generation_node
from nodes.retrieval_grader_chain import grade_documents
from nodes.retriever_chain import retriver_process
from nodes.webSearchTool import websearch_tool

def decide_websearch(state: customGraph):

    if state["web_search"]:
        return "trigger_webSearch"
    else:
        return "trigger_generation"
    
def create_flow():
    graph = StateGraph(state_schema = customGraph)
    graph.add_node("retrieve", retriver_process)
    graph.add_node("grade_retrievedDocs", grade_documents)
    graph.add_node("use_web", websearch_tool)
    graph.add_node("generate_finalResult", generation_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_retrievedDocs")
    graph.add_conditional_edges("grade_retrievedDocs", 
                                decide_websearch, 
                                {
                                    "trigger_webSearch": "use_web",
                                    "trigger_generation": "generate_finalResult"
                                })
    graph.add_edge("use_web", "generate_finalResult")
    graph.add_edge("generate_finalResult", END)
    flow = graph.compile()
    return flow

if __name__ == "__main__":
    flow = create_flow()
    result = flow.invoke({"question": "Bake a cake"})
    print(f"Final Answer:\n{result["generation"]}")

