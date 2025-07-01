from langgraph.graph import StateGraph, START, END
from state import customGraph
from nodes.generation_node import generation_node
from nodes.retrieval_grader_chain import grade_documents
from nodes.retriever_chain import retriver_process
from nodes.webSearchTool import websearch_tool
from chains.hallucinator_grader import hallucinator_llm
from chains.answer_grader import answer_grader_chain
from chains.router_chain import routing_agent


def decide_websearch(state: customGraph):

    if state["web_search"]:
        return "trigger_webSearch"
    else:
        return "trigger_generation"


def decide_revision_flow(state: customGraph):

    question = state["question"]
    llm_generation = state["generation"]
    docs = state["documents"]
    hallucination_results = hallucinator_llm.invoke(
        {"relevant_docs": docs, "generated_result": llm_generation}
    )

    if hallucination_results.binary_score == "yes":

        answer_results = answer_grader_chain.invoke(
            {"question_asked": question, "generated_response": llm_generation}
        )
        if answer_results.binary_score == "yes":
            print("*" * 8, "Answer useful")
            return "useful answer"
        else:
            print("*" * 8, "Answer not useful")
            return "not useful answer"

    else:
        print("*" * 8, "Hallucination Detected")
        return "hallucinated answer"


def conditional_start_point(state: customGraph):
    question = state["question"]
    path = routing_agent.invoke({"question": question}).datasource
    return path


def create_flow():
    graph = StateGraph(state_schema=customGraph)
    graph.add_node("retrieve", retriver_process)
    graph.add_node("grade_retrievedDocs", grade_documents)
    graph.add_node("use_web", websearch_tool)
    graph.add_node("generate_finalResult", generation_node)

    graph.add_conditional_edges(
        START,
        conditional_start_point,
        {"vectorstore": "retrieve", "websearch": "use_web"},
    )

    graph.add_edge("retrieve", "grade_retrievedDocs")
    graph.add_conditional_edges(
        "grade_retrievedDocs",
        decide_websearch,
        {"trigger_webSearch": "use_web", "trigger_generation": "generate_finalResult"},
    )
    graph.add_edge("use_web", "generate_finalResult")
    graph.add_conditional_edges(
        "generate_finalResult",
        decide_revision_flow,
        {
            "useful answer": END,
            "not_useful_answer": "use_web",
            "hallucinated answer": "use_web",
        },
    )
    flow = graph.compile()
    flow.get_graph().draw_mermaid_png(output_file_path="adaptive_self_rag.png")
    return flow


if __name__ == "__main__":
    flow = create_flow()
    # result = flow.invoke({"question": "What is agent memory?"})
    result = flow.invoke({"question": "How to bake a cake?"})
    print(f"Final Answer:\n{result["generation"]}")
