import sys

sys.path.append(
    "/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag"
)

from chains.generation import generation_chain
from state import customGraph


def generation_node(state: customGraph):
    """
    This creates the RAG Generation for the input question
    """
    print("*" * 8, "Generation Node")
    question = state["question"]
    docs = state["documents"]
    retrieved_results = generation_chain.invoke({"question": question, "context": docs})
    return {"question": question, "generation": retrieved_results, "documents": docs}
