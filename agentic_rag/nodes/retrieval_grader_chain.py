import sys

sys.path.append(
    "/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag"
)

from state import customGraph
from chains.retrieval_grader import retrieval_grader
from chains.retriever import vector_db


def grade_documents(state: customGraph):
    """
    Grades the retrieved documents and invokes the websearch flag if required
    """
    print("*" * 8, "Documents Grader")
    question = state["question"]
    docs = state["documents"]
    filtered_msgs = []
    websearch_flag = False

    for doc in docs:
        relevant_flag = retrieval_grader.invoke(
            {"documents": doc, "question": question}
        ).binary_score
        if relevant_flag == "yes":
            filtered_msgs.append(doc)
        else:
            print("*" * 8, "Would trigger web search")
            websearch_flag = True

    if len(docs) == 0:
        websearch_flag = True

    return {
        "question": question,
        "documents": filtered_msgs,
        "web_search": websearch_flag,
    }


if __name__ == "__main__":
    state = {}
    state["question"] = "Adverserial attack on LLM"
    state["documents"] = vector_db.invoke(state["question"])
    state = grade_documents(state)
    print(f"State: {state}")
