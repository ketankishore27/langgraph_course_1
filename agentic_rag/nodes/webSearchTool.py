import sys

sys.path.append(
    "/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag"
)

from langchain_tavily import TavilySearch
from langchain_core.documents.base import Document
from state import customGraph
from dotenv import load_dotenv

load_dotenv(override=True)

search = TavilySearch(max_results=6)


def websearch_tool(state: customGraph):
    """
    This function invokes the websearch for the question if the web_search flag in the GraphState is True
    """
    print("*" * 8, "Documents Grader")
    question = state["question"]
    documents = state.get("documents", None)

    search_results = search.invoke({"query": question})

    web_results = Document(
        page_content="\n---\n".join(
            [res["content"] for res in search_results["results"]]
        )
    )

    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"question": question, "documents": documents}


if __name__ == "__main__":
    results = websearch_tool({"question": "Agent Memory", "documents": None})
    print(results)
