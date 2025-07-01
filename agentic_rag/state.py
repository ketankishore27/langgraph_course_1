from typing import TypedDict, List


class customGraph(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question: Question
        generation: LLM Generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    final_result: str
