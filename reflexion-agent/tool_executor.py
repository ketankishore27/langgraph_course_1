from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from schemas import AnswerQuestion, ReviseAnswer
from dotenv import load_dotenv
load_dotenv(override=True)

tavily_search = TavilySearch(max_result = 5)

def run_queries(search_string: list[str], **kwargs):
    """
    Runs Google search for all the queries
    """
    return tavily_search.batch([{"query" : query} for query in search_string])

executor_node = ToolNode(
    tools=[
        StructuredTool.from_function(func=run_queries, name = AnswerQuestion.__name__),
        StructuredTool.from_function(func=run_queries, name = ReviseAnswer.__name__)
    ]
)



