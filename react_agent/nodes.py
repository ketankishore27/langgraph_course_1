from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from state import customState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage

load_dotenv(override=True)

tavily_search = TavilySearch(max_results=5)


@tool
def triple_num(num: float):
    """
    This tool accepts a number, triples it and returns the tripled number
    """
    return num * 3


tools = [tavily_search, triple_num]

llm = ChatOpenAI(model="gpt-4o", temperature=0.0).bind_tools(tools)

tool_node = ToolNode(tools=tools)


def reasoning_node(state: customState):
    output = llm.invoke(state["messages"])
    return {"messages": [output]}


def tool_node_handler(state: customState):
    output_retriever = tool_node.invoke(state["messages"])
    return {"messages": output_retriever}


if __name__ == "__main__":
    print(tavily_search.invoke({"query": "Current Temperature in Pune"}))
