from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage


class customState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
