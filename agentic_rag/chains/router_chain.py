import sys

sys.path.append(
    "/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag"
)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv(override=True)


class RouterAgent(BaseModel):
    """
    Route a user query to the most relevant datasource.
    """

    datasource: str = Field(
        description="Given a user question, redirect the query to either 'vectorstore' or 'websearch'"
    )


router_llm = ChatOpenAI(model="gpt-4o", temperature=0.0).with_structured_output(
    RouterAgent
)

system = """
You are an expert in routing a user question to a vectore store or a web-search.
The vector store contains information related to agents, prompt engineering and adverserial attacks.
Use the vectorstore for question on these topics. For all else, use web-search.
"""

router_template = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

routing_agent: RunnableSequence = router_template | router_llm

if __name__ == "__main__":
    # result = routing_agent.invoke({"question": "what is agent memory"})
    result = routing_agent.invoke({"question": "what is the rule in Cricket"})
    print(result)
