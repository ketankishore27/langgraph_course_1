import sys

sys.path.append(
    "/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag"
)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from nodes.retriever_chain import retriver_process
from nodes.generation_node import generation_node
from langchain_core.runnables import RunnableSequence


class GradeAnswer(BaseModel):
    """
    Binary score to tell if Answer address the question.
    """

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


answer_grader = ChatOpenAI(model="gpt-4o", temperature=0.0).with_structured_output(
    GradeAnswer
)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User Question:\n {question_asked}, LLM Generation:\n {generated_response}",
        ),
    ]
)

answer_grader_chain: RunnableSequence = answer_prompt | answer_grader

if __name__ == "__main__":
    question = "agent_memory"
    documents = retriver_process(state={"question": question})["documents"]
    generation = generation_node(state={"question": question, "documents": documents})[
        "generation"
    ]
    answer_grader = answer_grader_chain.invoke(
        {"question_asked": question, "generated_response": generation}
    )
    print(answer_grader)
