import sys

sys.path.append(
    "/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag"
)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from nodes.retriever_chain import retriver_process
from nodes.generation_node import generation_node


class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generation answer.
    """

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


hallucinator_detector = ChatOpenAI(
    model="gpt-4o", temperature=0.0
).with_structured_output(GradeHallucinations)

system_template = """
You are assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
Give a binary score 'yes' or 'no'. 'yes' means the answer is grounded in / supported by the set of facts.
"""
hallucination_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        (
            "human",
            "Set of facts: \n{relevant_docs}, LLM Generations: \n{generated_result}",
        ),
    ]
)

hallucinator_llm: RunnableSequence = hallucination_template | hallucinator_detector

if __name__ == "__main__":
    question = "agent memory"
    documents = retriver_process(state={"question": question})["documents"]
    generation = generation_node(state={"question": question, "documents": documents})[
        "generation"
    ]
    result = hallucinator_llm.invoke(
        {"relevant_docs": documents, "generated_result": generation}
    )
    print(result)
