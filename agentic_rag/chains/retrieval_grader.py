from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)


class GradeDocument(BaseModel):
    """
    Binary score for relevance check on retrievedd document.
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm = ChatOpenAI(model="gpt-4o", temperature=0.0).with_structured_output(
    GradeDocument
)

system_msg = """
You are a grader assessing relevance of a retrieved document to a user question. 
If the document conatins keyword(s) or semantic menaing related to the question, grade it as relevant.
Give it a bonary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_msg),
        (
            "human",
            "Retrieved Document: \n\n {documents} \n\n User Question: {question}",
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm
