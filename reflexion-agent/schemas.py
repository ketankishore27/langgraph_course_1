from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    unnecessary: str = Field(description="Critique of what is unnecessary")

class AnswerQuestion(BaseModel):
    """
    Answer the question.
    """
    answer: str = Field(description="Detailed Answer to the query in around 250 words")
    reflection: Reflection = Field(description="Your reflection on the initial response")
    search_string: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """
    Revise your original answer to your question.
    """
    References: List[str] = Field(
        description = "Citations Motivating the answer" 
    )