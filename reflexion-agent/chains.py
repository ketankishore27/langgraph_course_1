from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts.prompt import PromptTemplate
import datetime
from schemas import AnswerQuestion, ReviseAnswer
from dotenv import load_dotenv

load_dotenv(override=True)


actor_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher
            Current time: {time}

            1. {first_instruction}
            2. Reflect and Critique your answer. Be severe to maximize your improvement.
            3. Recommend search queries to research information and improve your answer. 
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format and make sure to suggest search queries in the required format (if asked for).",
        ),
    ]
).partial(time=datetime.datetime.now().isoformat())

llm = ChatOpenAI(model="o4-mini")
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])
first_responder_template = actor_template.partial(
    first_instruction="Provide a detailed~250 word answer"
)

chain_responder_initial = first_responder_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

first_responder = (
    first_responder_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    | parser_pydantic
)

revise_instruction = """
Revise your previous answer using the new information.
-  You should use the previous critique to add importatnt information.
    - You must include numerical citations in your revised answer to ensure it can be verified.
    - Add a "search_string" section to the bottom of your answer (which does not count towards the word limit). In form of:
        - [1] https://example.com
        - [2] https://example.com
"""

revisor_template = actor_template.partial(first_instruction=revise_instruction)

chain_revisor = revisor_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)


summarize_instruction = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Current time: {time}
            You are an assistant that summarizes multi-turn conversations or workflows.
            Below is a sequence of messages exchanged during a complex task. Each message may represent a user request, a system/tool response, or an assistant's reply. Your goal is to generate a **concise, informative summary** that captures the essence of the interaction.

            Summarize the entire process, focusing on:
            - The original objective or problem the user wanted to solve
            - Key steps or sub-tasks completed
            - Important findings, results, or tool outputs
            - Any suggestions or next steps provided

            ---

            Messages:
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            """
            ---
            Summarize the wconversation. 
            ---

            Summary: 
            """,
        ),
    ]
).partial(time=datetime.datetime.now().isoformat())

chain_summary = summarize_instruction | llm

if __name__ == "__main__":
    response = first_responder.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write about AI-powered SOC / autonomous problem domain, List startups which does this and raises captial."
                )
            ]
        }
    )
    print(response)
