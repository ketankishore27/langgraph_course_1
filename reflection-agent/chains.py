from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
             You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet.
             Always provide a detailed recommendations, including request for length, virality, style, etc.
             """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a twitter techie influencer assistant tasked with writing excellent twitter post.
            Generate the best twitter post possible for the users request.
            If the user provides critique, respond with a revised version of your previous attempts
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")

reflection_chain = reflection_prompt | llm

generation_chain = generation_prompt | llm
