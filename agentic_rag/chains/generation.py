from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-4o")
generation_chain = prompt | llm | StrOutputParser()
