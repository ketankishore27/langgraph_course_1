import sys
sys.path.append("/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Dict, Any
from state import customGraph
from ingestion import vector_db
load_dotenv(override=True)

def retriver_process(state: customGraph) -> Dict[str, Any]:
    print("*" * 8, "Retrieve Documents")
    question = state['question']
    docs = vector_db.invoke(question)
    return {"question": question, "documents": docs}

if __name__ == "__main__":
    results = retriver_process({"question": "LLM adverserial attacks"})
    print(results)