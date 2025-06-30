import sys
sys.path.append("/Users/A118390615/Library/CloudStorage/OneDrive-DeutscheTelekomAG/Projects/COE_Projects/langgraph_course_1/agentic_rag")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv(override=True)

vector_db = Chroma(collection_name = "ai_docs",
                   persist_directory=".",
                   embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002")).as_retriever()

if __name__ == "__main__":
    results = vector_db.invoke("agent memory")
    print(results)
