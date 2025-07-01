from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import os

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

loader = WebBaseLoader(urls)
text_docs = loader.load()
doc_corpus = "\n---\n".join([i.page_content for i in text_docs])
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
documents = splitter.split_documents(text_docs)
chroma_db = Chroma.from_documents(
    documents=documents,
    collection_name="ai_docs",
    embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
    persist_directory=".",
)

vector_db = Chroma(
    collection_name="ai_docs",
    persist_directory=".",
    embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
).as_retriever()
