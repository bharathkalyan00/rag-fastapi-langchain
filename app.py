import os


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from fastapi import FastAPI
import uvicorn

# Fix for the Protobuf error
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# print(os.getenv("LANGCHAIN_API_KEY"))
google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")


def load_chain():
    # load
    docs = PyPDFLoader("attention.pdf").load()

    # chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    # embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_key
    )

    # vector db
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

    # retriever
    retriever = db.as_retriever()

    # prompt
    prompt = ChatPromptTemplate.from_template("""
    Only answer what is asked. Give your answer based on the context provided.
    Context : {Context}
    Question : {Question}
    """)
    # llm
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)

    chain = (
        {"Context": retriever, "Question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


chain = load_chain()

app = FastAPI(
    title="RAG with FastAPI",
    version="1.0",
    description="Existing RAG app with a FastAPI endpoint",
)

add_routes(app, chain, path=("/AttentionQnA"))

if __name__ == "__main__":
    # Get port from environment variable, default to 10000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
