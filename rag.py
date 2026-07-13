import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable


# Configuration
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
CORPUS_PATH = "data"


# Load API keys
try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()

    if LLM_PROVIDER == "groq":
        os.environ["GROQ_API_KEY"] = secrets.get_secret(
            "GROQ_API_KEY"
        )

    elif LLM_PROVIDER == "gemini":
        os.environ["GOOGLE_API_KEY"] = secrets.get_secret(
            "GOOGLE_API_KEY"
        )

    elif LLM_PROVIDER == "openai":
        os.environ["OPENAI_API_KEY"] = secrets.get_secret(
            "OPENAI_API_KEY"
        )

    os.environ["LANGCHAIN_API_KEY"] = secrets.get_secret(
        "LANGCHAIN_API_KEY"
    )

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

except Exception:
    load_dotenv()

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"


# Load HR PDF documents
loader = PyPDFDirectoryLoader(CORPUS_PATH)

documents = loader.load()


# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)


# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS vector database
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)


# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6
    }
)


# Initialize Groq LLM
if LLM_PROVIDER == "groq":

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512
    )

elif LLM_PROVIDER == "gemini":

    from langchain_google_genai import (
        ChatGoogleGenerativeAI
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1,
        max_output_tokens=512
    )

elif LLM_PROVIDER == "openai":

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512
    )

else:

    raise ValueError(
        "Unsupported LLM provider."
    )


# RAG prompt
RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are an HR Assistant for Zyro Dynamics.

Answer ONLY using the provided context.

Rules:

1. If the context fully answers the question,
provide the complete answer.

2. If the context contains only partial information,
provide the available information and clearly state
that additional details are not available in the
retrieved documents.

3. Do NOT make up or assume facts.

4. Only if the retrieved context has no relevant
information at all, reply:

"I can only answer questions based on
Zyro Dynamics HR policy documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


# Format retrieved documents
def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# RAG pipeline
@traceable
def rag_chain(question: str):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response


# Refusal response
REFUSAL_MESSAGE = (
    "I can only answer questions based on "
    "Zyro Dynamics HR policy documents."
)


# Main chatbot function
@traceable
def ask_bot(question: str):

    answer = rag_chain(question)

    if REFUSAL_MESSAGE.lower() in answer.lower():

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }

    docs = retriever.invoke(question)

    sources = list(
        set(
            doc.metadata.get(
                "source",
                "Unknown"
            )
            for doc in docs
        )
    )

    return {
        "answer": answer,
        "sources": sources
    }
