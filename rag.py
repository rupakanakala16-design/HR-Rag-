import os

import streamlit as st
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

# Resolve "data" relative to THIS file, not the process's current working
# directory. Streamlit Cloud can launch the app from a different CWD than
# the repo root, which silently breaks a bare relative path like "data".
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "data")


def _setup_environment():
    """Populate os.environ with API keys from whichever secret store
    is available: Kaggle secrets, Streamlit secrets, or a local .env file.
    """

    # Kaggle notebooks
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()

        if LLM_PROVIDER == "groq":
            os.environ["GROQ_API_KEY"] = secrets.get_secret("GROQ_API_KEY")
        elif LLM_PROVIDER == "gemini":
            os.environ["GOOGLE_API_KEY"] = secrets.get_secret("GOOGLE_API_KEY")
        elif LLM_PROVIDER == "openai":
            os.environ["OPENAI_API_KEY"] = secrets.get_secret("OPENAI_API_KEY")

        os.environ["LANGCHAIN_API_KEY"] = secrets.get_secret("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
        return
    except Exception:
        pass

    # Streamlit Cloud: keys live in st.secrets (Settings -> Secrets), not a
    # committed .env file (which is normally gitignored and won't deploy).
    try:
        if hasattr(st, "secrets") and len(st.secrets) > 0:
            for key in (
                "GROQ_API_KEY",
                "GOOGLE_API_KEY",
                "OPENAI_API_KEY",
                "LANGCHAIN_API_KEY",
            ):
                if key in st.secrets:
                    os.environ[key] = st.secrets[key]

            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_PROJECT", "zyro-rag-challenge")
            return
    except Exception:
        pass

    # Local development
    load_dotenv()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "zyro-rag-challenge")


def _load_documents():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"Data folder not found at '{CORPUS_PATH}'. Make sure a "
            f"'data' folder containing your HR PDFs is committed to the "
            f"repo (check .gitignore) and sits next to rag.py."
        )

    pdf_files = [f for f in os.listdir(CORPUS_PATH) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(
            f"'{CORPUS_PATH}' exists but contains no PDF files. Add at "
            f"least one .pdf and make sure it's actually committed to git "
            f"(not just present locally)."
        )

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    if not documents:
        raise ValueError(
            f"Found {len(pdf_files)} PDF file(s) in '{CORPUS_PATH}' but "
            f"none could be loaded. They may be corrupted or password "
            f"protected."
        )

    return documents


@st.cache_resource(show_spinner=False)
def _build_pipeline():
    """Build (and cache across reruns) the retriever + LLM.

    Streamlit reruns the whole script on every interaction. Without
    caching, this function's contents would re-load PDFs, re-embed, and
    re-build the FAISS index on every single button click.
    """

    _setup_environment()

    documents = _load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    if len(chunks) == 0:
        raise ValueError(
            f"Loaded {len(documents)} document page(s) from '{CORPUS_PATH}' "
            f"but they produced zero text chunks. This usually means the "
            f"PDFs are scanned images with no selectable text (pypdf can't "
            f"OCR them) — try re-exporting them as text-based PDFs, or run "
            f"them through an OCR step first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)

    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=0.1, max_output_tokens=512
        )

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, max_tokens=512)

    else:
        raise ValueError("Unsupported LLM provider.")

    return retriever, llm


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

    retriever, llm = _build_pipeline()

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

    retriever, _ = _build_pipeline()

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
