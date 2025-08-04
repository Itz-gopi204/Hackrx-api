#-*- Imports -*-
import os
import re
import json
import time
import logging
import asyncio
import tempfile
import threading
import requests
import uvicorn
import warnings
from typing import List
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- LangChain Imports ---
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# --- Initial Setup ---
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# --- Global State & Rate Limiting ---
LLM_CALLS = []
LLM_LIMIT = 55  # requests per minute (60 seconds)
LLM_LOCK = threading.Lock()
ml_models = {}

# --- Rate Limiting Functions ---
def clean_old_calls():
    """Removes timestamps older than 60 seconds from the call list."""
    now = time.time()
    with LLM_LOCK:
        while LLM_CALLS and now - LLM_CALLS[0] > 60:
            LLM_CALLS.pop(0)
    logging.debug(f"[RateLimiter] Cleaned old calls. Current count: {len(LLM_CALLS)}")

async def wait_for_llm_slot():
    """Blocks until a slot is available based on the rate limit."""
    while True:
        clean_old_calls()
        with LLM_LOCK:
            if len(LLM_CALLS) < LLM_LIMIT:
                LLM_CALLS.append(time.time())
                logging.info(f"[RateLimiter] LLM call allowed. Total in last 60s: {len(LLM_CALLS)}")
                break
            else:
                logging.warning(f"[RateLimiter] LLM rate limit hit. Waiting. Current: {len(LLM_CALLS)}/{LLM_LIMIT}")
        await asyncio.sleep(1)

# --- Helper Functions ---
def download_file(url: str):
    """Downloads a file from a URL and returns its local path and extension."""
    resp = requests.get(url)
    resp.raise_for_status()
    path = requests.utils.urlparse(url).path
    suffix = os.path.splitext(path)[1][1:].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(resp.content)
        return tmp.name, suffix

def load_document_by_ext(file_path: str, ext: str):
    """Loads a document using the appropriate LangChain loader based on its extension."""
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "eml":
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, and EML are supported.")
    return loader.load()

def trim_answer(answer: str) -> str:
    return re.sub(r'\s+', ' ', answer)

# --- FastAPI Application ---
app = FastAPI(title="HackRx RAG API")

@app.on_event("startup")
async def startup_event():
    """Loads models and sets up the environment when the application starts."""
    logging.info("[Startup] Loading embeddings model...")
    ml_models["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

    if "SSL_CERT_FILE" in os.environ:
        del os.environ["SSL_CERT_FILE"]
        logging.info("[Startup] Removed SSL_CERT_FILE from environment.")

    logging.info("[Startup] Loading LLM model...")
    ml_models["llm"] = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="moonshotai/kimi-k2-instruct"
    )
    logging.info("[Startup] LLM model loaded successfully.")

# --- Pydantic Models ---
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]


@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_run(request: QARequest):
    """Processes a document and answers questions using a RAG pipeline."""
    logging.info(f"[hackrx_run] Received request for document: {request.documents}")
    try:
        # 1. Download and Load Document
        file_path, ext = download_file(request.documents)
        logging.info(f"[hackrx_run] Downloaded {ext} file to {file_path}")
        documents = load_document_by_ext(file_path, ext)
        os.remove(file_path)
        logging.info(f"[hackrx_run] Loaded and removed temp file {file_path}")

        # 2. Split and Create Vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=130)
        splits = text_splitter.split_documents(documents)
        logging.info(f"[hackrx_run] Split document into {len(splits)} chunks")
        
        vectorstore = await asyncio.to_thread(
            FAISS.from_documents, splits, ml_models["embeddings"]
        )
        logging.info("[hackrx_run] Built FAISS vectorstore")

        # 3. Setup RAG Chain
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are a helpful AI assistant specialized in question answering related to any matter. "
            "Use the provided context to answer precisely. If the answer is not in the context, say so. "
            "Give only the answer related to the question asked "
            "and don't deviate from the questin asked. "
            "Provide numbers in numerical format. Extract relevant table data if the question requires it.\n\n"
            "Context: {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        rag_chain = create_stuff_documents_chain(
            llm=ml_models["llm"],
            prompt=qa_prompt,
            output_parser=StrOutputParser()
        )
        retrieval_chain = create_retrieval_chain(retriever, rag_chain)

        # 4. Process questions concurrently
        async def get_rag_answer(q: str):
            await wait_for_llm_slot()
            try:
                logging.info(f"[get_rag_answer] Processing question: {q}")
                response = await asyncio.to_thread(retrieval_chain.invoke, {"input": q})
                return trim_answer(response['answer'])
            except Exception as e:
                logging.error(f"[get_rag_answer] Error for question '{q}': {e}")
                return f"Error processing question: {str(e)}"

        logging.info(f"[hackrx_run] Processing {len(request.questions)} questions...")
        all_answers = await asyncio.gather(*(get_rag_answer(q) for q in request.questions))
        logging.info("[hackrx_run] Finished processing all questions.")
        
        return QAResponse(answers=all_answers)

    except Exception as e:
        logging.error(f"[hackrx_run] A critical error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
