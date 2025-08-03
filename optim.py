import logging
logging.basicConfig(level=logging.INFO)
import os
import tempfile
import requests
from dotenv import load_dotenv
import re
from typing import List
import time
import json
import warnings
import hashlib # <-- 1. ADDED HASHLIB TO CREATE UNIQUE FILENAMES
warnings.filterwarnings("ignore", category=FutureWarning)

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader

load_dotenv()

# --- Functions (No changes needed here) ---
def download_file(url):
    resp = requests.get(url)
    resp.raise_for_status()
    path = requests.utils.urlparse(url).path
    suffix = os.path.splitext(path)[1][1:].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name, suffix

def load_document_by_ext(file_path, ext):
    if ext == "pdf": loader = PyPDFLoader(file_path)
    elif ext == "docx": loader = Docx2txtLoader(file_path)
    elif ext == "eml": loader = UnstructuredEmailLoader(file_path)
    else: raise ValueError("Unsupported file type.")
    return loader.load()

def trim_answer(answer, max_sentences=3, max_chars=350):
    sentences = re.split(r'(?<=[.!?]) +', answer)
    trimmed = ' '.join(sentences[:max_sentences])
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rsplit(' ', 1)[0] + '...'
    # Remove newlines and extra spaces
    trimmed = trimmed.replace('\n', '.')
    trimmed = re.sub(r'\s+', ' ', trimmed).strip()
    return trimmed


from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
from contextlib import asynccontextmanager
import os
import requests
import tempfile
import re
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredEmailLoader


# Load environment variables from .env file
load_dotenv()
ml_models = {}

# Initialize FastAPI App
app = FastAPI(title="HackRx RAG API")

# Use the @app.on_event decorator for startup tasks
@app.on_event("startup")
async def startup_event():
    logging.info("[Startup] Loading Google embeddings and Ollama LLM...")
    ml_models["embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ml_models["llm"] = ChatOllama(model=ollama_model, base_url=ollama_base_url)
    logging.info(f"[Startup] Ollama LLM loaded: {ollama_model} at {ollama_base_url}, Google embeddings loaded.")



class QARequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]



def load_document_by_ext(file_path, ext):
    # Try both PyMuPDFLoader and PyPDFLoader for PDFs
    if ext == "pdf":
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
        except Exception:
            loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "eml":
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only PDF, DOCX, and EML are supported.")
    docs = loader.load()
    return docs

def download_file(url):
    resp = requests.get(url)
    resp.raise_for_status()
    path = requests.utils.urlparse(url).path
    suffix = os.path.splitext(path)[1][1:].lower()
    if not suffix:
        # Try to guess from content-type
        content_type = resp.headers.get('content-type', '').lower()
        if 'pdf' in content_type:
            suffix = 'pdf'
        elif 'word' in content_type or 'docx' in content_type:
            suffix = 'docx'
        elif 'eml' in content_type or 'email' in content_type:
            suffix = 'eml'
        else:
            raise ValueError("Could not determine file type from URL or content-type header.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name, suffix

def trim_answer(answer, max_sentences=3, max_chars=200):
    sentences = re.split(r'(?<=[.!?]) +', answer)
    trimmed = ' '.join(sentences[:max_sentences])
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rsplit(' ', 1)[0] + '...'
    return trimmed
@app.get("/")
def read_root():
    return {"message": "HackRx RAG API is running."}




@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_run(request: QARequest):
    logging.info("[hackrx_run] Received request with documents: %s and %d questions", request.documents, len(request.questions))
    try:
        file_path, ext = download_file(request.documents)
        logging.info(f"[hackrx_run] Downloaded file to {file_path} with extension {ext}")
        documents = load_document_by_ext(file_path, ext)
        logging.info(f"[hackrx_run] Loaded document, got {len(documents)} chunks")
        os.remove(file_path)
        logging.info(f"[hackrx_run] Removed temp file {file_path}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        logging.info(f"[hackrx_run] Split document into {len(splits)} splits")

        vectorstore = await asyncio.to_thread(
            FAISS.from_documents, splits, ml_models["embeddings"]
        )
        logging.info(f"[hackrx_run] Built FAISS vectorstore")

        # Set up retriever and RAG chain
        retriever = vectorstore.as_retriever()
        from langchain_core.output_parsers import StrOutputParser
        parser = StrOutputParser()
        system_prompt = (
           "You are a helpful AI assistant specialized in question answering from the documents that are divided into chunks . "
            "Use the provided context to answer the question as clearly and precisely as possible. "
            "If the answer is not known from the context, then give the answer which is related to the contest. "
            "Do not provide any background, or causes. "
            "Give the answer as you are a human"
            "Keep the answers that are needed don't bluff or give answers about other matters"
            "and if there are any answers reflecting the numbers also give the number in numerical format. "
            "and if there are any questions that have answer in the table so extract the content in the table related to the query. "
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        rag_chain = create_stuff_documents_chain(
            llm=ml_models["llm"],
            prompt=qa_prompt,
            output_parser=parser
        )
        retrieval_chain = create_retrieval_chain(retriever, rag_chain)
        async def get_rag_answer(q: str):
            try:
                logging.info(f"Processing question: '{q}'")
                response = await retrieval_chain.ainvoke({"input": q})
                answer = trim_answer(response['answer'])
                logging.info(f"Answer for '{q}': '{answer}'")
                return answer
            except Exception as e:
                logging.error(f"Error processing question '{q}': {e}")
                return "Error processing this question."

        # async def get_rag_answer(q):
        #     loop = asyncio.get_event_loop()
        #     try:
        #         logging.info(f"[get_rag_answer] Processing question: {q}")
        #         response = await loop.run_in_executor(None, retrieval_chain.invoke, {"input": q})
        #         logging.info(f"[get_rag_answer] Got response for question: {q}")
        #         return trim_answer(response['answer'])
        #     except Exception as e:
        #         logging.error(f"[get_rag_answer] Error for question: {q} - {e}")
        #         return f"Error: {str(e)}"

        # Process all questions asynchronously


        logging.info(f"[hackrx_run] Processing {len(request.questions)} questions async...")
        all_answers = await asyncio.gather(*(get_rag_answer(q) for q in request.questions))
        logging.info(f"[hackrx_run] Finished processing all questions.")
        return QAResponse(answers=all_answers)
    except Exception as e:
        logging.error(f"[hackrx_run] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

