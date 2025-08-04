# Debug logging
import threading
import logging
import time
import os
import requests
import tempfile
import re
import asyncio

# --- Third-party imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from dotenv import load_dotenv

# --- Vertex AI and LangChain specific imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain_core.documents import Document as LangchainDocument


logging.basicConfig(level=logging.INFO)

# --- Global state & Rate Limiting ---
LLM_CALLS = []
# Set to match a high-RPM Gemini model like Flash 2.0 (e.g., 2000 RPM). Using 1800 as a buffer.
LLM_LIMIT = 1800
LLM_LOCK = threading.Lock()

def clean_old_calls():
    now = time.time()
    with LLM_LOCK:
        while LLM_CALLS and now - LLM_CALLS[0] > 60:
            LLM_CALLS.pop(0)
    logging.debug(f"[RateLimiter] Cleaned old calls. Current count: {len(LLM_CALLS)}")

async def wait_for_llm_slot():
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

# --- FastAPI App Initialization ---
load_dotenv()
ml_models = {}

app = FastAPI(title="RAG API with Google Vertex AI")

@app.on_event("startup")
async def startup_event():
    # --- Initialize Vertex AI for the generative model ---
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GCP_REGION = os.getenv("GCP_REGION")
    if not GCP_PROJECT_ID or not GCP_REGION:
        raise ValueError("GCP_PROJECT_ID and GCP_REGION must be set in .env file")
    
    try:
        # This uses the GOOGLE_APPLICATION_CREDENTIALS environment variable
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        logging.info(f"Vertex AI initialized successfully for project '{GCP_PROJECT_ID}' in region '{GCP_REGION}'.")
    except Exception as e:
        raise RuntimeError(f"Could not initialize Vertex AI. Ensure you have authenticated correctly. Error: {e}")

    # --- Configure Models ---
    
    # 1. Use Google AI Studio Embeddings (via API Key)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY must be set in your .env file for embeddings.")
    embedding_model_name = "models/embedding-001"
    ml_models["embeddings"] = GoogleGenerativeAIEmbeddings(
        model=embedding_model_name,
        google_api_key=GOOGLE_API_KEY
    )
    logging.info(f"[Startup] Loaded Google AI Embeddings model: {embedding_model_name}")

    # 2. Configure the Gemini generative model from Vertex AI
    generative_model_name = "gemini-2.5-pro"
    ml_models["generative_model"] = GenerativeModel(generative_model_name)
    logging.info(f"[Startup] Loaded Vertex AI Generative model: {generative_model_name}")

    # 3. Define the system prompt for Gemini
    system_prompt = (
        "You are a helpful AI assistant specialized in question answering related to anything from the document chunks. "
        "Use ONLY the provided context from the documents to answer the question as clearly and precisely as possible. "
        "If the answer is not found in the provided context, state that the information is not available in the documents. "
        "do NOT use unnecessary spaces, new lines, or tabs in general answers. "
        "Only use new lines or formatting when required for clarity, such as for tables, lists, or when the document context demands it. "
        "Do not provide any background or causes. "
        "Give the answer without any deviations from the question asked. "
        "If any answers reflect numbers, give the number in numerical format. "
        "If any question requires a table, extract and format only the relevant content in the table related to the query, using new lines only where needed for readability."
    )
    ml_models["system_prompt"] = system_prompt
    
# --- Pydantic Models ---
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --- Helper Functions (no changes here) ---
def download_file(url):
    resp = requests.get(url)
    resp.raise_for_status()
    path = requests.utils.urlparse(url).path
    suffix = os.path.splitext(path)[1][1:].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(resp.content)
        return tmp.name, suffix

def load_document_by_ext(file_path, ext):
    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "eml":
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, and EML are supported.")
    return loader.load()

# --- API Endpoints ---
@app.get("/")
def read_root():
    return "RAG API with Google Vertex AI is running."

@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_run(request: QARequest):

    logging.info("[hackrx_run] Request started. Document URL: %s, Number of questions: %d", request.documents, len(request.questions))
    try:
        logging.info("[hackrx_run] Downloading document...")
        file_path, ext = download_file(request.documents)
        logging.info(f"[hackrx_run] Document downloaded to {file_path} with extension {ext}")

        logging.info("[hackrx_run] Loading document by extension...")
        documents = load_document_by_ext(file_path, ext)
        logging.info(f"[hackrx_run] Loaded document. Number of pages/chunks: {len(documents)}")
        os.remove(file_path)
        logging.info(f"[hackrx_run] Temporary file {file_path} removed.")

        logging.info("[hackrx_run] Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        logging.info(f"[hackrx_run] Document split into {len(splits)} chunks.")

        logging.info("[hackrx_run] Creating FAISS vectorstore...")
        vectorstore = await asyncio.to_thread(
            FAISS.from_documents, splits, ml_models["embeddings"]
        )
        logging.info("[hackrx_run] FAISS vectorstore created.")

        logging.info("[hackrx_run] Creating retriever...")
        retriever = vectorstore.as_retriever()
        logging.info("[hackrx_run] Retriever created.")

        async def get_rag_answer(question: str) -> str:
            logging.info(f"[get_rag_answer] Started for question: {question}")
            await wait_for_llm_slot()
            logging.info(f"[get_rag_answer] LLM slot acquired for question: {question}")

            logging.info(f"[get_rag_answer] Retrieving relevant documents for question: {question}")
            retrieved_docs: List[LangchainDocument] = await retriever.ainvoke(question)
            logging.info(f"[get_rag_answer] Retrieved {len(retrieved_docs)} relevant documents for question: {question}")

            context_str = "\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in retrieved_docs])
            logging.info(f"[get_rag_answer] Constructed context string for question: {question}")

            prompt_parts = [
                ml_models["system_prompt"],
                "\n\nCONTEXT:\n",
                context_str,
                f"\n\nQUESTION: {question}"
            ]
            logging.info(f"[get_rag_answer] Prompt constructed for question: {question}")

            try:
                logging.info(f"[get_rag_answer] Calling Vertex AI for question: {question[:50]}...")
                model: GenerativeModel = ml_models["generative_model"]
                response = await model.generate_content_async(
                    prompt_parts,
                    generation_config={"temperature": 0.2},
                )
                logging.info(f"[get_rag_answer] Received response from Vertex AI for question: {question}")
                answer = response.text.strip()
                logging.info(f"[get_rag_answer] Final answer for question: {question}: {answer}")
                return answer
            except Exception as e:
                logging.error(f"[get_rag_answer] Error calling Vertex AI for question: {question}: {e}")
                return f"Error processing question: {e}"

        logging.info(f"[hackrx_run] Processing {len(request.questions)} questions asynchronously...")
        all_answers = await asyncio.gather(*(get_rag_answer(q) for q in request.questions))
        logging.info("[hackrx_run] Finished processing all questions.")
        logging.info(f"[hackrx_run] Final answers: {all_answers}")
        return QAResponse(answers=all_answers)
    except Exception as e:
        logging.error(f"[hackrx_run] Unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Assumes the script is named 'main.py'
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
