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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
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


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
import os
import requests
import tempfile
import re
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader

load_dotenv()

app = FastAPI()

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

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
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "eml":
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, and EML are supported.")
    docs = loader.load()
    return docs

def trim_answer(answer, max_sentences=3, max_chars=350):
    sentences = re.split(r'(?<=[.!?]) +', answer)
    trimmed = ' '.join(sentences[:max_sentences])
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rsplit(' ', 1)[0] + '...'
    return trimmed

@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_run(request: QARequest):
    
    file_path, ext = download_file(request.documents)
    documents = load_document_by_ext(file_path, ext)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
    parser=StrOutputParser()
    system_prompt = (
        "You are a helpful AI assistant specialized in question answering related to insurance policies and related documents. "
        "Use the provided context to answer the question as clearly and precisely as possible. "
        "If the answer is not known from the context, then give the answer which is related to the contest "
        "Keep answers concise, within two to three sentences.\n\n"
        "and if there are any answers reflecting the numbers also give the number in numerical format."
        "and if their are any questions that have answer in the table so extract the content in the table related to the query"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    Youtube_chain = create_stuff_documents_chain(llm=llm,prompt=qa_prompt,output_parser=parser)
    retrieval_chain = create_retrieval_chain(retriever, Youtube_chain)

    async def get_answer(q):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, retrieval_chain.invoke, {"input": q})
        return trim_answer(response['answer'])

    answers = await asyncio.gather(*(get_answer(q) for q in request.questions))
    return QAResponse(answers=answers)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)