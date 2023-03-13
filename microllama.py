"""The smallest possible LLM API"""

__version__ = "0.2"

import inspect
import json
import os
import sys
from functools import lru_cache
from typing import Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from openai import ChatCompletion

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index")
SOURCE_JSON = os.environ.get("SOURCE_JSON", "source.json")
MAX_RELATED_DOCUMENTS = int(os.environ.get("MAX_RELATED_DOCUMENTS", 5))
EXTRA_CONTEXT = os.environ.get(
    "EXTRA_CONTEXT",
    """
        Answer in no more than three sentences. If the answer is not included 
        in the context, say 'Sorry, this is no answer for this in my sources.'.
    """,
)
UVICORN_HOST = os.environ.get("UVICORN_HOST", "0.0.0.0")
UVICORN_PORT = int(os.environ.get("UVICORN_PORT", 8080))


def log(msg):
    sys.stdout.write(f"INFO:     {msg} \n")


def create_documents_from_texts():
    # create langchain documents from the text sources
    documents = []
    for item in json.load(open(SOURCE_JSON)):
        document = Document(
            page_content=item["content"],
            metadata={"source": item["source"]},
        )
        if item.get("url"):
            document.metadata["url"] = item["url"]
        documents.append(document)
    return documents


def get_text_chunks(sources=create_documents_from_texts()):
    # split the langchain documents into smaller chunks to reduce tokens
    # and improve accuracy
    # sourcery skip: for-append-to-extend
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    source_chunks = []
    for source in sources:
        for chunk in splitter.split_text(source.page_content):
            doc = Document(page_content=chunk, metadata=source.metadata)
            source_chunks.append(doc)
    return source_chunks


def get_index(path=FAISS_INDEX_PATH, create=False):
    # get a saved FAISS index if it exists, otherwise create a new one
    if os.path.exists(path) and not create:
        log("Loading index from disk")
        return FAISS.load_local(path, embeddings=OpenAIEmbeddings())
    log("Creating index")
    index = FAISS.from_documents(get_text_chunks(), OpenAIEmbeddings())
    index.save_local(path)
    return index


def find_similar_docs(question, index):
    # find similar documents to the question
    return index.similarity_search(question, k=MAX_RELATED_DOCUMENTS)


@lru_cache
def answer(question, index, extra_context=EXTRA_CONTEXT):
    similar_docs = find_similar_docs(question, index)
    sources = {
        (doc.metadata["source"], doc.metadata.get("url")) for doc in similar_docs
    }
    prompt_context = " ".join([doc.page_content for doc in similar_docs])
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": "Use the following context to answer the questions.",
        },
        {"role": "system", "content": prompt_context},
        {
            "role": "system",
            "content": "Don't mention the context in your answer.",
        },
    ]
    if extra_context:
        prompt_messages.append(
            {"role": "system", "content": inspect.cleandoc(extra_context)}
        )
    prompt_messages.append({"role": "user", "content": question})
    resp = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_messages,
    )
    answer = resp["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": sources}


def make_front_end():
    # create a simple front end for the API
    import pathlib
    import requests

    html = requests.get(
        "https://raw.githubusercontent.com/tomdyson/microllama/main/static/index.html"
    ).text

    pathlib.Path("static").mkdir(exist_ok=True)
    with open("static/index.html", "w") as f:
        f.write(html)


# FastAPI app
app = FastAPI()


@app.on_event("startup")
def index():
    global index
    index = get_index()


@app.get("/api/ask")
def ask(q: Union[str, None] = None):
    return {"response": answer(q, index)}


try:
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
except RuntimeError:
    log("static directory not found, front end will not be available")


def serve():
    import uvicorn

    uvicorn.run(app, host=UVICORN_HOST, port=UVICORN_PORT)
