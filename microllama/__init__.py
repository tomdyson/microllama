"""The smallest possible LLM API"""

__version__ = "0.4.10"

import inspect
import json
import os
import sys
from functools import lru_cache
from typing import Optional, Union

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from openai import ChatCompletion
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index")
SOURCE_JSON = os.environ.get("SOURCE_JSON", "source.json")
MAX_RELATED_DOCUMENTS = int(os.environ.get("MAX_RELATED_DOCUMENTS", 5))
EXTRA_CONTEXT = os.environ.get(
    "EXTRA_CONTEXT",
    """
        Answer in no more than three sentences. If the answer is not included 
        in the context, say 'Sorry, there is no answer for this in my sources.'.
    """,
)
UVICORN_HOST = os.environ.get("UVICORN_HOST", "0.0.0.0")
UVICORN_PORT = int(os.environ.get("UVICORN_PORT", 8080))
DEBUG = os.environ.get("DEBUG", False)
MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")


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


def get_text_chunks():
    # split the langchain documents into smaller chunks to reduce tokens
    # and improve accuracy
    # sourcery skip: for-append-to-extend
    sources = create_documents_from_texts()
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
        model=MODEL,
        messages=prompt_messages,
    )
    answer = resp["choices"][0]["message"]["content"].strip()
    if DEBUG:
        return {
            "answer": answer,
            "sources": sources,
            "prompt_messages": prompt_messages,
        }
    return {"answer": answer, "sources": sources}


def streaming_answer(question, index, extra_context=EXTRA_CONTEXT):
    similar_docs = find_similar_docs(question, index)
    sources = {
        (doc.metadata["source"], doc.metadata.get("url")) for doc in similar_docs
    }
    yield f"SOURCES::{json.dumps(list(sources))}"
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
    if DEBUG:
        yield f"PROMPT::{json.dumps(prompt_messages)}"
    resp = ChatCompletion.create(
        model=MODEL,
        messages=prompt_messages,
        stream=True,
    )
    for event in resp:
        if "choices" in event:
            yield event["choices"][0]["delta"].get("content", "")
        if event["choices"][0]["finish_reason"] == "stop":
            break
    yield "stream-complete"


def make_front_end():
    # copy the sample front end to the current directory
    package_dir = os.path.dirname(os.path.realpath(__file__))
    html = open(os.path.join(package_dir, "index.html")).read()
    with open("./index.html", "w") as f:
        f.write(html)
    log("Front end created at index.html. Access it at /")


def make_dockerfile():
    # copy the sample Dockerfile to the current directory
    package_dir = os.path.dirname(os.path.realpath(__file__))
    dockerfile = open(os.path.join(package_dir, "Dockerfile")).read()
    dockerfile = dockerfile.format(
        openai_api_key=os.environ.get("OPENAI_API_KEY", "your OpenAI API key")
    )
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    log("Dockerfile created.")


def deploy_instructions():
    # print instructions on how to deploy the app
    openai_api_key = os.environ.get("OPENAI_API_KEY", "your OpenAI API key")
    instructions = inspect.cleandoc(
        f"""
        # For fly.io:
        fly launch # answer no to Postgres, Redis and deploying now 
        fly secrets set OPENAI_API_KEY={openai_api_key} 
        fly deploy
        
        # For Google Cloud Run:
        gcloud run deploy --source . --set-env-vars="OPENAI_API_KEY={openai_api_key}"
        """
    )
    print(instructions)


# FastAPI app
app = FastAPI()


class Query(BaseModel):
    question: str
    # sources: Optional[List[str]] = None
    # thread: Optional[List[Dict]] = None


@app.on_event("startup")
def index():
    global index
    index = get_index()


@app.post("/api/ask")
def ask(q: Query):
    return {"response": answer(q.question, index)}


@app.get("/api/stream")
def message_stream(q: Union[str, None] = None):
    response = streaming_answer(q, index)
    return EventSourceResponse(response)


def serve():
    try:
        app.mount("/", StaticFiles(directory="./", html=True), name="static")
    except RuntimeError:
        log("static directory not found, front end will not be available")
    uvicorn.run(app, host=UVICORN_HOST, port=UVICORN_PORT)


def main(action: Optional[str] = typer.Argument(None)):
    if not action or action == "serve":
        serve()
    elif action == "index":
        get_index()
    elif action == "make-front-end":
        make_front_end()
    elif action == "make-dockerfile":
        make_dockerfile()
    elif action == "deploy":
        deploy_instructions()
    else:
        print(f"Unknown action: {action}")


# wrapper for Typer CLI app, aliased to `microllama` in `pyproject.toml`
def cli_wrapper():
    typer.run(main)
