"""The smallest possible LLM API"""

__version__ = "0.4.10"

import inspect
import json
import os
import sys
from functools import lru_cache
from typing import Optional, Union

import llm
import typer
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
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

    # Prepare messages for llm.chat()
    # Note: llm expects conversations to start with 'user', not 'system' usually.
    # We'll construct a system prompt first, then the user question.
    system_prompt_parts = [
        "You are a helpful assistant.",
        "Use the following context to answer the questions.",
        prompt_context,
        "Don't mention the context in your answer.",
    ]
    if extra_context:
        system_prompt_parts.append(inspect.cleandoc(extra_context))

    system_prompt = "\\n".join(system_prompt_parts)

    # Get the model instance from llm
    model = llm.get_model(MODEL)

    # Call the model using llm's chat method
    # Note: llm.chat() takes system prompt separately
    response = model.chat(
        messages=[{"role": "user", "content": question}], system=system_prompt
    )

    # Extract the answer from the llm response object
    ans = response.text()

    # Debug output structure remains similar, but includes the llm response object maybe?
    # For simplicity, let's keep the original debug structure for now.
    # If needed, we can log `response.prompt_messages()` etc. separately.
    if DEBUG:
        # Construct prompt_messages similar to old format for debugging if needed
        debug_prompt_messages = [{"role": "system", "content": system_prompt}]
        debug_prompt_messages.append({"role": "user", "content": question})
        return {
            "answer": ans,
            "sources": sources,
            "prompt_messages": debug_prompt_messages,  # Show the combined system prompt
        }
    return {"answer": ans, "sources": sources}


def streaming_answer(question, index, extra_context=EXTRA_CONTEXT):
    similar_docs = find_similar_docs(question, index)
    sources = {
        (doc.metadata["source"], doc.metadata.get("url")) for doc in similar_docs
    }
    yield f"SOURCES::{json.dumps(list(sources))}"
    prompt_context = " ".join([doc.page_content for doc in similar_docs])

    # Prepare system prompt like in the non-streaming version
    system_prompt_parts = [
        "You are a helpful assistant.",
        "Use the following context to answer the questions.",
        prompt_context,
        "Don't mention the context in your answer.",
    ]
    if extra_context:
        system_prompt_parts.append(inspect.cleandoc(extra_context))

    system_prompt = "\\n".join(system_prompt_parts)

    # Get model instance
    model = llm.get_model(MODEL)

    if DEBUG:
        # Construct prompt_messages similar to old format for debugging
        debug_prompt_messages = [{"role": "system", "content": system_prompt}]
        debug_prompt_messages.append({"role": "user", "content": question})
        yield f"PROMPT::{json.dumps(debug_prompt_messages)}"

    # Use llm's streaming chat
    response_stream = model.chat(
        messages=[{"role": "user", "content": question}],
        system=system_prompt,
        stream=True,
    )

    # Iterate through the response stream chunks
    for chunk in response_stream:
        yield chunk  # llm yields text chunks directly

    yield "stream-complete"  # Keep our custom end-of-stream marker


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
    # Note: OPENAI_API_KEY is always required for embeddings, regardless of the chat model used.
    openai_api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
    # You might need other keys depending on the MODEL, e.g.:
    # gemini_api_key = os.environ.get("GEMINI_API_KEY", "your_gemini_api_key")
    # anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "your_anthropic_api_key")

    instructions = inspect.cleandoc(
        f"""
        # For fly.io:
        fly launch # answer no to Postgres, Redis and deploying now 
        # Set keys as secrets. OPENAI_API_KEY is always needed for embeddings.
        fly secrets set OPENAI_API_KEY={openai_api_key} 
        # Set other keys if your chosen MODEL requires them, e.g.:
        # fly secrets set GEMINI_API_KEY=your_gemini_key
        fly deploy
        
        # For Google Cloud Run:
        # Set keys as env vars. OPENAI_API_KEY is always needed for embeddings.
        # Add other keys if MODEL requires them, e.g., GEMINI_API_KEY=your_gemini_key
        # Ensure MODEL env var is also set if not using the default.
        gcloud run deploy --source . --set-env-vars="OPENAI_API_KEY={openai_api_key},MODEL=gpt-3.5-turbo"
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
