import json
import os
from functools import lru_cache
import sys

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index")
SOURCE_JSON = os.environ.get("SOURCE_JSON", "source.json")
MAX_RELATED_DOCUMENTS = int(os.environ.get("MAX_RELATED_DOCUMENTS"), 5)


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


@lru_cache
def answer(question, index):
    chain = load_qa_with_sources_chain(OpenAI(temperature=0))
    output = chain(
        {
            "input_documents": index.similarity_search(
                question, k=MAX_RELATED_DOCUMENTS
            ),
            "question": question,
        },
        return_only_outputs=False,
    )
    # TODO: extract sources without silly string splitting
    answer = output["output_text"].split("SOURCES:")[0].strip()
    sources = [doc.metadata["source"] for doc in output["input_documents"]]
    return {"answer": answer, "sources": sources}
