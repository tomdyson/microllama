![llama-small](https://user-images.githubusercontent.com/15543/221690917-1ca1dcb7-4a88-4ef8-842c-98268e3f4e63.jpg)

# MicroLlama

The smallest possible LLM API. Build a question and answer interface to your own
content in a few minutes. Uses
[OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings),
[gpt-3.5](https://platform.openai.com/docs/guides/chat) and
[Faiss](https://faiss.ai), via
[Langchain](https://langchain.readthedocs.io/en/latest/).

## Usage

1. Combine your source documents into a single JSON file called `source.json`.
   It should look like this:

```json
[
    {
        "source": "Reference to the source of your content. Typically a title.",
        "url": "URL for your source. This key is optional.",
        "content": "Your content as a single string. If there's a title or summary, put these first, separated by new lines."
    }, 
    ...
]
```

See `example.source.json` for an example.

2. Install dependencies:

```bash
pip install langchain faiss-cpu openai fastapi "uvicorn[standard]"
```

3. Get an [OpenAI API key](https://platform.openai.com/account/api-keys) and add
   it to the environment, e.g. `export OPENAI_API_KEY=sk-etc`. Note that
   indexing and querying require OpenAI credits, which
   [aren't free](https://openai.com/api/pricing/).

4. Run your server with `uvicorn serve:app`. If the search index doesn't exist,
   it'll be created and stored.

5. Query your documents at
   [/api/ask?your question](http://127.0.0.1:8000/api/ask?your%20question) or
   use the simple front-end at [/](http://127.0.0.1:8000/)

## Deploying your API

### On Fly.io

Sign up for a [Fly.io](https://fly.io) account and
[install flyctl](https://fly.io/docs/hands-on/install-flyctl/). Then:

```bash
fly launch # answer no to Postgres, Redis and deploying now 
fly secrets set OPENAI_API_KEY=sk-etc 
fly deploy
```

### On Google Cloud Run

```bash
gcloud run deploy --source . --set-env-vars="OPENAI_API_KEY=sk-etc"
```

For Cloud Run and other serverless platforms you should probably generate the
FAISS index at container build time, to reduce cold starts. See the two
commented lines in `Dockerfile`.

## Based on

- [Langchain](https://langchain.readthedocs.io/en/latest/)
- Simon Willison's
  [blog post](https://simonwillison.net/2023/Jan/13/semantic-search-answers/),
  [datasette-openai](https://datasette.io/plugins/datasette-openai) and
  [datasette-faiss](https://datasette.io/plugins/datasette-faiss).
- [FastAPI](https://fastapi.tiangolo.com)
- [GPT Index](https://gpt-index.readthedocs.io/en/latest/)
- [Dagster blog post](https://dagster.io/blog/chatgpt-langchain)

## TODO

- [ ] Use splitting which generates more meaningful fragments, e.g.
      text_splitter =
      `SpacyTextSplitter(chunk_size=700, chunk_overlap=200, separator=" ")`
