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

2. Install MicroLlama into a virtual environment:

```bash
pip install microllama
```

3. Get an [OpenAI API key](https://platform.openai.com/account/api-keys) and add
   it to the environment, e.g. `export OPENAI_API_KEY=sk-etc`. Note that
   indexing and querying require OpenAI credits, which
   [aren't free](https://openai.com/api/pricing/).

4. Run your server with `microllama`. If a vector search index doesn't exist,
   it'll be created from your `source.json`, and stored.

5. Query your documents at
   [/api/ask?your question](http://127.0.0.1:8000/api/ask?your%20question).

6. Microllama includes an optional web front-end, which is generated with
   `microllama make-front-end`. This command creates a single `index.html` file
   which you can edit. It's served at [/](http://127.0.0.1:8000/).

## Configuration

Microllama is configured through environment variables, with the following
defaults:

- `OPENAI_API_KEY`: required
- `FAISS_INDEX_PATH`: "faiss_index"
- `SOURCE_JSON`: "source.json"
- `MAX_RELATED_DOCUMENTS`: "5"
- `EXTRA_CONTEXT`: "Answer in no more than three sentences. If the answer is not
  included in the context, say 'Sorry, this is no answer for this in my
  sources.'."
- `UVICORN_HOST`: "0.0.0.0"
- `UVICORN_PORT`: "8080"

## Deploying your API

Create a Dockerfile with `microllama make-dockerfile`. Then:

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

For Cloud Run and other serverless platforms you should generate the FAISS index
at container build time, to reduce startup time. See the two commented lines in
`Dockerfile`.

You can also generate these commands with `microllama deploy`.

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
