# Micro Llama

The smallest possible LLM API.

## Usage

1. Combine your source documents into a single JSON file called `source.json`.
   It should look like this:

```json
[
    {
        "source": "Reference to the source of your content. This could be a URL or a title or a filename",
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

3. Get an OpenAI API key and add it to the environment, e.g.
   `export OPENAI_API_KEY=sk-etc`. Note that indexing and querying require
   OpenAI credits, which aren't free.

4. Run your server with `uvicorn serve:app`. If the search index doesn't exist,
   it'll be created and stored.

5. Query your documents at `/api/ask?your question`.

## Deploying your API

### On Fly.io

```bash
fly launch # answer no to Postgres, Redis and deploying now 
fly secrets set OPENAI_API_KEY=sk-etc 
fly deploy
```

## Inspiration

- [Langchain](https://langchain.readthedocs.io/en/latest/)
- Simon Willison's
  [blog post](https://simonwillison.net/2023/Jan/13/semantic-search-answers/),
  [datasette-openai](https://datasette.io/plugins/datasette-openai) and
  [datasette-faiss](https://datasette.io/plugins/datasette-faiss).
- [FastAPI](https://fastapi.tiangolo.com)
- [GPT Index](https://gpt-index.readthedocs.io/en/latest/)
- [Dagster blog post](https://dagster.io/blog/chatgpt-langchain)
