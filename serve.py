from typing import Union

from fastapi import FastAPI

from microllama import answer, get_index

app = FastAPI()
index = get_index()


@app.get("/api/ask/")
def ask(q: Union[str, None] = None):
    return {"answer": answer(q, index)}
