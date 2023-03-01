from typing import Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from microllama import answer, get_index

app = FastAPI()
index = get_index()


@app.get("/api/ask")
def ask(q: Union[str, None] = None):
    return {"response": answer(q, index)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
