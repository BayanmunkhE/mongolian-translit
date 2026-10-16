"""Annotation UI for the real test set: Latin message on the left, Cyrillic reference on the right.

    uvicorn annotate.main:app --port 8000        # http://localhost:8000
    docker compose up annotate

Items are stored in SQLite (see annotate/db.py). Export the finished ones and add
them with tools/real_test.py, the only way into data/real_test.jsonl.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annotate import db  # noqa: E402
from baselines import get_baseline  # noqa: E402

STATIC = Path(__file__).parent / "static"
naive = get_baseline("naive")  # optional hint only; references are always typed/checked by hand

app = FastAPI(title="Real test annotation", version="0.1.0")


def get_con():
    con = db.connect()
    try:
        yield con
    finally:
        con.close()


class Save(BaseModel):
    tgt: str = Field("", max_length=1000)
    status: Literal["todo", "done", "skip"]
    note: str = Field("", max_length=500)


class NewItem(BaseModel):
    src: str = Field(..., min_length=1, max_length=1000)
    source: str = Field("", max_length=100)
    note: str = Field("", max_length=500)


def found(item):
    if item is None:
        raise HTTPException(404, "item not found")
    return item


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
def stats(con=Depends(get_con)):
    return db.stats(con)


@app.get("/api/todo/next")
def next_todo(after: int = 0, con=Depends(get_con)):
    return {"item": db.next_todo(con, after)}  # null when everything is annotated


@app.get("/api/items/{item_id}")
def get_item(item_id: int, con=Depends(get_con)):
    return found(db.get(con, item_id))


@app.get("/api/items/{item_id}/prev")
def prev_item(item_id: int, con=Depends(get_con)):
    return {"item": db.neighbor(con, item_id, -1)}


@app.get("/api/items/{item_id}/next")
def next_item(item_id: int, con=Depends(get_con)):
    return {"item": db.neighbor(con, item_id, 1)}


@app.put("/api/items/{item_id}")
def save_item(item_id: int, body: Save, con=Depends(get_con)):
    found(db.get(con, item_id))
    try:
        item, warnings = db.save(con, item_id, body.tgt, body.status, body.note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"item": item, "warnings": warnings}


@app.post("/api/items", status_code=201)
def add_item(body: NewItem, con=Depends(get_con)):
    try:
        item_id = db.add(con, body.src, body.source, body.note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if item_id is None:
        raise HTTPException(409, "this message is already in the database")
    return db.get(con, item_id)


@app.get("/api/hint")
def hint(src: str):
    return {"guess": naive(src[:1000])}


@app.get("/api/export.csv")
def export(con=Depends(get_con)):
    buf = io.StringIO()
    db.write_csv(con, buf)
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",  # BOM for Excel
                    headers={"Content-Disposition": 'attachment; filename="real_test_todo.csv"'})
