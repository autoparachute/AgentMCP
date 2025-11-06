from fastapi import FastAPI
from pydantic import BaseModel

# Reuse the independent MCP service's tool functions without modifying it
from src.servers.rank_server import quicksort, mergesort


class NumbersBody(BaseModel):
    numbers: list[float]


app = FastAPI(
    title="Rank Server",
    description="Ranking Service (HTTP wrapper over MCP tools)",
    version="1.0.0",
)


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {"service": "rank", "status": "ok"}


@app.post("/quicksort", tags=["sort"])
async def quicksort_endpoint(body: NumbersBody) -> dict:
    return {"result": quicksort(body.numbers)}


@app.post("/mergesort", tags=["sort"])
async def mergesort_endpoint(body: NumbersBody) -> dict:
    return {"result": mergesort(body.numbers)}


