from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Reuse the independent MCP service's tool functions without modifying it
from src.servers.calculate_server import add, subtract, multiply, divide


class BinaryOpBody(BaseModel):
    left: float
    right: float


app = FastAPI(
    title="Calculate Server",
    description="Calculation Service (HTTP wrapper over MCP tools)",
    version="1.0.0",
)


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {"service": "calculate", "status": "ok"}


@app.post("/add", tags=["ops"])
async def add_endpoint(body: BinaryOpBody) -> dict:
    return {"result": add(body.left, body.right)}


@app.post("/subtract", tags=["ops"])
async def subtract_endpoint(body: BinaryOpBody) -> dict:
    return {"result": subtract(body.left, body.right)}


@app.post("/multiply", tags=["ops"])
async def multiply_endpoint(body: BinaryOpBody) -> dict:
    return {"result": multiply(body.left, body.right)}


@app.post("/divide", tags=["ops"])
async def divide_endpoint(body: BinaryOpBody) -> dict:
    try:
        return {"result": divide(body.left, body.right)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


