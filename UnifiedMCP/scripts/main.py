from pathlib import Path
import sys
from fastapi import FastAPI
import uvicorn

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.servers.rank_server.server import app as rank_app
from src.servers.calculate_server.server import app as calculate_app


app = FastAPI(
    title="MCP Platform Unified Gateway",
    description="MCP 服务统一入口",
    version="1.0.0",
)


# Mount sub-apps under distinct path prefixes
app.mount(settings.rank_prefix, rank_app)
app.mount(settings.calculate_prefix, calculate_app)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)


