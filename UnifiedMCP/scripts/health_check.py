import asyncio
from typing import Any

import httpx


async def check(url: str) -> tuple[str, int, Any]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return url, resp.status_code, resp.json()
    except Exception as exc:  # noqa: BLE001
        return url, 0, {"error": str(exc)}


async def main() -> None:
    base = "http://127.0.0.1:8000"
    urls = [
        f"{base}/health",
        f"{base}/rank/",
        f"{base}/rank/quicksort",
        f"{base}/calculate/",
        f"{base}/calculate/add",
    ]
    results = await asyncio.gather(*(check(u) for u in urls))
    for url, status, body in results:
        print(f"{url} -> {status} :: {body}")


if __name__ == "__main__":
    asyncio.run(main())


