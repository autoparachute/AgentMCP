import os
from typing import Any, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain


mcp = FastMCP("Neo4jMovieServer")


def _build_chain() -> GraphCypherQAChain:
    """Initialize Neo4j GraphCypherQAChain from environment variables."""
    load_dotenv()

    # Neo4j connection (env defaults are same as demo)
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4jroot")

    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_user
    os.environ["NEO4J_PASSWORD"] = neo4j_password

    # LLM via OpenAI-compatible endpoint (DeepSeek supported via base_url)
    api_key = os.getenv("LLM_API_KEY") or ""
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_API_BASE")
    model = os.getenv("MODEL") or "deepseek-chat"

    if not api_key:
        raise ValueError("未找到 LLM_API_KEY，请在 .env 中配置")

    graph = Neo4jGraph()
    graph.refresh_schema()

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        base_url=base_url,
        api_key=api_key,
    )

    chain = GraphCypherQAChain.from_llm(
        graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True
    )
    return chain


# Lazy singleton for the chain so the server starts fast
_CHAIN: GraphCypherQAChain | None = None


def _get_chain() -> GraphCypherQAChain:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = _build_chain()
    return _CHAIN


@mcp.tool()
def neo4j_query(question: str) -> str:
    """
    使用自然语言查询 Neo4j 电影图数据库。

    Args:
        
        question: 中文或英文问题，例如 "1990 年代的高分电影有哪些？"

    Returns:
        查询结果的字符串表示。
    """
    try:
        chain = _get_chain()
        result: Dict[str, Any] = chain.invoke({"query": question})
        if isinstance(result, dict):
            return str(result.get("result") or result.get("text") or result)
        return str(result)
    except Exception as exc:
        return f"Neo4j 查询失败: {exc}"


if __name__ == "__main__":
    # Run as stdio MCP server
    mcp.run(transport="stdio")


