import asyncio
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI


class Configuration:
    """加载 .env 并提供 LLM 与 Neo4j 配置。"""

    def __init__(self) -> None:
        load_dotenv()
        # LLM 配置
        self.api_key: str = os.getenv("LLM_API_KEY") or ""
        self.base_url: str | None = os.getenv("BASE_URL")
        self.model: str = os.getenv("MODEL") or "deepseek-chat"
        if not self.api_key:
            raise ValueError("未找到 LLM_API_KEY，请在 .env 中配置")

        # Neo4j 配置（可在 .env 中覆盖）
        self.neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4jroot")


def build_neo4j_chain(cfg: Configuration) -> GraphCypherQAChain:
    """构建基于 Neo4j 的自然语言到 Cypher 问答链。"""
    # 设置环境变量（langchain_neo4j 会从环境变量读取配置）
    os.environ["NEO4J_URI"] = cfg.neo4j_uri
    os.environ["NEO4J_USERNAME"] = cfg.neo4j_user
    os.environ["NEO4J_PASSWORD"] = cfg.neo4j_password

    graph = Neo4jGraph()
    graph.refresh_schema()

    # 基于 OpenAI 兼容接口初始化 DeepSeek/OpenAI 聊天模型
    llm = ChatOpenAI(
        model=cfg.model,
        temperature=0,
        base_url=cfg.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_API_BASE"),
        api_key=cfg.api_key,
    )

    chain = GraphCypherQAChain.from_llm(
        graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True
    )
    return chain


def create_neo4j_tool(chain: GraphCypherQAChain) -> Tool:
    """创建一个 LangChain Tool，使 Agent 能够调用 Neo4j QA 链。"""

    def _run(question: str) -> str:
        try:
            result: Dict[str, Any] = chain.invoke({"query": question})
            # GraphCypherQAChain 可能返回 dict 或字符串，这里做统一处理
            if isinstance(result, dict):
                # 常见返回键可能为 'result' 或 'text'
                return str(result.get("result") or result.get("text") or result)
            return str(result)
        except Exception as exc:
            return f"Neo4j 查询失败: {exc}"

    async def _arun(question: str) -> str:
        try:
            result: Dict[str, Any] = await chain.ainvoke({"query": question})
            if isinstance(result, dict):
                return str(result.get("result") or result.get("text") or result)
            return str(result)
        except Exception as exc:
            return f"Neo4j 查询失败: {exc}"

    return Tool(
        name="neo4j_query",
        description=(
            "查询 Neo4j 图数据库的工具。输入自然语言问题，工具会自动生成 Cypher 并查询图数据库。"
        ),
        func=_run,
        coroutine=_arun,
    )


async def run_neo4j_agent_chat() -> None:
    """启动仅包含 Neo4j 查询工具的 Agent 聊天循环。"""
    cfg = Configuration()

    # 兼容 DeepSeek SDK 环境变量
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("LLM_API_KEY", "")
    if cfg.base_url:
        os.environ["DEEPSEEK_API_BASE"] = cfg.base_url

    # LangChain OpenAI/DeepSeek 兼容环境变量
    os.environ["OPENAI_API_KEY"] = cfg.api_key
    if cfg.base_url:
        os.environ["OPENAI_BASE_URL"] = cfg.base_url

    # 构建 Neo4j QA 链与工具
    chain = build_neo4j_chain(cfg)
    neo4j_tool = create_neo4j_tool(chain)

    # 构造 Agent
    system_prompt = (
        "你是一个电影图数据库智能助理。\n"
        "图结构说明：\n"
        "- 节点标签：Movie、Person、Genre。\n"
        "- 关系类型：(:Person)-[:DIRECTED]->(:Movie)，(:Person)-[:ACTED_IN]->(:Movie)，(:Movie)-[:IN_GENRE]->(:Genre)。\n"
        "- 关键属性：Movie(title, released, imdbRating, id)，Person(name)，Genre(name)。\n"
        "使用指南：\n"
        "- 当用户提出与电影、演员、导演、类型、评分、年份、路径/关联统计相关的问题时，优先调用工具 neo4j_query 自动生成 Cypher 并查询。\n"
        "- 在仅需通识解释或闲聊时再直接回答。\n"
        "- 回答使用简体中文，给出关键实体名与数字；必要时列出前若干条并保持简洁。\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = init_chat_model(
        model=cfg.model,
        model_provider="deepseek" if "deepseek" in cfg.model else "openai",
    )

    agent = create_openai_tools_agent(llm, [neo4j_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[neo4j_tool], verbose=True)

    print("\n Neo4j Agent 已启动，输入 'quit' 退出")
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            result = await agent_executor.ainvoke({"input": user_input})
            print(f"\nAI: {result['output']}")
        except Exception as exc:
            print(f"\n 出错: {exc}")

    print(" 资源已清理，Bye!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_neo4j_agent_chat())


