import asyncio
import json
import logging
import os
from typing import Any, Dict
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient


class Configuration:
    """读取 .env 与 servers_config.json，用 MCP 工具调用 Neo4j 电影查询。"""

    def __init__(self) -> None:
        load_dotenv()
        self.api_key: str = os.getenv("LLM_API_KEY") or ""
        self.base_url: str | None = os.getenv("BASE_URL")
        self.model: str = os.getenv("MODEL") or "deepseek-chat"
        if not self.api_key:
            raise ValueError("未找到 LLM_API_KEY，请在 .env 中配置")

    @staticmethod
    def load_servers(file_path: str = None) -> Dict[str, Any]:
        if file_path is None:
            file_path = Path(__file__).parent / "neo4j_agent_mcp_config.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})


async def run_chat_loop() -> None:
    """启动仅使用 MCP-neo4j 工具的 Agent 聊天循环。"""
    cfg = Configuration()

    # 兼容 DeepSeek SDK 环境变量
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("LLM_API_KEY", "")
    if cfg.base_url:
        os.environ["DEEPSEEK_API_BASE"] = cfg.base_url

    # LangChain OpenAI/DeepSeek 兼容环境变量
    os.environ["OPENAI_API_KEY"] = cfg.api_key
    if cfg.base_url:
        os.environ["OPENAI_BASE_URL"] = cfg.base_url

    # 1. 加载 MCP 工具
    servers_cfg = Configuration.load_servers()
    mcp_client = MultiServerMCPClient(servers_cfg)

    try:
        tools = await mcp_client.get_tools()
        logging.info(f"已加载 {len(tools)} 个 MCP 工具： {[t.name for t in tools]}")
    except Exception as e:
        logging.error(f"加载 MCP 工具失败: {e}")
        print(f"警告: 无法连接到 MCP 服务器: {e}")
        tools = []

    # 2. 初始化大模型
    llm = init_chat_model(
        model=cfg.model,
        model_provider="deepseek" if "deepseek" in cfg.model else "openai",
    )

    # 3. 构造 Agent（系统提示强调遇到电影图查询要优先调用工具）
    system_prompt = (
        "你是一个电影图数据库智能助理。\n"
        "当用户提出与电影、演员、导演、类型、评分、年份、路径/关联统计相关的问题时，\n"
        "请优先调用 MCP 工具 neo4j_query（自然语言 -> Cypher -> Neo4j 查询）。\n"
        "在仅需常识解释或闲聊时可直接回答。\n"
        "回答使用简体中文，给出关键实体名与数字，并保持简洁。\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 4. CLI 聊天
    print("\n Neo4j MCP Agent 已启动，输入 'quit' 退出")
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
    asyncio.run(run_chat_loop())


