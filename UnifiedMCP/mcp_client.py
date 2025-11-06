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
    """读取 .env 与 servers_config.json"""

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
            file_path = Path(__file__).parent / "servers_config.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})


async def run_chat_loop() -> None:
    """启动 MCP-Agent 聊天循环"""
    cfg = Configuration()

    # 兼容 DeepSeek SDK 环境变量
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("LLM_API_KEY", "")
    if cfg.base_url:
        os.environ["DEEPSEEK_API_BASE"] = cfg.base_url

    # LangChain OpenAI/DeepSeek 兼容环境变量
    os.environ["OPENAI_API_KEY"] = cfg.api_key
    if cfg.base_url:
        os.environ["OPENAI_BASE_URL"] = cfg.base_url

    # 1️. 连接多台 MCP 服务器
    servers_cfg = Configuration.load_servers()
    mcp_client = MultiServerMCPClient(servers_cfg)
    
    # 初始化工具列表
    tools = []
    
    try:
        tools = await mcp_client.get_tools()
        logging.info(f"已加载 {len(tools)} 个 MCP 工具： {[t.name for t in tools]}")
    except Exception as e:
        logging.error(f"加载 MCP 工具失败: {e}")
        print(f"警告: 无法连接到 MCP 服务器，将使用基础功能。错误: {e}")
        # 继续运行，但不使用 MCP 工具
        tools = []

    # 2️. 初始化大模型
    llm = init_chat_model(
        model=cfg.model,
        model_provider="deepseek" if "deepseek" in cfg.model else "openai",
    )

    # 3️. 构造 LangChain Agent
    prompt_md_path = Path(__file__).with_name("promptTemplate.md")
    system_prompt = prompt_md_path.read_text(encoding="utf-8").strip()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 4️. CLI 聊天
    print("\n MCP Agent 已启动，输入 'quit' 退出")
    print(" 可用工具:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    print()
    
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            result = await agent_executor.ainvoke({"input": user_input})
            print(f"\n AI: {result['output']}")
        except Exception as exc:
            print(f"\n 出错: {exc}")

    # 5️. 清理
    print("\n 资源已清理，Bye!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_chat_loop())
