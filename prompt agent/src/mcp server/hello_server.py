from mcp.server.fastmcp import FastMCP


# 初始化 MCP 服务器
mcp = FastMCP("HelloServer")


@mcp.tool()
async def hello(name: str) -> str:
    """
    返回一个简单的问候语。
    Args:
        name: 要问候的名称
    Returns:
        str: 问候语
    """
    return f"Hello, {name}!"


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport="stdio")


