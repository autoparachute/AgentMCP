from mcp.server.fastmcp import FastMCP


# 初始化 MCP 服务器
mcp = FastMCP("OperationServer")


@mcp.tool()
def add(a: int, b: int) -> int:
    """
    两数相加。
    Args:
        a: 第一个整数
        b: 第二个整数
    Returns:
        int: a + b 的结果
    """
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """
    两数相减。
    Args:
        a: 被减数
        b: 减数
    Returns:
        int: a - b 的结果
    """
    return a - b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """
    两数相乘。
    Args:
        a: 第一个整数
        b: 第二个整数
    Returns:
        int: a * b 的结果
    """
    return a * b


@mcp.tool()
def divide(a: int, b: int) -> float:
    """
    两数相除。
    Args:
        a: 被除数
        b: 除数
    Returns:
        float: a / b 的结果
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport="stdio")
