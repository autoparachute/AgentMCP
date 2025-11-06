from fastmcp import FastMCP
from typing import Annotated


mcp = FastMCP("CalculateServer")


@mcp.tool()
def add(
	left: Annotated[float, "左操作数"],
	right: Annotated[float, "右操作数"],
) -> float:
	"""返回两个数的和。"""

	return left + right


@mcp.tool()
def subtract(
	left: Annotated[float, "左操作数"],
	right: Annotated[float, "右操作数"],
) -> float:
	"""返回两个数的差（left - right）。"""

	return left - right


@mcp.tool()
def multiply(
	left: Annotated[float, "左操作数"],
	right: Annotated[float, "右操作数"],
) -> float:
	"""返回两个数的积。"""

	return left * right


@mcp.tool()
def divide(
	left: Annotated[float, "左操作数"],
	right: Annotated[float, "右操作数"],
) -> float:
	"""返回两个数的商（left / right）。当右操作数为0时抛出错误。"""

	if right == 0:
		raise ValueError("Division by zero is not allowed")
	return left / right


if __name__ == "__main__":
	mcp.run()
