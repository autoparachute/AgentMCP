from fastmcp import FastMCP
from typing import Annotated


mcp = FastMCP("RankServer")


def _quicksort(arr: list[float]) -> list[float]:
	if len(arr) <= 1:
		return arr
	pivot = arr[len(arr) // 2]
	left = [x for x in arr if x < pivot]
	middle = [x for x in arr if x == pivot]
	right = [x for x in arr if x > pivot]
	return _quicksort(left) + middle + _quicksort(right)


def _merge(left: list[float], right: list[float]) -> list[float]:
	merged: list[float] = []
	i = 0
	j = 0
	while i < len(left) and j < len(right):
		if left[i] <= right[j]:
			merged.append(left[i])
			i += 1
		else:
			merged.append(right[j])
			j += 1
	if i < len(left):
		merged.extend(left[i:])
	if j < len(right):
		merged.extend(right[j:])
	return merged


def _mergesort(arr: list[float]) -> list[float]:
	if len(arr) <= 1:
		return arr
	mid = len(arr) // 2
	left = _mergesort(arr[:mid])
	right = _mergesort(arr[mid:])
	return _merge(left, right)


@mcp.tool()
def quicksort(
	numbers: Annotated[list[float], "待排序的数字列表"],
) -> list[float]:
	"""使用快速排序返回升序排序后的列表。"""

	return _quicksort(numbers)


@mcp.tool()
def mergesort(
	numbers: Annotated[list[float], "待排序的数字列表"],
) -> list[float]:
	"""使用归并排序返回升序排序后的列表。"""

	return _mergesort(numbers)


if __name__ == "__main__":
	mcp.run()
