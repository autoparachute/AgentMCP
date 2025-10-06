from fastmcp import FastMCP
from typing import Annotated
import os

mcp = FastMCP("PromptServer")

# 定义提示词模板目录
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(template_name: str) -> str:
    """从文件加载提示词模板"""
    file_path = os.path.join(PROMPT_DIR, template_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt template not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# 用于生成前端开发提示词的工具
@mcp.tool()
def generate_frontend_prompt(
    files_to_modify: Annotated[str, "需要修改的文件"],
    requirements: Annotated[str, "需求说明"],
    references: Annotated[str, "功能参考"],
    special_rules: Annotated[str, "特殊规则"],
) -> str:
    """基于结构化输入生成详细的前端开发提示词。"""
    template = load_prompt("frontend_prompt.txt")
    return template.format(
        files_to_modify=files_to_modify,
        requirements=requirements,
        references=references,
        special_rules=special_rules
    )

# 用于生成系统功能分析提示词的工具
@mcp.tool()
def generate_system_analysis_prompt(
    vue_file_path: Annotated[str, "Vue文件路径"]
) -> str:
    """基于Vue文件路径生成系统功能分析提示词。"""
    template = load_prompt("system_analysis_prompt.txt")
    return template.format(vue_file_path=vue_file_path)

# 用于生成coding需求提示词的工具
@mcp.tool()
def generate_coding_requirement_prompt(
    feishu_doc_link: Annotated[str, "飞书需求文档链接"]
) -> str:
    """基于飞书需求文档链接生成coding需求提示词。"""
    template = load_prompt("coding_requirement_prompt.txt")
    return template.format(feishu_doc_link=feishu_doc_link)


if __name__ == "__main__":
    mcp.run()