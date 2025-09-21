import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.tools.retriever import create_retriever_tool
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).parent.parent / '.env')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化 MCP 服务器
mcp = FastMCP("RAGServer")

# 全局变量
embeddings = None
FAISS_DB_PATH = "faiss_db"


def initialize_embeddings():
    """初始化嵌入模型"""
    global embeddings
    if embeddings is None:
        logger.info("正在初始化嵌入模型...")
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_api_key:
            logger.error("未找到 DASHSCOPE_API_KEY，请在 .env 中配置")
            raise ValueError("未找到 DASHSCOPE_API_KEY，请在 .env 中配置")
        
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1", 
            dashscope_api_key=dashscope_api_key
        )
        logger.info("嵌入模型初始化成功")
    return embeddings


def check_database_exists() -> bool:
    """检查FAISS数据库是否存在"""
    return os.path.exists(FAISS_DB_PATH) and os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss"))



@mcp.tool()
def query_documents(question: str, top_k: int = 5) -> str:
    """
    基于已处理的文档回答问题
    
    Args:
        question: 用户问题
        top_k: 返回最相关的文档片段数量
        
    Returns:
        str: AI回答或错误信息
    """
    logger.info(f"开始查询文档 - 问题: '{question}', top_k: {top_k}")
    
    try:
        # 检查数据库是否存在
        logger.debug("检查FAISS数据库是否存在")
        if not check_database_exists():
            logger.warning("FAISS数据库不存在，返回提示信息")
            return " 请先上传并处理PDF文件！"
        
        logger.info("FAISS数据库存在，继续查询")
        
        # 初始化嵌入模型
        logger.debug("初始化嵌入模型")
        embeddings = initialize_embeddings()
        
        # 加载FAISS数据库
        logger.info("开始加载FAISS数据库")
        vector_store = FAISS.load_local(
            FAISS_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS数据库加载成功")
        
        # 执行相似性搜索
        logger.info(f"执行相似性搜索，查询: '{question}', 返回数量: {top_k}")
        docs = vector_store.similarity_search(question, k=top_k)
        logger.info(f"搜索完成，找到 {len(docs)} 个相关文档")
        
        if not docs:
            logger.warning("没有找到相关的文档内容")
            return " 没有找到相关的文档内容"
        
        # 格式化搜索结果
        logger.debug("格式化搜索结果")
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 这里可以集成LLM来生成回答
        # 暂时返回检索到的相关内容
        result = f" 基于文档的相关内容：\n\n{context}\n\n"
        result += f" 提示：您可以基于以上内容进一步提问，或者我可以为您生成更详细的回答。"
        
        logger.info(f"查询文档成功完成，返回结果长度: {len(result)} 字符")
        return result
        
    except Exception as e:
        logger.error(f"查询文档时发生错误: {str(e)}", exc_info=True)
        return f" 查询文档时出错: {str(e)}"


@mcp.tool()
def get_database_status() -> str:
    """
    获取向量数据库的状态信息
    
    Returns:
        str: 数据库状态信息
    """
    try:
        if not check_database_exists():
            return " 状态：数据库未创建，请先上传并处理PDF文件"
        
        # 获取数据库文件信息
        db_path = Path(FAISS_DB_PATH)
        index_file = db_path / "index.faiss"
        pkl_file = db_path / "index.pkl"
        
        index_size = index_file.stat().st_size if index_file.exists() else 0
        pkl_size = pkl_file.stat().st_size if pkl_file.exists() else 0
        
        return f" 数据库状态：已就绪\n" \
               f" 索引文件大小：{index_size / 1024:.2f} KB\n" \
               f" 元数据文件大小：{pkl_size / 1024:.2f} KB"
        
    except Exception as e:
        return f" 获取数据库状态时出错: {str(e)}"


@mcp.tool()
def clear_database() -> str:
    """
    清除向量数据库
    
    Returns:
        str: 清除结果信息
    """
    try:
        if os.path.exists(FAISS_DB_PATH):
            shutil.rmtree(FAISS_DB_PATH)
            return " 数据库已清除"
        else:
            return " 数据库不存在，无需清除"
            
    except Exception as e:
        return f" 清除数据库时出错: {str(e)}"


@mcp.tool()
def search_similar_documents(query: str, top_k: int = 3) -> str:
    """
    搜索与查询最相似的文档片段

    Args:
        query: 搜索查询
        top_k: 返回结果数量

    Returns:
        str: 相似文档片段列表
    """
    try:
        if not check_database_exists():
            return " 请先上传并处理PDF文件！"

        # 初始化嵌入模型
        embeddings = initialize_embeddings()

        # 加载FAISS数据库
        vector_store = FAISS.load_local(
            FAISS_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # 执行相似性搜索
        docs = vector_store.similarity_search(query, k=top_k)

        if not docs:
            return " 没有找到相关的文档内容"

        # 格式化结果
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"{i}. {doc.page_content[:200]}...")

        return f" 找到 {len(docs)} 个相关文档片段：\n\n" + "\n\n".join(results)

    except Exception as e:
        return f" 搜索文档时出错: {str(e)}"


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport="stdio")
