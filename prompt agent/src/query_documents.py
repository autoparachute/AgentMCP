#!/usr/bin/env python3
"""
独立的文档查询脚本
基于FAISS向量数据库进行文档检索和查询
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(project_root / '.env')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
            return "请先上传并处理PDF文件！"
        
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
            return "没有找到相关的文档内容"
        
        # 格式化搜索结果
        logger.debug("格式化搜索结果")
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 这里可以集成LLM来生成回答
        # 暂时返回检索到的相关内容
        result = f"基于文档的相关内容：\n\n{context}\n\n"
        result += f"提示：您可以基于以上内容进一步提问，或者我可以为您生成更详细的回答。"
        
        logger.info(f"查询文档成功完成，返回结果长度: {len(result)} 字符")
        return result
        
    except Exception as e:
        logger.error(f"查询文档时发生错误: {str(e)}", exc_info=True)
        return f"查询文档时出错: {str(e)}"


def get_database_status() -> str:
    """
    获取向量数据库的状态信息
    
    Returns:
        str: 数据库状态信息
    """
    try:
        if not check_database_exists():
            return "状态：数据库未创建，请先上传并处理PDF文件"
        
        # 获取数据库文件信息
        db_path = Path(FAISS_DB_PATH)
        index_file = db_path / "index.faiss"
        pkl_file = db_path / "index.pkl"
        
        index_size = index_file.stat().st_size if index_file.exists() else 0
        pkl_size = pkl_file.stat().st_size if pkl_file.exists() else 0
        
        return f"数据库状态：已就绪\n" \
               f"索引文件大小：{index_size / 1024:.2f} KB\n" \
               f"元数据文件大小：{pkl_size / 1024:.2f} KB"
        
    except Exception as e:
        return f"获取数据库状态时出错: {str(e)}"


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description="基于FAISS向量数据库的文档查询工具")
    parser.add_argument("question", nargs="?", help="要查询的问题")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="返回最相关的文档片段数量 (默认: 5)")
    parser.add_argument("-s", "--status", action="store_true", help="显示数据库状态")
    parser.add_argument("-i", "--interactive", action="store_true", help="进入交互模式")
    
    args = parser.parse_args()
    
    # 显示数据库状态
    if args.status:
        print(get_database_status())
        return
    
    # 交互模式
    if args.interactive:
        print("文档查询工具 - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'status' 查看数据库状态")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                elif question.lower() == 'status':
                    print(get_database_status())
                    continue
                elif not question:
                    print("请输入有效的问题")
                    continue
                
                # 执行查询
                result = query_documents(question, args.top_k)
                print(f"\n{result}")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")
        
        return
    
    # 单次查询模式
    if args.question:
        result = query_documents(args.question, args.top_k)
        print(result)
    else:
        # 如果没有提供问题，显示帮助信息
        parser.print_help()


if __name__ == "__main__":
    main()
