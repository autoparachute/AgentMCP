#!/usr/bin/env python3
"""
PDF处理脚本 - 上传并处理PDF文件，创建向量数据库
独立于MCP服务器的PDF处理工具
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).parent / '.env')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
FAISS_DB_PATH = "faiss_db"


def initialize_embeddings():
    """初始化嵌入模型"""
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


def upload_and_process_pdf(pdf_path: str) -> str:
    """
    上传并处理PDF文件，创建向量数据库
    
    Args:
        pdf_path: PDF文件的路径
        
    Returns:
        str: 处理结果信息
    """
    try:
        logger.info(f"开始处理PDF文件: {pdf_path}")
        
        # 步骤1: 检查文件是否存在
        logger.info("步骤1: 检查PDF文件是否存在")
        if not os.path.exists(pdf_path):
            logger.error(f"文件不存在: {pdf_path}")
            return f"文件不存在: {pdf_path}"
        
        logger.info(f"文件存在，路径: {pdf_path}")
        
        # 步骤2: 初始化嵌入模型
        logger.info("步骤2: 初始化嵌入模型")
        embeddings = initialize_embeddings()
        logger.info("嵌入模型初始化完成")
        
        # 步骤3: 读取PDF内容
        logger.info("步骤3: 读取PDF内容")
        raw_text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF总页数: {total_pages}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                raw_text += page_text
                logger.debug(f"已处理第 {page_num} 页，提取字符数: {len(page_text)}")
        
        logger.info(f"PDF文本提取完成，总字符数: {len(raw_text)}")
        
        if not raw_text.strip():
            logger.error("无法从PDF中提取文本，请检查文件是否有效")
            return "无法从PDF中提取文本，请检查文件是否有效"
        
        # 步骤4: 分割文本
        logger.info("步骤4: 分割文本")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_text(raw_text)
        logger.info(f"文本分割完成，生成 {len(text_chunks)} 个文本片段")
        
        # 步骤5: 创建向量数据库
        logger.info("步骤5: 创建向量数据库")
        logger.info("正在生成文本嵌入向量...")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        logger.info("向量数据库创建完成")
        
        # 步骤6: 保存向量数据库到本地
        logger.info("步骤6: 保存向量数据库到本地")
        vector_store.save_local(FAISS_DB_PATH)
        logger.info(f"向量数据库已保存到: {FAISS_DB_PATH}")
        
        success_msg = f"PDF处理完成！已创建向量数据库，包含 {len(text_chunks)} 个文本片段"
        logger.info(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"处理PDF时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDF处理脚本 - 上传并处理PDF文件，创建向量数据库')
    parser.add_argument('pdf_path', help='PDF文件的路径')
    parser.add_argument('--output-dir', '-o', default='faiss_db', 
                       help='向量数据库输出目录 (默认: faiss_db)')
    
    args = parser.parse_args()
    
    # 设置输出目录
    global FAISS_DB_PATH
    FAISS_DB_PATH = args.output_dir
    
    # 处理PDF文件路径，支持引号包围的路径
    pdf_path = args.pdf_path.strip('"\'')
    
    # 处理PDF文件
    result = upload_and_process_pdf(pdf_path)
    print(result)
    
    # 根据结果设置退出码
    if "处理完成" in result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
