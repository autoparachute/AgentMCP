#!/usr/bin/env python3
"""
Neo4j 图数据库连接和操作
"""

import os
import csv
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 使用 DeepSeek
# 确保在项目根目录或同级 .env 中配置 DEEPSEEK_API_KEY

# 定义 Neo4j 凭证
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "neo4jroot"

def main() -> None:
    graph = Neo4jGraph()
    # 读取与本文件同目录下的本地 CSV 文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "movies_small.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到本地文件: {csv_path}")

    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "movieId": row.get("movieId", ""),
                "title": row.get("title", ""),
                "released": row.get("released", ""),
                "imdbRating": row.get("imdbRating", ""),
                "director": row.get("director", ""),
                "actors": row.get("actors", ""),
                "genres": row.get("genres", ""),
            })

    movies_query = """
UNWIND $rows AS row
MERGE (m:Movie {id: row.movieId})
SET m.released = CASE WHEN row.released = "" THEN NULL ELSE date(row.released) END,
    m.title = row.title,
    m.imdbRating = CASE WHEN row.imdbRating = "" THEN NULL ELSE toFloat(row.imdbRating) END
WITH m, row
FOREACH (director IN (CASE WHEN row.director = "" THEN [] ELSE split(row.director, '|') END) |
    MERGE (p:Person {name: trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor IN (CASE WHEN row.actors = "" THEN [] ELSE split(row.actors, '|') END) |
    MERGE (p2:Person {name: trim(actor)})
    MERGE (p2)-[:ACTED_IN]->(m))
FOREACH (genre IN (CASE WHEN row.genres = "" THEN [] ELSE split(row.genres, '|') END) |
    MERGE (g:Genre {name: trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""
    graph.query(movies_query, params={"rows": rows})
    
    # 刷新并打印图数据库的 schema，便于后续 LLM 生成 Cypher 使用
    graph.refresh_schema()
    print(graph.schema)

    # 使用 GraphCypherQAChain 进行自然语言到 Cypher 的问答
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com")

    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        base_url=base_url,
        api_key=api_key,
    )
    chain = GraphCypherQAChain.from_llm(
        graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True
    )
    response = chain.invoke({"query": "What was the cast of the Casino?"})
    print(response)


if __name__ == "__main__":
    main()
