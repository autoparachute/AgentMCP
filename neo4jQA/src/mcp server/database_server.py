import mysql.connector
from mysql.connector import Error
from mcp.server.fastmcp import FastMCP


# 初始化 MCP 服务器
mcp = FastMCP("DatabaseServer")

# 数据库配置
DB_NAME = 'crm'  # 指定数据库名称


def get_db_connection():
    """
    获取数据库连接
    """
    try:
        connection = mysql.connector.connect(
            host='localhost',
            port=3306,
            user='root',
            password='root',
            database=DB_NAME  # 使用配置的数据库名称
        )
        return connection
    except Error as e:
        raise Exception(f"数据库连接失败: {e}")


@mcp.tool()
def execute_mysql_query(query: str) -> str:
    """
    执行给定的 SQL 查询语句并返回结果或错误信息。
    
    Args:
        query: 要执行的 SQL 查询语句
        
    Returns:
        str: 查询结果或错误信息
    """
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute(query)
        
        # 检查是否为 SELECT 查询
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            if not results:
                return "查询成功，但没有返回任何结果。"
            
            # 格式化结果
            formatted_results = []
            for row in results:
                formatted_results.append(str(row))
            
            return f"查询成功，返回 {len(results)} 行结果:\n" + "\n".join(formatted_results)
        else:
            # 对于非 SELECT 查询，提交事务
            connection.commit()
            affected_rows = cursor.rowcount
            return f"查询执行成功，影响了 {affected_rows} 行数据。"
            
    except Error as e:
        return f"SQL 执行错误: {e}"
    except Exception as e:
        return f"执行过程中发生错误: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


@mcp.tool()
def list_tables() -> str:
    """
    获取当前数据库中所有数据表的列表。
    
    Returns:
        str: 数据表列表或错误信息
    """
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # 查询所有表
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if not tables:
            return "当前数据库中没有数据表。"
        
        # 格式化表名列表
        table_names = [table[0] for table in tables]
        return f"当前数据库中共有 {len(table_names)} 个数据表:\n" + "\n".join(f"- {name}" for name in table_names)
        
    except Error as e:
        return f"获取表列表时发生错误: {e}"
    except Exception as e:
        return f"执行过程中发生错误: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


@mcp.tool()
def get_table_schema(table_name: str) -> str:
    """
    获取指定数据表的表结构（列信息）。
    
    Args:
        table_name: 数据表名称
        
    Returns:
        str: 表结构信息或错误信息
    """
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # 查询表结构
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        
        if not columns:
            return f"表 '{table_name}' 不存在或没有列信息。"
        
        # 格式化列信息
        schema_info = []
        for column in columns:
            field_name = column[0]
            field_type = column[1]
            null_allowed = column[2]
            key_info = column[3] if len(column) > 3 else ""
            default_value = column[4] if len(column) > 4 else ""
            extra_info = column[5] if len(column) > 5 else ""
            
            schema_info.append(
                f"列名: {field_name} | "
                f"类型: {field_type} | "
                f"允许空值: {null_allowed} | "
                f"键信息: {key_info} | "
                f"默认值: {default_value} | "
                f"额外信息: {extra_info}"
            )
        
        return f"表 '{table_name}' 的结构信息 (共 {len(columns)} 列):\n" + "\n".join(schema_info)
        
    except Error as e:
        return f"获取表结构时发生错误: {e}"
    except Exception as e:
        return f"执行过程中发生错误: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport="stdio")
