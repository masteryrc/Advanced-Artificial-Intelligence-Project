import pymysql
from pymysql.cursors import DictCursor

def execute_query(sql: str) -> list:
    """执行 SQL 并返回结果列表"""
    connection = None
    cursor = None
    try:
        # 建立数据库连接
        connection = pymysql.connect(
            host='127.0.0.1',
            user='bankUser',
            password='Bank_User123',
            database='bank_database',
            charset='utf8mb4',
            cursorclass=DictCursor   # 以字典形式返回行
        )
        cursor = connection.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()  # 返回列表，元素为字典
        print(results)
        return results
    except pymysql.Error as e:
        # 可以记录日志，这里直接抛出
        print(e)
        raise Exception(f"数据库查询失败: {e}")
    finally:
        # 确保关闭游标和连接
        if cursor:
            cursor.close()
        if connection:
            connection.close()