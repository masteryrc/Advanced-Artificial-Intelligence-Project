# Please install OpenAI SDK first: `pip3 install openai`
from cgitb import text
from cmd import PROMPT
from itertools import count
import os
from random import sample
from unittest import result
from openai import OpenAI
from knowledge_base.search import retrieve_similar_queries
from datasource import execute_query

client = OpenAI(
    api_key="sk-9401a685fc04497d85f2681b49b7fa17",
    base_url="https://api.deepseek.com")

def classify_with_llm(question):
    classify_prompt = """
    你是一个对话路由器。用户会输入一段话，你需要判断它是：
    - "professional"：用户询问某个金融App（如云闪付、中国工商银行app等）的具体使用方法、功能操作、账户问题、交易异常、优惠活动等业务相关的问题。
    - "casual"：用户只是日常聊天、打招呼、表达情绪、或者与金融App完全无关的闲聊。

    只输出一个单词：professional 或 casual。不要输出任何其他内容。
    """

    # question = input().strip()
    messages = [{"role": "system", "content": classify_prompt},
                {"role": "user", "content": question}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    # print(f"{response.choices[0].message}")
    label = response.choices[0].message.content.strip().lower()
    return label if label in ["professional", "casual"] else "casual"

def casual_chat_llm(user_text: str) -> str:
    """处理日常闲聊，返回友好的回答"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个友好、轻松的聊天助手，回答日常问题。"},
            {"role": "user", "content": user_text}
        ],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content

def is_data_query(user_query: str, context: str) -> bool:
    """判断用户问题是否涉及查询交易记录/消费流水等数据"""
    prompt = f"""
    你是一个路由器。用户的问题是：“{user_query}”
    参考资料是：“{context}”
    请判断该问题是否需要通过查询数据库解答，如果查回表结构相关知识，优先考虑通过查询数据库解答。
    只回答“是”或“否”。
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    answer = response.choices[0].message.content.strip()
    print("is_data_query answer is:"+answer)
    return answer == "是"

def generate_sql(user_query: str, context: str) -> str:
    prompt = f"""
根据以下表结构，将用户问题转换为 SQL 查询语句。只输出 SQL 语句，不要有其他解释。SQL 必须是只读的 SELECT 查询。

客观知识：
{context}

用户问题：{user_query}

SQL：
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    sql = response.choices[0].message.content.strip()
    print("generate_sql result is :"+sql)
    if not sql.upper().startswith("SELECT"):
        raise ValueError("生成的 SQL 不是 SELECT 查询")
    return sql


def professional_rag_llm(user_query):
    # 检索 top-k
    retrieved = retrieve_similar_queries(user_query, top_k=2)
    
    if not retrieved or retrieved[0]['score'] < 0.5:  # 相似度过低
        return "抱歉，我暂时没有找到相关的操作指引。请确认您的问题是否与云闪付使用相关，或换个方式描述。"
    
    # 构造上下文
    context = "\n\n".join([
        f"【参考问题】{item['question']}\n【解决方法】{item['answer']}"
        for item in retrieved
    ])
    print(f"context:{context}")
    
    # 判断是否为数据查询类
    if is_data_query(user_query,context):
        try:
            sql = generate_sql(user_query,context)
            select_result = execute_query(sql)
            # 调用 LLM 生成最终回答
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是金融App专家。请根据以下参考知识和真实数据回答用户问题。如果知识不足以解答，请直接说明，不要编造。"},
                    {"role": "user", "content": f"用户问题：{user_query}\n\n参考知识：\n{context}\n\n查询语句：\n{sql}\n\n结果数据：\n{select_result}"}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"查询失败：{str(e)}。请确认问题是否明确，或联系管理员。"
        return answer
    else:
        # 调用 LLM 生成最终回答
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是金融App客服。请根据以下参考知识回答用户问题。如果知识不足以解答，请直接说明，不要编造。"},
                {"role": "user", "content": f"用户问题：{user_query}\n\n参考知识：\n{context}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

def main():
    user_input = "user_input"
    while True:
        user_input = input("请输入您的问题: ").strip()
        if not user_input:
            print("输入为空，请重新输入。")
            continue
        if user_input == "exit":
            print("结束问答。")
            return

        # 1. 分类
        category = classify_with_llm(user_input)
        print(f"[分类结果] {category}")

        # 2. 根据类别选择处理流程
        if category == "casual":
            answer = casual_chat_llm(user_input)
        else:  # professional
            answer = professional_rag_llm(user_input)

        # 3. 输出最终回答
        print("\n[助手回答]")
        print(answer)

# ---------- 程序入口 ----------
if __name__ == "__main__":
    main()
