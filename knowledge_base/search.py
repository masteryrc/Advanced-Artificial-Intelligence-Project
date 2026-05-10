# build_index.py
import json
import faiss
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer

# ---------- 全局配置 ----------
KNOWLEDGE_JSON = "./knowledge_base/questions.json"      # 原始知识文件
INDEX_FILE = "kb_index.faiss"          # 保存的 FAISS 索引
METADATA_FILE = "kb_metadata.json"     # 保存知识库元数据
MODEL_NAME = '/Users/yrc/my_computer/code/Advanced-Artificial-Intelligence-Project/model_dir'

# ---------- 加载模型（全局，供检索使用） ----------
model = SentenceTransformer(MODEL_NAME)

# ---------- 加载索引和元数据（供检索使用） ----------
def load_knowledge_base():
    """加载之前构建好的索引和元数据"""
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    return index, knowledge_base

# ---------- 检索函数（供外部调用） ----------
def retrieve_similar_queries(user_query, top_k=3):
    """
    从知识库中检索与用户问题最相似的 top_k 条知识。
    返回列表，每个元素包含 question, answer, score
    """
    # 加载索引和知识库（实际使用中可缓存到全局变量，避免重复加载）
    index, knowledge_base = load_knowledge_base()
    
    # 编码用户问题
    query_vec = model.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    # 检索
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx != -1 and idx < len(knowledge_base):
            item = knowledge_base[idx]
            results.append({
                'question': item['question'],
                'answer': item['answer'],
                'score': float(score)
            })
    return results

# ---------- 构建索引（仅在直接运行此脚本时执行） ----------
if __name__ == "__main__":
    # 1. 读取知识库
    with open(KNOWLEDGE_JSON, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    
    # 2. 编码问题
    questions = [item['question'] for item in knowledge_base]
    question_embeddings = model.encode(questions, convert_to_numpy=True)
    
    # 3. 构建 FAISS 索引
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(question_embeddings)
    index.add(question_embeddings)
    
    # 4. 保存索引和元数据
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    print(f"索引构建完成，共 {len(knowledge_base)} 条记录，保存至 {INDEX_FILE} 和 {METADATA_FILE}")