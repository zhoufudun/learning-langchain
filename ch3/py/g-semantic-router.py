from dotenv import load_dotenv
load_dotenv()
"""
Ch3-g: 语义路由（Semantic Router）

【学习目标】
- 理解语义路由的原理
- 学会用向量相似度进行路由

【与 f-router.py 的区别】
- f-router: 用 LLM 判断路由（慢，但准确）
- g-semantic: 用向量相似度判断路由（快，适合简单场景）

【原理】
1. 预先定义几个提示词模板
2. 把模板转换成向量
3. 用户查询也转换成向量
4. 计算查询向量和模板向量的相似度
5. 选择最相似的模板

【Python 语法】
- argmax(): 返回最大值的索引
- cosine_similarity(): 计算余弦相似度
"""

# ============ 导入 ============
from langchain.utils.math import cosine_similarity  # 计算余弦相似度
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

# ============ 定义专家提示词 ============
# 物理专家模板
physics_template = """You are a very smart physics professor. You are great at     answering questions about physics in a concise and easy-to-understand manner.     When you don't know the answer to a question, you admit that you don't know. Here is a question: {query}"""

# 数学专家模板
math_template = """You are a very good mathematician. You are great at answering     math questions. You are so good because you are able to break down hard     problems into their component parts, answer the component parts, and then     put them together to answer the broader question. Here is a question: {query}"""

# ============ 预计算模板向量 ============
embeddings = BatchingZhipuEmbeddings(model="embedding-3")
prompt_templates = [physics_template, math_template]

# 把两个模板都转换成向量
# 这一步只需要做一次（通常在启动时做好）
prompt_embeddings = embeddings.embed_documents(prompt_templates)
print(f"模板向量维度: {len(prompt_embeddings[0])}")


# ============ 语义路由函数 ============
@chain
def prompt_router(query):
    """
    根据查询内容，选择最合适的提示词模板

    【Python 语法】
    - similarity.argmax(): 返回最大值的索引
      例如: [0.8, 0.9, 0.7].argmax() → 1
    """
    # 1. 把用户查询转换成向量
    query_embedding = embeddings.embed_query(query)

    # 2. 计算查询向量和每个模板向量的余弦相似度
    # 返回值是 numpy 数组: [[0.85, 0.72]]（和 physics、math 的相似度）
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]

    # 3. 找出最相似的模板
    # argmax() 返回最大值的索引
    most_similar = prompt_templates[similarity.argmax()]

    # 打印路由结果
    print("使用:", "MATH" if most_similar == math_template else "PHYSICS")

    # 4. 返回对应的提示词模板
    return PromptTemplate.from_template(most_similar)


# ============ 完整语义路由链 ============
# 1. prompt_router: 根据查询选择模板
# 2. ChatOpenAI(): 调用 LLM
# 3. StrOutputParser(): 解析输出为字符串
semantic_router = prompt_router | ChatOpenAI() | StrOutputParser()

# ============ 测试 ============
result = semantic_router.invoke("What's a black hole")
print("
语义路由结果:", result)
# 会自动选择 physics_template，因为"黑洞"和物理模板更相似
