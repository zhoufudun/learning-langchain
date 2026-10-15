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
# LangChain 1.x 已移除 langchain.utils；旧功能迁移到 langchain_classic。
from langchain_classic.utils.math import cosine_similarity  # 计算余弦相似度
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

# ============ 定义专家提示词 ============
# 物理专家模板
physics_template = """You are a very smart physics professor. You are great at     answering questions about physics in a concise and easy-to-understand manner.     When you don't know the answer to a question, you admit that you don't know. Here is a question: {query}"""

# 数学专家模板
math_template = """You are a very good mathematician. You are great at answering     math questions. You are so good because you are able to break down hard     problems into their component parts, answer the component parts, and then     put them together to answer the broader question. Here is a question: {query}"""

# ============ 分批嵌入的包装类 ============
# class BatchingZhipuEmbeddings(ZhipuAIEmbeddings):
#     """智谱嵌入，自动分批（单批最多 64 条）"""
#     batch_size: int = 64
#
#     def embed_documents(self, texts):
#         all_embeddings = []
#         for i in range(0, len(texts), self.batch_size):
#             batch = texts[i:i + self.batch_size]
#             all_embeddings.extend(super().embed_documents(batch))
#         return all_embeddings
#
#
# # ============ 预计算模板向量 ============
# embeddings = BatchingZhipuEmbeddings(model="embedding-3",batch_size=32)

embeddings = ZhipuAIEmbeddings(model="embedding-3")

prompt_templates = [physics_template, math_template]

# 把两个模板都转换成向量
# 这一步只需要做一次（通常在启动时做好）
prompt_embeddings = embeddings.embed_documents(prompt_templates)

print(f"模板向量维度: {len(prompt_embeddings[0])}")
# 向量有 2048 个数字，完整打印会淹没终端；查看前 5 个值即可。
print(f"第一个模板向量的前 5 个值: {prompt_embeddings[0][:5]}")

# ============ 语义路由函数 ============
@chain
def prompt_router(query: str):
    """
        根据查询内容，选择最合适的提示词模板
        【Python 语法】
        - similarity.argmax(): 返回最大值的索引
          例如: [0.8, 0.9, 0.7].argmax() → 1
        """
    # 1. 把用户查询转换成向量
    embed_query = embeddings.embed_query(query)
    # 2. 计算查询向量和每个模板向量的余弦相似度
    # 返回值是 numpy 数组: [[0.85, 0.72]]（和 physics、math 的相似度）
    similarity = cosine_similarity([embed_query], prompt_embeddings)[0]
    # 3. 找出最相似的模板
    # argmax() 返回最大值的索引
    # most_similar 要么physics_template, 要么 math_template
    most_similar = prompt_templates[similarity.argmax()]
    # 打印路由结果
    print(f"most_similar= {most_similar}")
    print("使用:", "MATH" if most_similar == math_template else "PHYSICS")
    # 4. 用当前问题填入模板的 {query} 占位符。
    # invoke() 返回 PromptValue，它是 ChatOpenAI 可以直接接收的提示词对象。
    selected_prompt = PromptTemplate.from_template(most_similar)
    return selected_prompt.invoke({"query": query})


# ============ 完整语义路由链 ============
# 1. prompt_router: 根据查询选择模板
# 2. ChatOpenAI(): 调用 LLM
# 3. StrOutputParser(): 解析输出为字符串
llm = ChatOpenAI(model="gpt-5.6-luna", temperature=0)
# StrOutputParser() 必须带 ()：
# - StrOutputParser：类（相当于 Java 的类名）
# - StrOutputParser()：对象实例，才能作为链中的输出解析组件
semantic_router = prompt_router | llm | StrOutputParser()

# ============ 测试 ============
result = semantic_router.invoke("What's a black hole")
print("\n语义路由结果:", result)
# 会自动选择 physics_template，因为"黑洞"和物理模板更相似

result = semantic_router.invoke("10+10=？")
print("\n语义路由结果:", result)
