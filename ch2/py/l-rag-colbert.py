"""
Ch2-l: ColBERT 检索（更精确的多向量检索）

【学习目标】
- 了解 ColBERT 检索模型
- 学会用 RAGatouille 库

【什么是 ColBERT？】
- 传统检索: 整个文档变成一个向量
- ColBERT: 每个 token 都有向量，更精确
- 特别适合需要精确匹配的场景

【限制】
- 不支持 Windows（可以用 WSL2）
- 只支持 Python

【依赖安装】
pip install -U ragatouille transformers

【Python 语法】
- def 函数名(参数: 类型) -> 返回类型: 类型注解
- next(iter(...)): 获取迭代器的第一个元素
- requests.get(): 发送 HTTP GET 请求
"""
from dotenv import load_dotenv
load_dotenv()


# ============ 导入 ============
from ragatouille import RAGPretrainedModel
import requests

# ============ 加载预训练的 ColBERT 模型 ============
# colbert-ir/colbertv2.0 是 Hugging Face 上的预训练模型
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


# ============ 辅助函数：获取 Wikipedia 页面内容 ============
def get_wikipedia_page(title: str):
    """
    获取 Wikipedia 页面的完整文本内容

    参数:
        title: str - Wikipedia 页面标题

    返回:
        str - 页面的纯文本内容

    【Python 语法】
    - title: str 是类型注解，表示 title 参数应该是字符串
    - 三引号字符串是函数的文档字符串（docstring）
    """
    # Wikipedia API 地址
    URL = "https://en.wikipedia.org/w/api.php"

    # API 请求参数（字典）
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # 请求头，遵守 Wikipedia 的最佳实践
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1"}

    # 发送 GET 请求
    response = requests.get(URL, params=params, headers=headers)

    # 解析 JSON 响应
    data = response.json()

    # 提取页面内容
    # data["query"]["pages"] 是字典，键是页面 ID
    # next(iter(...)) 获取第一个（也是唯一的）页面
    page = next(iter(data["query"]["pages"].values()))

    # 如果有 extract 字段就返回，否则返回 None
    return page["extract"] if "extract" in page else None


# ============ 获取文档 ============
full_document = get_wikipedia_page("Hayao_Miyazaki")
print(f"文档长度: {len(full_document)} 字符")

# ============ 创建索引 ============
# index() 会自动分割文档并创建索引
RAG.index(
    collection=[full_document],     # 文档列表
    index_name="Miyazaki-123",      # 索引名称
    max_document_length=180,        # 每个块的最大长度
    split_documents=True,           # 自动分割
)

# ============ 搜索 ============
results = RAG.search(
    query="What animation studio did Miyazaki found?",
    k=3  # 返回 3 个结果
)
print("\n搜索结果:", results)

# ============ 作为 LangChain 检索器使用 ============
# as_langchain_retriever() 让 ColBERT 可以和 LangChain 集成
retriever = RAG.as_langchain_retriever(k=3)

# invoke() 是 LangChain 的标准接口
langchain_results = retriever.invoke("What animation studio did Miyazaki found?")
print("\nLangChain 检索结果:", langchain_results)
