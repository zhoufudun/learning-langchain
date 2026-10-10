from dotenv import load_dotenv
load_dotenv()
from typing import List, TypedDict
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
# DeepSeek 没有 Embedding API，改用智谱 AI
from langchain_community.embeddings import ZhipuAIEmbeddings


# ============ 分批嵌入的包装类 ============
class BatchingZhipuEmbeddings(ZhipuAIEmbeddings):
    """智谱嵌入，自动分批（单批最多 64 条）"""
    batch_size: int = 64

    def embed_documents(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(super().embed_documents(batch))
        return all_embeddings
from langchain import hub
from langchain_openai import ChatOpenAI


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        scraped_documents: list of documents
        vectorstore: vectorstore
    """

    question: str
    scraped_documents: List[str]
    vectorstore: InMemoryVectorStore
    answer: str


def scrape_blog_posts(state) -> List[Document]:
    """
    Scrape the blog posts and create a list of documents
    """

    urls = [
        "https://blog.langchain.dev/top-5-langgraph-agents-in-production-2024/",
        "https://blog.langchain.dev/langchain-state-of-ai-2024/",
        "https://blog.langchain.dev/introducing-ambient-agents/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    return {"scraped_documents": docs_list}


def indexing(state):
    """
    Index the documents
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(state["scraped_documents"])

# Add to vectorDB
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=BatchingZhipuEmbeddings(model="embedding-3"),
    )
    return {"vectorstore": vectorstore}


def retrieve_and_generate(state):
    """
    Retrieve documents from vectorstore and generate answer
    """
    question = state["question"]
    vectorstore = state["vectorstore"]

    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # fetch relevant documents
    docs = retriever.invoke(question)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": question})  # generate answer
    answer = llm.invoke(formatted)
    return {"answer": answer}


# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve_and_generate", retrieve_and_generate)  # retrieve
workflow.add_node("scrape_blog_posts", scrape_blog_posts)  # scrape web
workflow.add_node("indexing", indexing)  # index

# Build graph
workflow.add_edge(START, "scrape_blog_posts")
workflow.add_edge("scrape_blog_posts", "indexing")
workflow.add_edge("indexing", "retrieve_and_generate")

workflow.add_edge("retrieve_and_generate", END)

# Compile
graph = workflow.compile()
