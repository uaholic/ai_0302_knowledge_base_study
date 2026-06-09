from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_state(state: QueryGraphState)->tuple[str, list[str]]:
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    if not rewritten_query:
        raise ValueError("rewritten_query is empty")
    if not item_names or len(item_names) == 0:
        raise ValueError("item_names is empty")
    return rewritten_query, item_names


def llm_hyde_ans(rewritten_query):
    chat_mode = llm_provider.chat()
    human_prompt = load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    messages = [
        HumanMessage(content=human_prompt)
    ]
    chain = chat_mode | StrOutputParser()
    resp = chain.invoke(input=messages)
    return resp


def query_hyde_from_milvus(item_names, rewritten_query, hyde_answer):
    embed_result = llm_provider.embed_documents([rewritten_query + hyde_answer])
    dense_vector = embed_result["dense"][0]
    sparse_vector = embed_result["sparse"][0]
    ann_request_list = milvus_gateway.create_requests(dense_vector, sparse_vector, expr=f"item_name in {item_names}")
    search_result = milvus_gateway.hybrid_search(collection_name=milvus_gateway.chunk_collection_name,
                                                 reqs=ann_request_list, norm_score=True,
                                                 output_fields=["chunk_id", "title", "parent_title", "file_title",
                                                                "part", "content", "item_name"], limit=5)
    result = search_result[0]
    if not result or len(result) == 0:
        logger.info("没有查询到数据")
        return []

    result_dict = [{
        "chunk_id": item.get("id") or item.get("entity").get("chunk_id"),  # 片段ID
        "item_name": item.get("entity").get("item_name", ""),  # 归属主体名称
        "title": item.get("entity").get("title"),  # 片段标题
        "parent_title": item.get("entity").get("parent_title"),  # 父标题/章节
        "part": item.get("entity").get("part"),  # 部分标识
        "file_title": item.get("entity").get("file_title"),  # 来源文件标题
        "content": item.get("entity").get("content", ""),  # 片段文本内容
        "score": item.get("distance", 0.0),  # 相似度分数
        "type": "milvus",  # 来源类型（向量库）
        "url": None,  # 附件URL（无）
    } for item in result]
    return result_dict


def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """
    rewritten_query,item_names = validate_state(state)
    hyde_answer = llm_hyde_ans(rewritten_query)
    state["hyde_embedding_chunks"] = query_hyde_from_milvus(item_names,rewritten_query,hyde_answer)
    return state