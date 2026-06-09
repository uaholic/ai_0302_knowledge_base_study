from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger


def validate_state(state: QueryGraphState) -> tuple[str, str]:
    rewritten_query = state['rewritten_query']
    item_names = state['item_names']

    if not rewritten_query:
        raise ValueError("rewritten_query is empty")
    if not item_names or len(item_names) == 0:
        raise ValueError("item_names is empty")
    return rewritten_query, item_names[0]


def query_from_milvus(rewritten_query, item_name) -> list[dict]:
    embed_result = llm_provider.embed_documents([rewritten_query + item_name])
    dense_vector = embed_result["dense"][0]
    sparse_vector = embed_result["sparse"][0]
    ann_request_list = milvus_gateway.create_requests(dense_vector, sparse_vector, expr=f"item_name='{item_name}'")
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


def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """

    rewritten_query, item_names = validate_state(state)
    query_from_milvus(rewritten_query, item_names[0])
    return state
