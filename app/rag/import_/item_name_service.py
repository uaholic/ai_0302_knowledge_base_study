from http import client

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_chunks_and_title(state: ImportGraphState) -> tuple[list[dict], str]:
    chunks = state.get("chunks")
    file_title = state.get("file_title")
    if not chunks:
        logger.error("chunks is empty, failed")
        raise ValueError("chunks is empty, failed")
    if not file_title:
        file_title = chunks[0]['file_title'] or 'default_file_title'
        logger.warning(f"file_title is empty, use {file_title}")

    return chunks, file_title


def build_document_context(chunks) -> str:
    top_chunk = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    context = ""
    for index, chunk in enumerate(top_chunk, start=1):
        context += f"切片:{index} 标题:{chunk['title']} 父标题:{chunk['parent_title']} 内容:{chunk['content']}\n"
    return context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]


def recognize_item_name(context, file_title) -> str:
    chat_mode = llm_provider.chat()
    system_prompt = load_prompt("product_recognition_system")
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    chain = chat_mode | StrOutputParser()
    resp = chain.invoke(input=messages)
    return resp


def embed_item_name(item_name: str) -> tuple[list[float], list[float]]:
    result = llm_provider.embed_documents([item_name])
    return result['dense'][0], result['sparse'][0]


def prepare_item_name_collection():
    client = milvus_gateway.client
    if client.has_collection(collection_name=milvus_gateway.item_collection_name):
        logger.info(f"item_name_collection {milvus_gateway.item_collection_name} 已存在，无需创建")
        return

    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_fiels=True
    )

    # https://milvus.io/docs/zh/v2.6.x/sparse_vector.md
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 4. 创建集合对应indexs [索引]
    # 3.3. Prepare index parameters
    index_params = client.prepare_index_params()

    # 3.4. Add indexes
    index_params.add_index(
        # 给哪个字段创建索引 字段应该是经常查询的字段
        field_name="dense_vector",
        # 索引的类型 索引就是外部创建一种高效的数据类型  [目录]-> 查询 -> 内存地址 -> 链接到对应的实体数据
        # 推荐: AUTOINDEX -> 自动创建索引 自动选择类型 我有点不推荐!
        # 为了减少学习曲线，Milvus 提供了AUTOINDEX。通过AUTOINDEX，Milvus 可以在建立索引的同时分析 Collections
        # 中的数据分布，并根据分析结果设置最优化的索引参数，从而在搜索性能和正确性之间取得平衡。
        # HNSW : 分层图 -> 类似地图搜索过程  [精度最高 / 内存在有最大]
        # IVF_FLAT : 分桶 nlist = 64 找到对应桶 / 细化筛选 [比 FLAT快, 占有内存中等]
        # FLAT :  直接所有向量搜索和比较 [最慢]
        index_type="HNSW",
        # 相识度算法 L2 [0-2] COSINE  IP  [-1 1]
        metric_type="COSINE",
        params={
            "M": 64,  # Maximum number of neighbors each node can connect to in the graph
            "efConstruction": 100  # Number of candidate neighbors considered for connection during index construction
        }  # I
    )

    index_params.add_index(
        field_name="sparse_vector",
        # 稀疏向量 2.6 只有倒排索引
        # 内容 -> 向量相似度
        # doc1 = {1:x 3:x}
        # doc2 = {1:x,4:x}
        # 1位置 = doc1 , doc2
        # 3位置 = doc1
        # 4位置 = doc2
        # 搜索的稀疏向量 {1:k} -> doc1 doc2
        index_type="SPARSE_INVERTED_INDEX",
        # IP (内积）：使用点积衡量相似性。
        metric_type="IP",
        # 算法识别 影响小的值跳过,提高相似度比较的效率
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )
    # 5. 创建集合 (集合的名字 schema indexs )
    client.create_collection(
        collection_name=milvus_gateway.item_collection_name,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"{milvus_gateway.item_collection_name}第一次完成初始化!!")


def apply_item_name(item_name, chunks: list[dict]):
    for chunk in chunks:
        chunk['item_name'] = item_name

    logger.info(f"将识别结果回填到 chunks[0]['item_name']: {chunks[0]['item_name']}")


def upsert_item_name(item_name, file_title, dense_vector, spare_vector):
    client = milvus_gateway.client
    client.delete(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title == '{file_title}'"
    )

    result = client.insert(
        collection_name=milvus_gateway.item_collection_name,
        data=[{
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": spare_vector
        }],
    )

    logger.info(f"{milvus_gateway.item_collection_name} upsert item_name: {item_name},return:{result}")

def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    chunks, file_title = validate_chunks_and_title(state)

    context = build_document_context(chunks)

    item_name = recognize_item_name(context, file_title)
    logger.info(f"识别结果：{item_name}")
    apply_item_name(item_name, chunks)
    dense_vector, spare_vector = embed_item_name(item_name)
    prepare_item_name_collection()
    upsert_item_name(item_name, file_title, dense_vector, spare_vector)
    state['chunks'] = chunks
    state['item_name'] = item_name
    return state
