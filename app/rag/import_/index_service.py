from pymilvus import DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger


def validate_chunks(state):
    chunks = state["chunks"]
    if not chunks or len(chunks) == 0:
        logger.error("chunks is empty")
        raise ValueError("chunks is empty")
    return chunks


def prepare_chunks_collection():
    client = milvus_gateway.client
    collection_name = milvus_gateway.chunk_collection_name
    if client.has_collection(collection_name=collection_name):
        logger.info(f"集合[{collection_name}]已存在，无需创建")
        return

    # 创建 schema，启用自动 ID 和动态字段
    schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)

    # 添加主键字段：chunk_id，INT64 类型，自增
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 添加文件标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加主体名称字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)

    # 添加父标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片序号字段：INT8 类型
    schema.add_field(field_name="part", datatype=DataType.INT8)

    # 添加内容字段：VARCHAR 类型，最大长度 65535（支持长文本）
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    # 添加稠密向量字段：FLOAT_VECTOR 类型，维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,  # Maximum number of neighbors each nodes can connect to in the graph
            "efConstruction": 100  # Number of candidate neighbors considered for connection during index construction
        }
    )

    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE"
        }
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    logger.info(f"{milvus_gateway.chunk_collection_name}第一次完成初始化!!")


def remove_old_chunks(file_title):
    client = milvus_gateway.client
    client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title=='{file_title}'"
    )

def insert_chunks(chunks):
    client = milvus_gateway.client
    result = client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=chunks
    )
    logger.info(f"{result.get("insert_count",0)}条数据入库成功")
    logger.info(f"{milvus_gateway.chunk_collection_name}入库完成!!")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")

def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    chunks = validate_chunks(state)
    prepare_chunks_collection()
    remove_old_chunks(state["file_title"])
    insert_chunks(chunks)

    logger.info(f"{milvus_gateway.chunk_collection_name}入库完成!!")
    return state
