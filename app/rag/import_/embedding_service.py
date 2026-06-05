from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger


def validate_chunks(state: ImportGraphState):
    chunks = state.get("chunks")

    if not chunks or len( chunks) ==0:
        logger.error("chunks is empty")
        raise ValueError("chunks is empty")
    return chunks

def embed_chunks(chunks: list[ dict],*,step: int = EMBEDDING_BATCH_SIZE):
    final_trunks: list[dict] = []
    for i in range(0, len(chunks), step):
        batch = chunks[i:i+step]
        batch_content=[]
        for chunk in batch:
            batch_content.append(f"主体名：{chunk['item_name']},内容:{chunk['content']}")
        result = llm_provider.embed_documents(batch_content)
        for i,chunk in enumerate(batch):
            new_chunk = chunk.copy()
            new_chunk["dense_vector"]=result["dense"][i]
            new_chunk["sparse_vector"]=result["sparse"][i]
            final_trunks.append(new_chunk)
    logger.info(f"向量化完成，向量数量：{len(final_trunks)}")
    return final_trunks


def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    chunks = validate_chunks( state)
    state["chunks"] = embed_chunks(chunks)
    return state