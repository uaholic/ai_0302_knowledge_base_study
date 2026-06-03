from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_chunks_and_title(state: ImportGraphState) -> tuple[list[str], str]:
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


def recognize_item_name(context, file_title)->str:
    chat_mode = llm_provider.chat()
    system_prompt = load_prompt("product_recognition_system")
    human_prompt = load_prompt("item_name_recognition",file_title=file_title,context=context)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    chain = chat_mode|StrOutputParser()
    resp = chain.invoke(input=messages)
    return resp


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

    item_name = recognize_item_name(context,file_title)
    logger.info(f"识别结果：{item_name}")
    return state
