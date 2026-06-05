import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_OVERLAP ,CHUNK_SIZE
from app.shared.runtime.logger import logger


def load_md_content(state: ImportGraphState, md_content: str, md_path_obj: Path, file_title: str) -> tuple[
    str, Path, str]:
    if not md_content:
        md_content = md_path_obj.read_text(encoding="utf-8")
        logger.warning(f"md_content为空默认从 {md_path_obj} 读取")
        state["md_content"] = md_content
    if not file_title:
        file_title = md_path_obj.stem
        logger.warning(f"file_title为空默认从 {md_path_obj} 获取")
        state["file_title"] = file_title

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    state["md_content"] = md_content

    return md_content, md_path_obj, file_title


def split_by_title(md_content: str, file_title: str) -> list[dict]:
    chunks: list[dict] = []
    cur_title: str = ""
    cur_content: list[str] = []
    is_code_scope: bool = False

    reg = re.compile(r"^\s*#{1,6}\s+.+$")

    for line in md_content.split("\n"):

        if not line.strip():
            logger.warning(f"当前行为空行，跳过")
            continue

        strip_line = line.lstrip()
        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            cur_content.append(line)
            is_code_scope = not is_code_scope
            continue

        if is_code_scope:
            cur_content.append(line)
            continue

        res = reg.search(line)
        if res:
            if not any(line.strip() for line in cur_content):
                cur_title = line
                cur_content = []
                continue
            else:
                if not cur_content:
                    # 空标题没有内容 跳过
                    cur_title = line
                    continue
                chunks.append({
                    "title": cur_title,
                    "content": "\n".join(cur_content),
                    "file_title": file_title,
                })
                cur_title = line
                cur_content = []
        else:
            cur_content.append(line)

    if cur_title and any(line.strip() for line in cur_content):
        chunks.append({
            "title": cur_title,
            "content": "\n".join(cur_content),
            "file_title": file_title,
        })

    logger.info(f"文档切分完成，总计切分 {len(chunks)} 个 chunk")

    return chunks


def _split_long_chunk(chunk):
    content = chunk.get("content", "") or ""
    title = chunk.get("title")
    body = content
    if content.startswith(title):
        body = content[len(title):].lstrip()
    prefix = title + "\n"
    available_length = CHUNK_MAX_SIZE-len(prefix)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_length,       # 拆分后的正文最大长度
        chunk_overlap=CHUNK_OVERLAP,                # 块之间无重叠
        separators=["\n\n", "\n", "。", "！", "？"],
    )
    sub_thunks = []
    for index, split_text in enumerate(splitter.split_text(body), start=1):
        text = split_text.strip()

        if not text:
            continue

        full_text = (prefix + text).strip()
        sub_thunks.append({
            "title": f"{title}_{index}" if title else f"chunk_{index}",  # 子标题：原标题_序号
            "content": full_text,  # 完整内容
            "parent_title": title,  # 父标题（用于溯源）
            "part": index,  # 序号（同一章节下的第N部分）
            "file_title": chunk.get("file_title"),  # 文档原始标题
        })

    logger.info(f"文档切分完成，总计切分 {len(sub_thunks)} 个子 chunk")
    return sub_thunks


def _merge_short_chunks(final_chunks,max_length:int = CHUNK_MAX_SIZE,min_length:int=CHUNK_SIZE):
    final_merge_chunks = []

    start_chunk = None

    for next_chunk in final_chunks:
        if not start_chunk:
            start_chunk = next_chunk
            continue

        is_lt_chunk_size = len(start_chunk.get("content")) < min_length
        is_same_parent_title = start_chunk.get("parent_title") == next_chunk.get("parent_title")
        if is_lt_chunk_size and is_same_parent_title:
            next_content_without_title = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 2:]
            start_content = start_chunk.get("content")

            merged_content = start_content + "\n" + next_content_without_title
            if len(merged_content)<=max_length:
                start_chunk["content"]=merged_content
                logger.info(f"合并 chunk: {start_chunk['title']} -> {next_chunk['title']}")
        else:
            final_merge_chunks.append(start_chunk)
            start_chunk = next_chunk

    if start_chunk:
        final_merge_chunks.append(start_chunk)

    return final_merge_chunks


def refine_chunks(chunks: list[dict], max_length: int = CHUNK_MAX_SIZE) -> list[dict]:
    final_chunks: list[dict] = []
    for chunk in chunks:
        content = chunk['content']
        if len(content) > max_length:
            sub_chunks = _split_long_chunk(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    for chunk in final_chunks:
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk['title']
        if "part" not in chunk:
            chunk['part'] = 1

    final_merge_chunks = _merge_short_chunks(final_chunks)

    return final_merge_chunks


def backup_chunks_json(final_chunks, md_path_obj):
    json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
    json_path_obj.write_text(json.dumps(final_chunks,indent=4,ensure_ascii=False), encoding="utf-8")
    logger.info(f"文档切分结果备份完成：{json_path_obj}")


def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    md_path = Path(state["md_path"])
    md_content = state["md_content"]
    file_title = state["file_title"]
    md_content, md_path, file_title = load_md_content(state, md_content, md_path, file_title)
    chunks = split_by_title(md_content, file_title)
    final_chunks = refine_chunks(chunks)
    backup_chunks_json(final_chunks, md_path)
    state["chunks"] = final_chunks
    return state
