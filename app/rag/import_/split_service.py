import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
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
            "content": cur_content,
            "file_title": file_title,
        })

    logger.info(f"文档切分完成，总计切分 {len(chunks)} 个 chunk")

    return chunks


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
    state["chunks"] = chunks
    return state
