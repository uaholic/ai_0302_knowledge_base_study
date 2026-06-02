import base64
import mimetypes
import re
from pathlib import Path
from tkinter import image_names
from typing import Dict

from langchain_core.messages import HumanMessage

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


# 从state获取返回 md_content, md_path, img_path
def load_md_and_img_dir(state: ImportGraphState)->tuple[str,Path,Path]:
    # 先获取md_path目录
    md_path = state['md_path']
    if not md_path:
        logger.error("md_path 不存在")
        raise ValueError("md_path 不存在")

    md_content = state['md_content']
    md_path_obj = Path(md_path)
    if not md_content:
        logger.warning("md_content 不存在,从md_path读取，继续执行")
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"从 {md_path} 读取 md_content 失败")
            raise ValueError(f"从 {md_path} 读取 md_content 失败")
        # 老师没加这行，但是感觉应该加
        # state['md_content'] = md_content 老师说在最后加 应该是考虑后面改造方便
    img_path = md_path_obj.parent / "images"
    return md_content, md_path_obj, img_path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# return [img_name,img_path,[上文,下文]]
def scan_images(md_content: str, img_path: Path,context_length: int = 100)->list[tuple[str, str,tuple[str,str]]]:
    result = []
    for img_file in img_path.iterdir():
        if img_file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.warning(f"当前文件{str(img_file)}不是图片，跳过")
            continue

        reg=re.compile(r"\!\[.*?\]\(.*?"+re.escape(img_file.name)+".*?\)")
        match = reg.search(md_content)

        if not match:
            logger.warning(f"未找到图片 {img_file.name},直接跳过")
            continue

        start = match.span()[0]
        end = match.span()[1]

        pre_context = md_content[max(0, start - context_length):start]
        post_context = md_content[end:min(len(md_content), end + context_length)]

        result.append((img_file.name, str(img_file), (pre_context, post_context)))

        logger.info(f"找到图片 {img_file.name}，上下文长度为 {context_length}，上下文为：{pre_context}...{post_context}")

    return result

# 使用视觉模型对图片进行意图识别
def summarize_images(image_context_list:list[tuple[str, str,tuple[str,str]]],stem:str)->dict[str, str]:
    vision_model = llm_provider.vision_chat()
    image_summary_dict:Dict[str, str] = {}
    for img_name, img_path, (pre_context, post_context) in image_context_list:
        image_summary_prompt = load_prompt("image_summary", root_folder=stem, image_content=(pre_context, post_context))
        image_path_obj=Path(img_path)
        image_base64_str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        human_message = HumanMessage(
            content = [
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:{mimetypes.guess_type(img_name)[0]};base64,{image_base64_str}"
                    },
                },
                {
                    "type":"text",
                    "text":image_summary_prompt,
                }
            ]
        )
        resp = vision_model.invoke([human_message])
        image_summary = resp.content
        image_summary_dict[img_name] = image_summary

    logger.info(f"图片识别结果：{image_summary_dict}")
    return image_summary_dict





def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """

    md_content, md_path_obj, img_path = load_md_and_img_dir(state)

    if not any(img_path.iterdir()):
        # 空文件夹
        logger.warning(f"当前{md_content}没有图片,无需图片处理!正常进入下一个节点!!")
        return state

    img_list_md = scan_images(md_content, img_path)

    images_info = summarize_images(img_list_md, md_path_obj.stem)

    return state