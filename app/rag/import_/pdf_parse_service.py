import shutil
import time

import requests

from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS, MINERU_POLL_TIMEOUT_SECONDS, \
    MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from app.infra.config.providers import infra_config

#   validate_pdf_paths(state: ImportState) -> tuple[Path pdf_path_obj, Path local_dir_path_obj]
#         1. 读取 `pdf_path` 和 `local_dir`
#         2. 校验 `pdf_path` 是否为空
#         3. 若 `local_dir` 为空，则写入默认输出目录
#         4. 转换为 `Path` 对象
#         5. 校验 PDF 文件是否真实存在
#         6. 若输出目录不存在，则自动创建
#         7. 返回 `pdf_path_obj` 与 `local_dir_obj`

# 1. pdf dir路径校验和完善
@step_log("validate_pdf_paths")
def validate_pdf_paths(state: ImportGraphState) -> tuple[Path, Path]:
    """
    路径参数校验
    :param state:
    :return:
    """
    # 1. 读取 `pdf_path` 和 `local_dir`
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    # 2. 校验 `pdf_path` 是否为空
    if not pdf_path:
        logger.error(f"进行pdf转化md,但是pdf_path为空,无法继续进行!!")
        raise ValueError("进行pdf转化md,但是pdf_path为空,无法继续进行!!")
    # 3. 若 `local_dir` 为空，则写入默认输出目录
    if not local_dir:
        logger.warning(f"进行pdf转化md,但是发现local_dir为空,我们给与默认值 项目/output")
        local_dir = PROJECT_ROOT / "output"  # Path
    # 4. 转换为 `Path` 对象
    pdf_path_obj = Path(pdf_path) # 后续对文件存在性校验! 有他更方便
    local_dir_obj = Path(local_dir) # 这块不报错
    # 5. 校验 PDF 文件是否真实存在
    if not pdf_path_obj.exists():
        logger.error(f"进行pdf转化md,pdf_path值:{pdf_path},但是不存在对应的文件,业务无法继续!!")
        raise FileNotFoundError(f"进行pdf转化md,pdf_path值:{pdf_path},但是不存在对应的文件,业务无法继续!!")
    # 6. 若输出目录不存在，则自动创建
    if not local_dir_obj.exists() :
        logger.warning(f"进行pdf转化md,local_dir值:{local_dir_obj},但是没有对应文件夹! 我们自行创建!!")
        # parents 自动创建多层文件夹  例如: x/x/x/x  -> True会自动创建 x  x   x  x
        # exist_ok = True 当存在的时候不会创建也不会报错!
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    # 7. 返回 `pdf_path_obj` 与 `local_dir_obj`
    return pdf_path_obj,local_dir_obj

# upload_pdf_and_poll(pdf_path_obj: Path) -> str
#        1. 校验 MinerU 配置是否完整
#        2. 调用 `/file-urls/batch` 申请上传地址与 `batch_id`
#        3. 使用 `Session(trust_env=False)` 上传 PDF 文件
#        4. 根据 `batch_id` 轮询任务状态
#        5. 若任务成功，返回 `full_zip_url`
#        6. 若任务失败或超时，抛出异常

@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj:Path) -> str:
    """
       minerU交互
    :param pdf_path_obj:  上传文件的path对象
    :return: 返回的下载zip地址
    """
    # 1. 校验 MinerU 配置是否完整
    if not infra_config.mineru.base_url or not infra_config.mineru.api_key:
        logger.error(f"minerU请求核心参数为空(base_url 或者 api_key),业务无法继续进行!")
        raise ValueError(f"minerU请求核心参数为空(base_url 或者 api_key),业务无法继续进行!")
    # 2. 调用 `/file-urls/batch` 申请上传地址与 `batch_id`
    token = infra_config.mineru.api_key
    url = f"{infra_config.mineru.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": MINERU_MODEL_VERSION
    }
    try:
        response =  requests.post(url,headers=header,json= data, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)

        # 状态码是否正常200  (网络状态 服务器状态)
        if response.status_code != 200:
            logger.error(f"服务器发生异常!无法进行业务!响应状态码为:{response.status_code}")
            raise RuntimeError(f"服务器发生异常!无法进行业务!响应状态码为:{response.status_code}")
        # 判断业务是否正常0  (业务状态)
        response_dict  = response.json()
        code = response_dict.get("code")
        if code != 0:
            logger.error(f"业务处理发生异常! 业务状态码为:{code},异常信息:{response_dict.get('msg')}")
            raise RuntimeError(f"业务处理发生异常! 业务状态码为:{code},异常信息:{response_dict.get('msg')}")

        batch_id = response_dict.get("data",{}).get("batch_id")
        upload_file_url = response_dict.get("data",{}).get("file_urls")[0]
        logger.info(f"调用 `/file-urls/batch` 申请上传地址与 `batch_id`, batch_id:{batch_id},上传地址:{upload_file_url}")
    except Exception as e:
        logger.exception(f"向minerU申请上传文件地址发生异常! url参数: {url},key参数:{token}")
        raise e

    # 3. 使用 `Session(trust_env=False)` 上传 PDF 文件
    try:
        with requests.Session() as session:
            # requests.Session() 获取请求会话
            # session使用和requests是一样的
            # 作用1: 可以复用请求 requests.Session() session.get post    session.close() [根本不服用]
            # 作用2: 有些特殊的设置 trust_env = False 我谁也不信!!! 向预签名地址传递数据避免干扰成功率更高!!
            session.trust_env = False
            put_response = session.put(upload_file_url,data=pdf_path_obj.read_bytes())
            # status_code |  code
            if put_response.status_code != 200:
                logger.error(f"向地址:{upload_file_url}上传文件发生异常,状态码:{put_response.status_code},业务无法继续!!")
                raise RuntimeError(f"向地址:{upload_file_url}上传文件发生异常,状态码:{put_response.status_code},业务无法继续!!")
    except Exception as e:
        logger.exception(f"向minerU文件服务器,上传文件发生异常{str(e)}! 业务无继续!!")
        raise e

    # 4. 根据 `batch_id` 轮询任务状态
    # 前置准备工作
    get_zip_url = f"{infra_config.mineru.base_url}/extract-results/batch/{batch_id}"
    timeout = MINERU_POLL_TIMEOUT_SECONDS  # 600
    interval_time = MINERU_POLL_INTERVAL_SECONDS # 3
    start_time = time.time()

    while True:
        # 获取结果  抛出异常  timeout
        # 1. 先判定是否超时
        if time.time() - start_time >= timeout:
            logger.error(f"轮询获取:{batch_id}结果超时! 用时:{time.time() - start_time}")
            raise TimeoutError(f"轮询获取:{batch_id}结果超时! 用时:{time.time() - start_time}")
        # 2. 发起网络请求(报错,再给一次机会)
        try:
            get_response = requests.get(get_zip_url,headers=header)
        except Exception as e:
            logger.warning(f"获取下载的zipurl地址, 网络请求失败!等待后继续尝试!")
            time.sleep(interval_time)
            continue
        # 3. 判断status_code
        # 客户端 -> 服务端 -> 1 2 3 4 5
        if get_response.status_code != 200:
            # 一定是错误了,看这个错误是否给机会! 5xx
            if 500 <= get_response.status_code < 600:
                # 给机会
                logger.warning(f"获取下载的zipurl地址,minerU对应服务器发生异常! 状态码:{get_response.status_code},等待后再次尝试!!")
                time.sleep(interval_time+2)
                continue
            logger.error(f"获取下载的zipurl地址,minerU对应服务器发生异常! 状态码:{get_response.status_code},业务无法继续了!")
            raise RuntimeError(f"获取下载的zipurl地址,minerU对应服务器发生异常! 状态码:{get_response.status_code},业务无法继续了!")

        # 4. 判断code
        get_response_dict = get_response.json()
        if get_response_dict.get('code') != 0:
            logger.error(
                f"获取下载的zipurl地址,minerU对应服务器发生异常! 业务码:{get_response_dict.get('code')} ,错误信息:"
                f"{get_response_dict.get('msg')},业务无法继续了!")
            raise RuntimeError( f"获取下载的zipurl地址,minerU对应服务器发生异常! 业务码:{get_response_dict.get('code')} ,错误信息:"
                f"{get_response_dict.get('msg')},业务无法继续了!")
        # 5. 获取结果信息(是否解析完毕)  正在解析 循环  解析完毕 获取结果 return 解析失败 抛出异常
        # 获取结果的dict
        result_dict = get_response_dict.get("data",{}).get("extract_result",[])[0]
        result_state = result_dict.get("state","failed")

        if result_state == "done":
            full_zip_url = result_dict.get("full_zip_url")
            if not full_zip_url:
                # 下载地址空
                logger.error(
                    f"获取下载的zipurl地址,minerU对应服务器发生异常! 获取zip地址为空!!业务无法继续进行了!")
                raise RuntimeError(
                    f"获取下载的zipurl地址,minerU对应服务器发生异常! 获取zip地址为空!!业务无法继续进行了!")
            return full_zip_url
        if result_state == "failed":
            # 下载地址空
            logger.error(
                f"获取下载的zipurl地址,minerU对应服务器发生异常! 解析失败了!!业务无法继续进行了!")
            raise RuntimeError(
                f"获取下载的zipurl地址,minerU对应服务器发生异常! 解析失败了!!业务无法继续进行了!")
        # 正在解析中.....
        logger.warning(f"{pdf_path_obj.name}minerU正在解析中......")
        time.sleep(interval_time)


    #        5. 若任务成功，返回 `full_zip_url`
    #        6. 若任务失败或超时，抛出异常
@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url:str,local_dir_path_obj:Path,stem:str) -> Path:
    response = requests.get(zip_url,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    if response.status_code != 200:
        logger.error(f"下载zip文件路径{zip_url}发生异常,状态码:{response.status_code},业务无法继续!!")
        raise RuntimeError(f"下载zip文件路径{zip_url}发生异常,状态码:{response.status_code},业务无法继续!!")

    zip_path_obj = local_dir_path_obj / f"{stem}_result.zip"
    # 把返回写成zip文件
    zip_path_obj.write_bytes(response.content)

    zip_extract_dir_obj =  local_dir_path_obj / stem

    if zip_extract_dir_obj.exists():
        shutil.rmtree(zip_extract_dir_obj)

    zip_extract_dir_obj.mkdir(parents=True,exist_ok=True)
    # 解压
    shutil.unpack_archive(zip_path_obj,zip_extract_dir_obj)
    # 在解压目录中递归查找
    md_file_obj_list = list(zip_extract_dir_obj.rglob("*.md"))
    if not md_file_obj_list or len(md_file_obj_list) == 0:
        logger.error(f"从{zip_url}下载后解压到{zip_extract_dir_obj}目录下没有找到md文件,业务无法继续!!")
        raise RuntimeError(f"从{zip_url}下载后解压到{zip_extract_dir_obj}目录下没有找到md文件,业务无法继续!!")

    for md_file_obj in md_file_obj_list:
        if md_file_obj.stem == stem:
            logger.info(f"解压的文件名为原文件名:{md_file_obj.name}，无需额外处理")
            return md_file_obj

    target_md_obj = None
    for md_file_obj in md_file_obj_list:
        if md_file_obj.name.lower()=="full.md":
            target_md_obj = md_file_obj
            break

    if not target_md_obj:
        target_md_obj = md_file_obj_list[0]
        logger.warning(f"从{zip_url}下载后解压到{zip_extract_dir_obj}目录下没有找到{stem}.md文件和full.md文件，默认选择第一个文件:{target_md_obj.name}!!")

    return target_md_obj.rename(target_md_obj.with_name(stem+".md"))

@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 1. pdf dir路径校验和完善
    pdf_path_obj , local_dir_obj = validate_pdf_paths(state)

    # 2. pdf上传和zip url地址获取
    zip_url = upload_pdf_and_poll(pdf_path_obj)

    # 3. 下载解压并返回md
    md_path_obj = download_and_extract_markdown(zip_url, local_dir_obj, pdf_path_obj.stem)

    # 4. 修改state 状态。md_path
    state['md_content'] = md_path_obj.read_text(encoding="utf-8")
    state['md_path'] = str(md_path_obj)

    print(zip_url)
    return state