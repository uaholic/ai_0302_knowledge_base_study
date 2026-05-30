import copy
from typing import TypedDict


class ImportGraphState(TypedDict):
    task_id: str  # 每次调用流程的标识

    # 文件状态判断
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    local_file_path: str  # 存储要解析文件的地址 pdf md
    local_dir: str  # 存储生成的md文件 pdf->md
    md_path: str  #
    pdf_path: str
    file_title: str

    md_content: str
    item_name: str
    chunks: list  # 存储切块内容
    embeddings_content: list  # 存储带有向量的切块内容


# 提供下对外快速创建的方法
default_state: ImportGraphState = {
    "task_id": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "local_file_path": "",
    "local_dir": "",
    "md_path": "",
    "pdf_path": "",
    "file_title": "",
    "md_content": "",
    "item_name": "",
    "chunks": [],
    "embeddings_content": []
}

# 提供一个方法, 可以返回我们state 并且可以根据传入参数进行对象的属性修改
# 1. 方法() -> default_state 2. 方法(参数) -> default_state (task_id = 传入参数) -> default_state
# 方法(task_id=007, local_file_path="./md.pdf")
def create_default_state(**overriders) -> ImportGraphState:
    """
    :param overriders: 传入的参数 key = x  key = x 转成字典, 方便调用update方法修改
    :return: 每次返回是基于模板创建的新的字典对象
    """

    # copy【深 和 浅 copy】
    # 深 copy.deepcopy
    # 浅 copy.copy | dict(字典) | 字典.copy()
    copy_state = copy.deepcopy(default_state)

    # 更新
    # ** {task_id:xx, local_file_path=} -> 解构 -> task_id = x, local_file_path=
    # **overriders -> task_id = x, local_file_path= -> {task_id:xx, local_file_path=}
    copy_state.update(overriders)

    return copy_state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(default_state)