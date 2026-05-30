from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.entry_service import resolve_input_file

"""
   @node_log("node_entry")  -> 方法进入 和 方法执行完毕 以及异常的日志 
   节点方法(state : ImportGraphState)  
       打印开始的日志
       1. add_running_task(state["task_id"], "node_entry")
       2. state = resolve_input_file(state) 调用节点业务函数 [节点 + 业务的关联]
       3. add_done_task(state["task_id"], "node_entry")
       开营结束的日志
       return state 

   为啥add_running_task / add_done_task: 记录节点的完成状态, 存储到对应task_id的列表中 (此时:还是英文)
   为啥要传入task_id :  因为我们有很多客户端,为每个客户端存储一个对应的列表, 最终存储到dict [task_id, [] ]
   节点名称是固定的么: 是的,因为后续 get的时候使用列表推到式 将英文 转成对应的中文,定义的时候 key 就是节点!

   @node_log("节点名")  -> 方法进入 和 方法执行完毕 以及异常的日志 
   节点方法(state : ImportGraphState)  
       打印开始的日志
       1. add_running_task(state["task_id"], "node_entry")
       2. state = resolve_input_file(state) 调用节点业务函数 [节点 + 业务的关联]
       3. add_done_task(state["task_id"], "node_entry")
       开营结束的日志
       return state 
"""


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """
    add_running_task(state["task_id"], "node_entry")
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")
    return state