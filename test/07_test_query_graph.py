
from app.process.query.agent.main_graph import query_graph_app
import json
from app.process.query.agent.state import create_query_default_state
from app.shared.runtime.logger import  logger

# 执行 动态测试
state = create_query_default_state(
    session_id = "session_007",
    original_query="中午吃鸡公煲好嘛!",
    is_stream=False
)
logger.info(f"开始执行,执行参数为:{state}")
result_state = query_graph_app.invoke(state)
logger.info(f"执行结束,执行结果为:{result_state}")

# 静态测试 获取跳转结果 图结构
query_graph_app.get_graph().print_ascii()