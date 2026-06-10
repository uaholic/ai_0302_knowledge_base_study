from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

# https://www.modelscope.cn/mcp/servers/@amap/amap-maps
# 61a062b000cb2b88dd40a2c932a92cca

# ====================== MCP 多服务客户端配置 ======================
# MultiServerMCPClient：LangChain 官方 MCP 适配器
# 作用：同时连接【远程 MCP 服务】 + 【本地 MCP 服务】，自动合并所有工具
client = MultiServerMCPClient(
    {
        # ====================== 远程 MCP 服务（通过 HTTP 连接） ======================
        "amap-maps": {
            # 传输方式：streamable_http = 长连接 HTTP 模式（适合远程 MCP 服务）
            "transport": "streamable_http",
            # 远程 MCP 服务的地址（必须） [私有化地址 已经和我的key绑定了]
            "url": "https://mcp.api-inference.modelscope.net/86f2ed7e664c48/mcp",

            # -------- 以下是 HTTP 模式【可选属性】 --------
            # "headers": {"Authorization": os.getenv("MCP_API_KEY")},  # 请求头：鉴权、Token
            # "timeout": 30,                                           # 超时时间（秒）
            # "proxy": "http://127.0.0.1:7890",                        # 代理配置
        },

        # ====================== 本地 MCP 服务（通过 stdio 启动子进程） ======================
        "local-demo": {
            # 传输方式：stdio = 标准输入输出（本地启动 Python 脚本最常用）
            "transport": "stdio",
            # 启动命令：使用 python 解释器
            "command": "python",
            # 命令参数：要运行的 MCP 服务端脚本路径（必须正确）
            "args": [r"/Users/gyq/PycharmProjects/ai_0302_knowledge_base/test/mcp_stdio/01_mcp_stdio_server.py"],

            # -------- 以下是 stdio 模式【可选属性】 --------
            # "env": {"XXX": "YYY"},        # 给服务端传递环境变量
            # "cwd": r"C:\workdir"          # 服务端运行的工作目录
            # "shell": True                 # 是否通过系统 shell 运行
        }
    }
)


# ====================== 消息打印工具函数 ======================
# 作用：格式化打印用户消息 / AI 消息 / 工具调用消息
# 方便调试查看 Agent 完整思考流程
def print_message(message):
    # 动态导入消息类型（避免全局依赖）
    from langchain.messages import AIMessage, HumanMessage, ToolMessage

    # AI 回复消息：包含回答内容 + 工具调用决策
    if isinstance(message, AIMessage):
        print("AI回复：", message.content)
        print("AI决定调用工具：", message.tool_calls)

    # 用户输入消息
    elif isinstance(message, HumanMessage):
        print("用户输入：", message.content)

    # 工具执行结果消息
    elif isinstance(message, ToolMessage):
        print("工具调用：", message.content)

    # 未识别的消息类型
    else:
        print("未知消息类型")


# ====================== 主函数：MCP + LangChain Agent 交互对话 ======================
async def main():
    # 从所有 MCP 服务获取工具
    # 自动把 MCP Tool → LangChain Tool，可直接给 Agent 使用
    tools = await client.get_tools()
    for tool in tools:
        print("工具：", tool)

    # 初始化大模型（Agent 的大脑）
    llm = ChatOpenAI(
        model=os.getenv("QWEN_MODEL_NAME")  # 从环境变量读取模型名称
    )

    # 创建 Agent 智能体
    # 自动具备：思考 → 选工具 → 调用 → 回答 的完整能力
    agent = create_agent(
        model=llm,
        tools=tools,
    )

    # 无限循环对话（控制台交互）
    while True:
        user_input = input("> ")

        # 输入 exit 退出对话
        if user_input == "exit":
            break

        # 异步调用 Agent
        # thread_id="1"：会话ID，用于记忆多轮对话
        res = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "1"}}
        )

        # 打印最终回答
        print("\n===== 最终回答 =====")
        print(res['messages'][-1].content, end="\n\n")

        # 打印完整消息流（用户 → AI → 工具 → AI）
        print("===== 完整对话流程 =====")
        for message in res["messages"]:
            print_message(message)
            print("-" * 50)


# 启动异步程序（Python 固定写法）
asyncio.run(main())