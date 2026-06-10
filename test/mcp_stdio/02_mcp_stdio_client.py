# 安装命令：MCP（模型控制协议）客户端依赖库
# pip install mcp

# 导入异步IO库（Python异步必须用async/await）
import asyncio
# 导入MCP客户端：通过stdio（标准输入输出）连接MCP服务端
from mcp.client.stdio import stdio_client
# 导入MCP核心会话类：管理与服务端的连接、工具调用、资源获取
from mcp import ClientSession, StdioServerParameters

# 异步主函数：通过stdio方式连接并调用MCP服务端
async def stdio_run():
    # 配置MCP服务端启动参数（告诉客户端如何启动/连接服务端）
    server_params = StdioServerParameters(
        # 启动服务端使用的Python解释器路径（必须是服务端所在虚拟环境）
        command=r"/Users/gyq/PycharmProjects/ai_0302_knowledge_base/.venv/bin/python",
        # 要运行的MCP服务端文件（告诉Python要执行哪个文件）
        #args=[r"./mcp_server_stdio.py"],
        args=[r"./01_mcp_stdio_server.py"],
    )

    # ------------------- 1. 建立与MCP服务端的stdio通信通道 -------------------
    # stdio_client：启动子进程运行服务端，返回 读/写 管道
    async with stdio_client(server_params) as (read, write):
        # ------------------- 2. 创建MCP客户端会话（封装连接逻辑） -------------------
        async with ClientSession(read, write) as session:
            # 初始化MCP协议连接（必须调用，完成握手）
            await session.initialize()

            # ------------------- 3. 获取服务端提供的所有工具 -------------------
            # 作用：让客户端/大模型知道服务端有哪些工具可以调用
            tools = await session.list_tools()
            print("=== 可用工具列表 ===")
            print(tools)
            print()
            
            # ------------------- 4. 调用MCP服务端的工具（核心功能） -------------------
            # 参数1：工具名（与服务端@mcp.tool定义一致）
            # 参数2：参数字典（严格匹配服务端工具参数名）
            call_res = await session.call_tool("add", {"a": 1, "b": 2})
            print("=== 调用工具结果 ===")
            print(call_res)
            print()

            # ------------------- 5. 获取服务端提供的所有资源 -------------------
            # 资源：可通过URI读取的静态/动态数据
            resources = await session.list_resources()
            print("=== 可用资源列表 ===")
            print(resources)
            print()

            # ------------------- 6. 读取MCP服务端的资源内容 -------------------
            # 参数：资源URI（与服务端@mcp.resource定义一致）
            read_res = await session.read_resource("greeting://default")
            print("=== 读取资源结果 ===")
            print(read_res)
            print()

            # ------------------- 7. 获取服务端提供的所有提示词模板 -------------------
            # 提示词模板：大模型可直接使用的结构化prompt
            prompts = await session.list_prompts()
            print("=== 可用提示词模板 ===")
            print(prompts)
            print()

            # ------------------- 8. 获取并渲染MCP服务端的提示词模板 -------------------
            # 参数1：提示词名（与服务端@mcp.prompt定义一致）
            # 参数2：模板参数（服务端定义的name、style等）
            get_res = await session.get_prompt("greet_user", {"name": "Jack"})
            print("=== 获取提示词结果 ===")
            print(get_res)
            print()

# 运行异步主函数（Python固定写法：启动异步事件队列）
asyncio.run(stdio_run())