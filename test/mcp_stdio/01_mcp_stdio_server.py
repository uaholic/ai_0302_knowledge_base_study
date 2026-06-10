# 导入 MCP 服务框架 FastMCP（用于构建 AI 可调用的工具/资源/提示词服务）
from mcp.server.fastmcp import FastMCP

# ====================== 1. 创建 MCP 服务实例 ======================
# 实例化一个 MCP 服务，命名为 Demo（用于 AI 识别服务名称）
mcp = FastMCP("Demo")

# ====================== 2. 定义 MCP 工具（Tool） ======================
# 注册工具：AI 可调用的功能函数
@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b

# ====================== 3. 定义 MCP 资源（Resource） ======================
# 注册资源：可通过固定 URI 获取的静态/动态数据
@mcp.resource("greeting://default")
def get_greeting() -> str:
    """获取问候语资源"""
    return "Hello from static resource!"

# ====================== 4. 定义 MCP 提示词（Prompt） ======================
# 注册提示词：AI 可直接调用的结构化提示词模板
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """
    根据风格生成问候语提示词
    :param name: 用户名称
    :param style: 风格：friendly/formal/casual
    :return: 提示词模板
    """
    styles = {
        "friendly": "写一句友善的问候",
        "formal": "写一句正式的问候",
        "casual": "写一句轻松的问候",
    }
    return f"为{name}{styles.get(style, styles['friendly'])}"

# ====================== 5. 启动 MCP 服务 ======================
if __name__ == "__main__":
    # 运行服务，使用 stdio 标准输入输出传输数据（AI 与 MCP 通信最常用方式）
    print(f"本地mcp服务器已经启动!!")
    mcp.run(transport="stdio")
