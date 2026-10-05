```
   ▄   ▄▄▄▄
   ▀██████▀          █▄
 ▀▀  ██              ██
 ██  ██     ▄███▄ ▄████ ▄█▀█▄
 ██  ██     ██ ██ ██ ██ ██▄█▀
▄██  ▀█████▄▀███▀▄█▀███▄▀█▄▄▄
```

简体中文 | [English](README.md)

> **开发中。** 接口、文件格式和默认设置在不同版本之间仍会变化。

## 什么是 iCode？

让截图来说明一切。

> **请使用现代终端。** iCode 在现代终端模拟器中效果最佳，例如 Windows 上的 [Windows Terminal](https://github.com/microsoft/terminal)，或 macOS 和 Linux 上的 [Ghostty](https://ghostty.org)。

### 首页

智能体、模型、审批模式和会话一目了然；在输入栏中只需一个按键，即可调出智能体（`#`）、模型（`$`）、命令（`/`）、文件（`@`）和 Shell（`!`）。

![iCode 启动界面，已选中 Code 智能体，对话侧边栏为空](screenshots/homepage.png)

### 对话

回答在终端中以富 Markdown 形式渲染，图表也不例外。

![一条对话回答，渲染了一张实体图和一个格式化的项目符号列表](screenshots/chat.png)

### 工作流

逐个节点观察工作流的运行，并查看每个智能体节点的模型、工具调用和 token 消耗。

![工作流模式正在运行演示工作流，显示运行中途的节点图](screenshots/workflow.png)

### 智能体

智能体是配置，而不是代码：指令、工具、子智能体、Skills、MCP 服务器和记忆，都在同一处编辑（**F2**）。

![“智能体配置”界面正在编辑内置的 Code 智能体](screenshots/agents.png)

### 模型

来自不同提供商的模型配置并列管理，并可在对话中途切换（**F4**）。

![“模型配置”界面正在编辑一个使用 Responses API 的 DeepSeek 配置](screenshots/models.png)

### 差异

智能体修改过的每个文件，既可按轮次查看，也可一次性全部查看，支持统一视图和分栏视图。

![差异查看器，显示变更树和一个 Python 文件的统一差异](screenshots/diff.png)

### 回滚

选择一个轮次，检查将被还原的文件修改，然后丢弃该轮次之后的对话。

![回滚对话框，列出所选轮次之后将被还原的文件修改](screenshots/rollback.png)

### 轨迹

时间和 token 都花在了哪里（**F12**）：模型、工具与等待各占多少，逐轮次的明细，以及 token 和缓存用量。

![轨迹概览，显示时间、token 和并行度的分解](screenshots/trajectory-overview.png)

一个轮次中的每次模型调用、钩子和工具调用，都按时间线排布。

![轨迹时间线，显示一个轮次的模型调用周期、钩子和工具调用](screenshots/trajectory-timeline.png)

### Shell

Shell（`!`）是一个真正的终端，因此全屏程序可以原地运行，每次按键都会转发给它们：图中是运行在 iCode 里的 iCode 里的 iCode 中的 `htop`。

![三层嵌套的 iCode Shell，最内层正在运行 htop](screenshots/shell.png)

## 如何运行

[uv](https://github.com/astral-sh/uv) 是唯一的前置条件，它会自动准备 Python 3.14。在代码检出目录中执行：

```bash
uv sync --extra all        # 不要用裸 `uv sync`，也不要用 `--all-extras`
./scripts/fetch_rg.sh      # 下载随附的 ripgrep；Windows：.\scripts\fetch_rg.ps1
uv run icode               # `uv run chrys` 启动的是同一个程序（`Chrys` 是我们的代号）
```

iCode 不附带任何模型配置，因此首次启动时请按 **F4** 添加一个。

## 用户指南

在 iCode 中按 **F8** 即可打开内置的用户指南，其中完整介绍了 iCode 的使用方法。相同的内容也位于 [`docs/`](docs/) 目录，提供 [English](docs/en/start/what-is-icode.md) 和 [简体中文](docs/zh-Hans/start/what-is-icode.md) 两个版本。

## 隐私

iCode 不收集你的数据。它没有遥测、分析或崩溃报告功能，默认情况下也不会向我们或任何其他第三方发送任何内容，包括使用数据。iCode 自身发送的数据只会发往你自己配置的目标：你添加的模型提供商、智能体使用的 MCP 服务器，以及你设置的钩子或 OpenTelemetry 导出。网络工具默认关闭；如果你为某个智能体开启了网络工具，它的搜索请求会发往你配置的搜索服务，未配置时发往 Exa 的公开搜索服务（`mcp.exa.ai`），读取的网页则直接向对应网站请求。你或智能体运行的 Shell 命令、Skill 脚本和工作流本身就是程序，拥有你的网络访问权限，它们发送什么由它们自己决定。你的数据只属于你，我们尊重你的隐私。

## 监管合规

iCode 仅是一款用于编排智能体和工作流的工具。它不包含也不提供任何 AI 模型；它使用的每个模型都由你自行接入。当你为特定业务或工程用途将 AI 模型集成到 iCode 中时，你需自行承担该用途在欧盟《人工智能法案》（Regulation (EU) 2024/1689）及任何其他适用法律或监管框架下的全部合规义务。

## 许可证

iCode 基于 [Apache License 2.0](LICENSE) 开源。iCode 还包含并随附了第三方软件，它们各自遵循其自身的许可证；[NOTICE](NOTICE) 列出了这些软件及其版权声明和许可条款。
