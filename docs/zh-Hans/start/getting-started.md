# 开始使用 iCode

本教程介绍如何安装 iCode、配置模型，并在项目中使用智能体完成一次开发任务。

## 1. 安装 iCode

iCode 为 macOS、Linux 和 Windows 提供预构建的离线安装包。安装包已包含 Python 和运行依赖，无需另行安装 Python。

请从 [iCode Releases](https://github.com/0x7c13/chrys/releases) 下载最新版本。当前不要运行 `pip install chrys`：PyPI 上的这个名称属于另一个项目。

以下命令中的 `<version>` 和 `<architecture>` 是占位符，请替换为下载文件名中的实际值。

### macOS

根据处理器选择 `icode-macos-aarch64-v<version>-offline.tar.gz`（Apple 芯片）或 `icode-macos-x86_64-v<version>-offline.tar.gz`（Intel），然后在下载目录运行：

```bash
tar xzf icode-macos-<architecture>-v<version>-offline.tar.gz
chmod +x ./icode
./icode install
```

### Linux

根据处理器选择 `icode-linux-x86_64-v<version>-offline.tar.gz` 或 `icode-linux-aarch64-v<version>-offline.tar.gz`，然后在下载目录运行：

```bash
tar xzf icode-linux-<architecture>-v<version>-offline.tar.gz
chmod +x ./icode
./icode install
```

Linux x86-64 安装包要求 glibc 2.17 或更高版本，ARM64 安装包要求 glibc 2.18 或更高版本。Alpine Linux 等只提供 musl 的发行版不在预构建包的支持范围内。

### Windows

下载 `icode-windows-x86_64-v<version>-offline.zip` 并解压，在 PowerShell 中进入解压目录，然后运行：

```powershell
.\icode.exe install
```

安装器会尝试将 iCode 加入当前用户的 `PATH`。安装完成后，请打开一个新终端。

### 验证安装

```shell
icode --version
```

命令应输出 iCode 的版本号。在 macOS 或 Linux 上，如果系统找不到 `icode`，请按安装器的提示将 `~/.local/bin` 加入 `PATH`；如果安装器提示 `~/.local/bin/icode` 已存在，该命令会启动其他程序，请改用 `chrys` 命令。在 Windows 上请打开新的终端；如果安装器提示更新用户 PATH 失败，请手动将 `%LOCALAPPDATA%\chrys\bin` 加入用户 `PATH`。

## 2. 在项目中启动 iCode

建议在使用 Git 管理或能够安全恢复的项目中开始本教程。在让智能体修改文件前，请先保存尚未写入磁盘的内容，并使用 Git commit 记录已有变更，以便区分智能体所做的修改，并在需要时恢复到修改前的状态。

在终端中进入要处理的项目目录并启动 iCode：

```shell
cd <project-directory>
icode
```

将 `<project-directory>` 替换为项目目录的实际路径。启动后如需切换目录，请参阅[工作目录](../guides/daily-use/workspaces.md)。

## 3. 配置模型

1. 按 **F4** 或点击屏幕底部的 `f4 模型`，打开“模型配置”窗口。
2. 点击“新建”。
3. 填写自定义的配置名称，例如 `我的模型`。
4. 在模型选项区域，选择模型提供商和 API 样式。仅 OpenAI 和 DeepSeek（OpenAI）提供 API 样式选项。配置前，请根据模型服务商的文档确认其支持的 API，并按文档建议选择 `Chat Completions` 或 `Responses`。如果服务商仅支持其中一种 API，请选择对应的 API 样式；如果两种 API 均支持，则根据服务商的建议或实际使用需求选择。
5. 填写模型 ID，并根据模型服务的要求填写服务地址和 API 密钥。这些字段名称以所选提供商开头，例如“OpenAI 模型”。
6. 根据模型提供商的说明，填写最大上下文窗口和最大输出词元数。
7. 保存配置并关闭配置窗口。若此前没有可用的模型配置，该配置将自动成为当前使用的模型配置。

> **提示**
>
> 直接填写的 API 密钥会随模型配置文件保存在本机。若不希望将密钥写入配置文件，请[通过环境变量提供 API 密钥](../guides/configuration/models.md#通过环境变量提供-api-密钥)。

### 验证模型连接

在输入栏中提交：

```text
你好
```

iCode 应正常返回问候回复。若模型服务返回错误，请重新检查模型 ID、服务地址、API 密钥等模型配置。

## 4. 使用智能体

### 选择智能体

iCode 默认提供两个主智能体：Q&A Agent（问答智能体）和 Code Agent（代码智能体）。两者都可以阅读项目、分析代码和讨论方案，但定位和可用工具不同：Q&A Agent 主要提供只读能力，适合了解项目、解释代码和分析问题；Code Agent 则可以修改文件、运行命令，适合编写、审查、重构、调试和验证代码。

当前没有正在运行的智能体任务时，可以切换智能体。在输入栏中输入 `#`，输入栏上方会显示智能体列表，可以通过鼠标点击，或使用上下方向键并按回车进行选择。也可以点击输入栏上方状态栏中的智能体名称，在智能体选择窗口中进行切换。

### 使用 Q&A Agent

Q&A Agent 适合在**不修改项目的情况下**了解项目、分析代码、追踪实现和讨论方案。它会基于项目内容给出回答，并尽量提供可追溯的依据。

切换到 **Q&A Agent** 后，在输入栏中输入 `/new` 并回车，开始新会话。

以下示例假设当前项目是一个待办事项应用，仅用于展示使用方式。实际使用时，请根据实际项目调整问题和任务。

```text
请分析任务列表的实现，并为增加“清除已完成任务”功能提出方案。说明需要修改哪些文件以及实现步骤。
```

Q&A Agent 可能会搜索文件、读取文件内容或运行只读命令。工具调用会以卡片形式显示，完成分析后会给出实现方案。

### 使用 Code Agent

Code Agent 适用于需要**修改文件、运行命令和验证结果**的任务。它会先阅读相关代码，再按任务要求实现、调试或重构代码。

Q&A Agent 给出实现方案后，可以在同一会话中切换到 **Code Agent**。切换智能体不会清除当前会话的上下文，Code Agent 可以基于已有方案继续完成开发任务：

```text
请根据刚才讨论的方案实现“清除已完成任务”功能。修改代码并运行必要的测试。
```

如果出现审批对话框，请检查即将执行的工具调用，尤其是命令内容、目标文件和影响范围，确认符合预期后再批准。有关不同审批模式的行为，请参阅[配置审批模式](../guides/configuration/approval.md)。

任务完成后，检查智能体的最终回复，并通过 `/diff` 浏览当前会话记录的文件修改。随后使用 Git 等版本控制工具检查工作目录的完整变更，并确认功能正确、必要的测试已经通过且不包含无关改动。确认无误后，可以提交这些变更。

### 阅读和复制回复

所有智能体的回复都支持 Markdown、数学公式和 Mermaid 图表。你自己发送的消息中的公式不作数学渲染。

列表和引用可以包含标题、表格、代码和公式。

回复中，`$...$` 或 `\(...\)` 包围的公式以单行显示；`math` 代码块、`$$...$$` 或 `\[...\]` 中的块公式可显示分式、矩阵和对齐公式。普通文字、价格和 shell 变量保留原有 Markdown 格式。不支持或过宽的公式会完整显示为可换行的源码。

例如，`$x^{n+1}$` 显示为 xⁿ⁺¹，`$90^\circ$` 显示为 90°。

Mermaid 代码块会显示为图表，点击**打开完整图表**可查看、滚动完整内容。部分图表会简化显示并提示，不支持的图表保留源码。

鼠标选择复制显示结果，不会因屏幕折行额外插入换行。消息标题旁的**复制**保留原始 Markdown 和 LaTeX。重新打开旧会话时，也会使用当前的渲染效果。

## 5. 结束本次使用

若要结束本次使用，输入 `/exit` 或按 `Ctrl+Q` 退出 iCode。

## 下一步

至此，已经完成 iCode 的安装和模型配置，并了解了 Q&A Agent 和 Code Agent 的基本使用方式，以及如何在同一会话中配合完成开发任务。

接下来，可以继续探索：

- [配置智能体](../guides/configuration/agents.md)，创建或修改智能体，并设置其指令、模型、工具和子智能体
- [配置模型](../guides/configuration/models.md)，连接并切换不同的模型服务
- [工作目录](../guides/daily-use/workspaces.md)
- [配置审批模式](../guides/configuration/approval.md)
- 通过 [Skills](../guides/extensions/skills.md) 和 [MCP](../guides/extensions/mcp.md) 扩展 iCode
