# Configure tools

Built-in tools let agents access files, search content, execute shell commands, and use other iCode features. This guide explains how to enable or disable built-in tool kinds for an agent in the terminal user interface (TUI).

Tools provided by MCP servers are not built-in tools and must be managed separately in [MCP configuration](../extensions/mcp.md).

For the tool kind values and individual tool names used in agent configuration and hooks, see [Tool kinds and names](../../reference/tool-kinds-and-names.md).

## Open tool configuration

Enter `/agents tools` in the input field to open the current agent's "Tools" tab. To configure another agent, select it from the list on the left.

## Select built-in tools

Each option corresponds to a kind of built-in tool. Enable the kinds the agent needs for its tasks:

| Tool kind | Available operations | When to enable |
| --- | --- | --- |
| Filesystem Read | Read files and view images (PNG, JPEG, or WebP) | Enable when the agent needs to analyze project files; the agent can use the image viewing tool only if the selected model supports image input |
| Filesystem Write | Create, overwrite, and precisely edit files | Enable only when the agent needs to modify files |
| File search | Search file contents or find files by name pattern | Enable when the agent needs to locate code and files in a project |
| Web search | Find online URLs and summaries; uses Exa (exa.ai) unless the agent profile names other providers | Enable when the agent needs current external information, such as documentation or release details. Queries go to that outside service and can carry details from your conversation, so mind your data and privacy |
| Web fetch | Read the text of a page by URL | Enable when the agent needs to read pages it finds or that you point it to |
| Shell | Execute terminal commands in a subprocess | Enable when the agent needs to run builds, tests, or other commands |
| Ask User | Request additional information or ask the user to make a choice during a task | Suitable for TUI and ACP sessions; headless CLI mode (`icode run`) does not provide this tool to the agent |
| Sleep | Pause a task for up to 3600 seconds; the TUI displays a countdown that can be skipped | Enable when the agent needs to wait before continuing a task |
| Document Converter (PDF, Office) | Convert PDF, DOCX, PPTX, XLSX, and XLS files to Markdown; if the current model supports image input and the agent has "Filesystem Read" enabled, images can also be extracted from PDF, DOCX, and PPTX files for the agent to view; XLSX and XLS conversion includes only text and tables | Enable when the agent needs to read these document formats |
| Todo List | Track tasks with multiple steps using a live checklist | Suitable for tasks with many steps that need visible progress tracking |

Check the kinds to enable, uncheck those you do not need, then click "Save".

If you modify the agent used by the current session, the new tool configuration takes effect starting with subsequent requests.

Web search is off in every agent shipped with iCode, and it is separate from File search, which only searches local files. Web fetch is a separate switch. With no search provider configured, queries go to Exa's public search service. Before you turn it on, see [Configure web tools](./web-tools.md) for where requests go, approval, and network limits.

## Use the Ask User tool

The Ask User tool requests additional information or asks the user to make a choice during a task. A single request can contain 1 to 5 questions, including single-select, multi-select, and free-text questions.

By default, questions appear in an ask-user dialog. If the dialog obscures conversation content you need to review, click **Answer Inline** to show the current question in the conversation instead; this applies only to the current question. If you want future questions to appear in the conversation by default, enable **Show questions in the chat** in [Configure iCode settings](./settings.md).

In the TUI, multiple questions appear in the same question dialog. You can switch between question tabs, fill in the answers, and review all answers on the summary page before submitting them together.

The wait time for an answer is controlled by the **Ask-user timeout (seconds)** setting in [Configure iCode settings](./settings.md).

## Configure tools as needed

Tool configuration determines which capabilities an agent can use. For a specialized agent, enable only the tool kinds needed to complete its tasks.

For example:

- An agent that only analyzes code can enable "Filesystem Read" and "File search" without enabling "Filesystem Write" or Shell.
- An agent that modifies code and runs tests typically needs "Filesystem Read", "Filesystem Write", "File search", and Shell.
- When working with an untrusted project, use caution when enabling Shell and "Filesystem Write".

Enabling a tool only makes that capability available to the agent; operations that require approval are still handled according to the current approval mode. For details on approval requests and how to confirm them, see [Configure approval modes](./approval.md).

For filesystem, search, and Shell tools, relative paths are resolved against the current [working directory](../daily-use/workspaces.md), and shell commands run in the current working directory by default. Switching working directories changes the base for these relative paths and the default working directory for commands. The working directory is the default location for operations; it does not restrict tools to paths within it. Whether an operation can run also depends on its tool arguments, whether approval is granted when required, and the permissions of the system user running iCode.

When a command prints a lot, the agent sees the beginning and the end of its output, and iCode saves the output in the session directory when it can. If a command prints more than 32 MiB, iCode keeps only the first and last parts; the middle is not saved anywhere, and the result tells the agent how much was left out.
