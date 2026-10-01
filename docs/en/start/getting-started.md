# Getting started with iCode

This tutorial shows how to install iCode, configure a model, and use agents to complete a development task in a project.

## 1. Install iCode

iCode provides prebuilt offline installation packages for macOS, Linux, and Windows. The packages include Python and runtime dependencies, so you do not need to install Python separately.

Download the latest version from [iCode Releases](https://github.com/0x7c13/chrys/releases). Do not run `pip install chrys` at this time: that name belongs to a different project on PyPI.

In the commands below, `<version>` and `<architecture>` are placeholders. Replace them with the actual values in the downloaded filename.

### macOS

Choose `icode-macos-aarch64-v<version>-offline.tar.gz` for Apple silicon or `icode-macos-x86_64-v<version>-offline.tar.gz` for Intel, then run the following in your download directory:

```bash
tar xzf icode-macos-<architecture>-v<version>-offline.tar.gz
chmod +x ./icode
./icode install
```

### Linux

Choose `icode-linux-x86_64-v<version>-offline.tar.gz` or `icode-linux-aarch64-v<version>-offline.tar.gz` for your processor, then run the following in your download directory:

```bash
tar xzf icode-linux-<architecture>-v<version>-offline.tar.gz
chmod +x ./icode
./icode install
```

The Linux x86-64 package requires glibc 2.17 or later, and the ARM64 package requires glibc 2.18 or later. Distributions that provide only musl, such as Alpine Linux, are not supported by the prebuilt packages.

### Windows

Download and extract `icode-windows-x86_64-v<version>-offline.zip`. Open PowerShell, go to the extracted directory, and run:

```powershell
.\icode.exe install
```

The installer attempts to add iCode to the current user's `PATH`. Open a new terminal after installation.

### Verify the installation

```shell
icode --version
```

The command should print the iCode version number. If your system cannot find `icode` on macOS or Linux, add `~/.local/bin` to `PATH` as the installer instructs. If the installer reported that `~/.local/bin/icode` already exists, that command starts another program; use `chrys` instead. On Windows, open a new terminal; if the installer reported that updating the user PATH failed, add `%LOCALAPPDATA%\chrys\bin` to your user `PATH` manually.

## 2. Start iCode in a project

Start this tutorial in a project managed with Git or one that you can safely restore. Before asking an agent to modify files, save any unsaved content and record existing changes with a Git commit. This lets you distinguish the agent's changes and restore the previous state if needed.

In a terminal, go to the project directory and start iCode:

```shell
cd <project-directory>
icode
```

Replace `<project-directory>` with the actual path to your project. To switch directories after startup, see [Working directory](../guides/daily-use/workspaces.md).

## 3. Configure a model

1. Press **F4** or click `f4 Models` at the bottom of the screen to open the **Model Configuration** dialog.
2. Click **New**.
3. Enter a **Profile Name** of your choice, such as `My Model`.
4. In the **Model Options** section, select the **Provider** and **API Style**. Only OpenAI and DeepSeek (OpenAI) provide an API Style option. Before configuring it, check the model service's documentation for supported APIs and follow its recommendation when choosing **Chat Completions** or **Responses**. If the service supports only one API, select its corresponding style; if it supports both, choose according to the service's recommendation or your needs.
5. Enter the model ID in the **Model** field, and fill in the **Base URL** and **API Key** fields as required by the model service. These field names start with the selected provider, for example **OpenAI Model**.
6. Set **Max Context Window** and the **Max Output Tokens** field according to the model provider's documentation.
7. Save the profile and close the dialog. If no model profile was previously available, this profile automatically becomes the active model profile.

> **Tip**
>
> An API key entered directly is stored locally in the model profile file. To avoid writing the key to that file, [provide the API key through an environment variable](../guides/configuration/models.md#provide-an-api-key-through-an-environment-variable).

### Verify the model connection

Submit the following in the input field:

```text
Hello
```

iCode should reply with a greeting. If the model service returns an error, check the model profile again, including the model ID, base URL, and API key.

## 4. Use agents

### Select an agent

iCode provides two main agents by default: Q&A Agent and Code Agent. Both can read the project, analyze code, and discuss approaches, but they have different roles and tools. Q&A Agent primarily provides read-only capabilities for understanding projects, explaining code, and analyzing problems. Code Agent can modify files and run commands, making it suitable for writing, reviewing, refactoring, debugging, and validating code.

You can switch agents when no agent task is running. Type `#` in the input field to display the agent list above it. Select an agent by clicking it, or use the up and down arrow keys and press Enter. You can also click the agent name in the status bar above the input field to switch agents in the agent selection dialog.

### Use Q&A Agent

Q&A Agent helps you understand a project, analyze code, trace implementations, and discuss approaches **without modifying the project**. It answers based on the project contents and aims to provide evidence you can trace back to its source.

After switching to **Q&A Agent**, type `/new` in the input field and press Enter to start a new session.

The following examples assume the current project is a to-do app and illustrate how to use the agents. Adapt the questions and tasks to your own project.

```text
Analyze the task list implementation and propose a plan for adding a "Clear completed tasks" feature. Explain which files need to change and the implementation steps.
```

Q&A Agent may search for files, read their contents, or run read-only commands. Tool calls appear as cards. After completing its analysis, the agent presents an implementation plan.

### Use Code Agent

Code Agent handles tasks that require **modifying files, running commands, and validating results**. It reads the relevant code first, then implements, debugs, or refactors it as requested.

After Q&A Agent provides an implementation plan, you can switch to **Code Agent** in the same session. Switching agents preserves the current session's context, so Code Agent can continue the development task using the existing plan:

```text
Implement the "Clear completed tasks" feature using the plan we just discussed. Modify the code and run the necessary tests.
```

If an approval dialog appears, review the tool call before approving it. Pay particular attention to the command, target files, and scope of its effects, and approve only when they match your expectations. For the behavior of each approval mode, see [Configure approval modes](../guides/configuration/approval.md).

When the task is complete, review the agent's final response and use `/diff` to browse the file changes recorded in the current session. Then use Git or another version control tool to inspect the complete working directory changes. Confirm that the feature works correctly, the necessary tests pass, and there are no unrelated changes. Once you have verified the changes, you can commit them.

### Read and copy replies

All agents' replies support Markdown, formulas and Mermaid diagrams. Formulas in your own messages are not rendered as math.

Lists and quotes can contain headings, tables, code and formulas.

Single-line math is written with `$...$` or `\(...\)`. Display formulas in `math` code blocks, `$$...$$` or `\[...\]` can show fractions, matrices and aligned equations. Ordinary text, prices and shell variables keep their Markdown formatting. Formulas that are unsupported or too wide are shown as complete, wrapped source.

For example, `$x^{n+1}$` displays as xⁿ⁺¹ and `$90^\circ$` as 90°.

Mermaid code blocks appear as diagrams; choose **Open full diagram** to view and scroll the full result. Some diagrams are simplified with a notice; unsupported diagrams stay as source.

Mouse selection copies what is displayed, without adding newlines for screen wrapping. Use **copy** beside the message header to copy the original Markdown and LaTeX. Reopening an older conversation uses the current rendering.

## 5. End the session

To finish using iCode, type `/exit` or press `Ctrl+Q` to exit.

## Next steps

You have installed iCode, configured a model, and learned the basics of Q&A Agent and Code Agent, including how to use them together in the same session to complete a development task.

You can continue exploring:

- [Configure agents](../guides/configuration/agents.md) to create or modify agents and set their instructions, models, tools, and sub-agents
- [Configure models](../guides/configuration/models.md) to connect to and switch between model services
- [Working directory](../guides/daily-use/workspaces.md)
- [Configure approval modes](../guides/configuration/approval.md)
- Extend iCode with [Skills](../guides/extensions/skills.md) and [MCP](../guides/extensions/mcp.md)
