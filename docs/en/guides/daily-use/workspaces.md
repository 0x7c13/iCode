# Working directory

The working directory is the default directory iCode uses for file operations and command execution. Before asking the agent to modify files, save any unsaved content to disk and record existing changes with a Git commit. This makes it easier to distinguish the agent's changes and restore the previous state if needed.

## Choose a working directory at startup

In a terminal, enter the project directory and start iCode:

```shell
cd <project-directory>
icode
```

iCode uses the directory it was started in as the working directory.

You can also specify a working directory with `-C` or `--workdir`, without first entering the project directory:

```shell
icode -C <project-directory>
```

## Change the working directory during a session

When the agent is idle, click the current working directory path at the right end of the conversation area's bottom border, or enter `/chdir` in the input field to open the "Change Directory" dialog. Browse to and select the target directory. If you have used the target working directory recently, select it from the "Recent" group on the "Favorites" tab.

If you know the target path, enter the following command in the input field to switch directly:

```text
/chdir <project-directory>
```

`<project-directory>` can be an absolute path or a path relative to the current working directory. If the target does not exist or is not a directory, iCode displays an error and keeps the current working directory.

After the switch, the new working directory path appears at the right end of the conversation area's bottom border. The agent uses the new working directory as the default location for subsequent file reads, commands, and other operations.

When switching working directories, iCode reloads the current agent and loads memory and skills from the new working directory. For configuration details, see [Configure memory](../configuration/memory.md) and [Install and use skills](../extensions/skills.md).

Switching working directories does not create a new session; the current conversation history is preserved. When working on an unrelated project, first enter `/new` to create a new session, then switch working directories in that session. This avoids recording the working directory change in the original session or mixing context from different projects.
