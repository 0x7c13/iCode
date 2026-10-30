# Configure shortcuts

A shortcut is a prompt you save once as a Markdown file and reuse by typing `/` and its name. Choosing it fills the prompt into the input field, where you can edit it before you send it. A shortcut can also take arguments, switch the agent or model for the session, and fill in the output of shell commands when it is sent.

## Where shortcut files go

| Folder | macOS and Linux | Windows | Loaded |
| --- | --- | --- | --- |
| Your shortcuts | `~/.chrys/shortcuts` | `%APPDATA%\chrys\shortcuts` | On by default |
| The project's shortcuts | `.chrys/shortcuts` in the working directory and its parent directories up to the repository root | Same | On by default |
| Other folders you add | Any folder, such as one shared by your team | Same | Once added |

Choose which folders are read under **Settings → Shortcuts** (`/settings shortcuts`); see [Shortcuts settings](../configuration/settings.md#shortcuts). Changes apply the next time you type `/`.

A project's `.chrys/shortcuts` must stay inside the repository: if it is a link to a folder outside it, iCode skips it. To use a folder shared by your team, add it under **Settings → Shortcuts** instead.

The file name is the shortcut's name: `review.md` is `/review`. A file in a subfolder keeps the folder in its name: `git/commit-msg.md` is `/git/commit-msg`. When two folders hold a shortcut with the same name, the first one read is used: your own shortcuts, then the folders you added, then the nearest project folder.

To see which shortcuts are loaded and where each comes from, enter `/runtime` and open the **Shortcuts** tab.

## Write a shortcut

A shortcut file is Markdown with optional settings at the top, between `---` lines:

```text
---
description: Review my uncommitted changes
agent: QA
model: my-provider/my-model
---

Review these changes and point out bugs first:

!`git diff --stat`

Focus on $ARGUMENTS.
```

| Setting | Effect |
| --- | --- |
| `description` | Shown next to the name in the `/` list. |
| `agent` | The agent to switch to when the prompt is sent. |
| `model` | The model to switch to when the prompt is sent: a model profile's name, or `provider/model`. |

Everything after the settings is the prompt. The format is the one opencode uses for its commands, so a file copied from `.opencode/commands` into a shortcuts folder works unchanged.

In the prompt you can use:

- `$ARGUMENTS`: everything typed after the name.
- `$1`, `$2`, …: one argument each, split at spaces; quote an argument that contains spaces (`"the meeting moved"`). The highest-numbered one takes all the remaining arguments.
- `` !`command` ``: runs `command` in the working directory when the prompt is sent, and puts its output in its place.

## Use a shortcut

1. Type `/` in the input field. Shortcuts are listed in the **Prompt Shortcuts** group.
2. Choose one, or type its name with arguments (`/review error handling`) and press **Tab** or **Enter**.
3. The prompt fills the input field. Arguments you did not give are highlighted: press **Tab** to move between them and type to fill them in. If you press **Enter** while some are still empty, iCode asks once before sending.
4. Edit the prompt if you like, then press **Enter** to send it.

### Commands in the prompt

When you send the prompt, each `` !`command` `` in the input field runs first and its output replaces it. A command may run for up to 10 seconds, and up to 16 KB of its output is kept. While the commands run you can press **Esc** to cancel. If a command fails to start, runs out of time or is cancelled, nothing is sent and your prompt stays in the input field.

Whenever sending will run commands, a card above the input field says how many, such as `2 commands in the input run when sent`.

Once you choose a shortcut, the input field holds that shortcut's prompt until you clear it or replace it, however much you edit it. Every `` !`…` `` you can see in it when you press **Enter** runs, including any you typed or pasted. A message you type yourself without choosing a shortcut never runs commands.

### Agent and model

If the shortcut sets an `agent` or `model` that differs from the one in use, a card above the input field shows the change, such as `Code → QA`. When you send the prompt, the session switches to it and stays on it for the rest of the session. A new session (`/new` or `/clear`) goes back to what you used before, and your default model does not change.

To send the prompt without switching, click **✕ Don't switch** on the card. Choosing an agent with `#` or a model with `$` yourself also replaces the card's switch. If the agent or model named in the file is not found, the card says so and the prompt is sent with the current one.
