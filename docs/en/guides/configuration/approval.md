# Configure approval modes

When an agent makes a tool call, such as running a Shell command or writing a file, iCode may require approval first. This guide explains how to choose an approval mode and handle approval requests in the terminal user interface (TUI), as well as how approval settings differ across ways of running iCode.

## Understand the three approval modes

The approval mode determines how iCode handles tool calls that require approval. Which calls require approval depends on both the `approval` policy in the agent profile and iCode's safety rules. See the [approval field in the agent profile reference](../../reference/agent-profile.md#approval).

The three approval modes behave as follows:

| Mode | Behavior |
| --- | --- |
| MANUAL | Tool calls that require approval open a dialog and wait for a person to approve or decline. |
| AUTO | An approval judge model evaluates tool calls. Calls judged safe are approved automatically; suspicious calls are flagged for a person to decide. |
| BYPASS | Tool calls run without asking, even when the agent configuration or safety rules require approval. |

**Approval judge model**: In automatic mode, iCode calls the approval judge model and sends it the current time, the workspace directories, all user prompts of the current turn and the latest of them, and the tool name, tool kind, and arguments. By default, the approval judge uses the current session's model. To change it, press **F10** to open **Settings**, select the **Models & Agents** tab, and change **Approval judge model** in the **Model roles** section.

> **Tip**
>
> **Manual mode does not mean every tool call opens a dialog.** It means that calls requiring approval are decided by the user. Calls that do not require approval or qualify for automatic approval run directly. See [Understand automatic approval and safety protections](#understand-automatic-approval-and-safety-protections).

## Switch approval modes in the TUI

Use either of these methods to switch the current approval mode in the TUI:

- Type `/approval` in the input field, then press **Space** or **Enter** to open the approval mode list and choose a mode. You can also specify a mode directly, for example `/approval auto`.
- Click the approval mode label in the upper-right corner of the interface and choose a mode in the **Approval Mode** dialog.

You can switch modes while a task is running. Approval requests that are already open are unaffected; subsequent tool calls use the new mode.

### Set the default approval mode

With the default settings unchanged, the TUI starts in manual approval mode on its first launch. Switching the current approval mode also updates the default for the next launch: choosing manual or automatic mode saves that mode as the default. Choosing bypass mode applies only to the current run; to avoid continuing to bypass approval protections after a restart, the default for the next launch is saved as automatic mode.

To change only the default for the next launch without changing the current approval mode, press **F10** to open **Settings** and change **Default approval mode** on the **Security** tab. **Settings** does not offer bypass mode as a default that can be saved.

## Handle approval requests in the TUI

Tool calls that require approval open an **Approval Required** dialog showing the tool name and call arguments. For file edits, it also shows the planned diff so you can review changes before they are made.

- Press **Y** to approve or **N** to decline, or click the corresponding button. You cannot close the dialog with **Esc**; you must explicitly approve or decline.
- When declining, you can provide a reason. The reason is sent to the agent to help it adjust its next steps. Once a reason is entered, the approve button is disabled.
- In automatic mode, the dialog initially shows **Evaluating**. If the model judges the call safe, the dialog closes automatically. If the model judges it suspicious, the title changes to **Flagged by Auto-Review** and the dialog shows the reason and waits for a person to decide. You can also approve or decline directly while evaluation is in progress. If the approval judge model is unavailable or evaluation fails, iCode keeps the dialog open for a person to decide instead of approving the tool call automatically.

## User Approval Reuse (Don't Ask Again, or DAA)

Enable this experimental feature with `CHRYS_APPROVAL_REUSE=1` or `approval.reuse_enabled: true` in [user settings](../../reference/settings.md). The default is off. In eligible dialogs, choose **Remember — this session** or **Remember — this project**; the default remains **Allow once**. Only an explicit human choice creates a grant. Editing a request approves it once; if a hook changes it again, the resulting request requires confirmation.

### Interfaces and data

| Interface / type | Contract |
| --- | --- |
| `ApprovalReuseService.match / remember` | Match returns covering grant IDs; remember returns whether storage succeeded. |
| `ApprovalRequest.reuse_offer` / `ApprovalResponse.remember_choice` | A typed display offer and one-time, session, project or extra-argument choice. Display text is never a matching key. |
| `CommandKey` / `FileKey` | Command tokens (or exact text), normalized working directory, Shell name, executable path, startup arguments and remaining execution options; or an actual file destination. |
| `ApprovalGrant` | ID, session/project scope and owner, normalized project path, explicit-user source, creation time, prefix flag and structured key. |

### Authorization rules

| Operation | What is remembered |
| --- | --- |
| Shell command | Command arguments, actual working directory and Shell configuration must match in both scopes. Simple commands compare normalized tokens; complex commands support project-only exact text. Environment variables, reason, timeout and output limit are not bound. An explicit `working_dir` uses ordinary approval. |
| File write/edit | Permission to modify the physical file, after resolving parent directory links; contents may change. Every affected file must be covered within one scope. Changing the destination requires approval again; the worker also checks the approved target. A final-component symlink uses ordinary approval and retains target-change checks. |
| Reads and other tools | No grant reuse. File reads also check the resolved target for sensitivity; custom tools keep ordinary approval, including valid date/UUID arguments. |

Scope: main-agent local Shell and file write/edit only; sensitive requests, sub-agents, workflow nodes, remote/MCP tools and other custom tools do not create or reuse grants. Existing automatic approval and bypass policies still apply.

**Known issue — generic prefix grants:** the extra-argument choice remains available for supported literal commands. Remembering `git push origin main` this way also permits `git push origin main --force`. Appended arguments may make an operation destructive; this risk is deferred, and the interface warns about it. Choose the ordinary remember option when arguments must remain identical.

### Persistence and management

Project grants live in the user-owned `<config_dir>/approval-grants.json`; session grants live in `<session_root>/sessions/<short-id>/approval-grants.json`. Both use versioned JSON, the existing file lock, a locked re-read and owner-only atomic writes. Unsafe files and symlink paths are rejected. Each file is limited to 1,000 grants / 4 MiB; a full or unwritable store allows the current approved call but reports that it was **not remembered**. Repository files cannot declare user approval.

Session grants survive rebuilding or restoring the same session, disappear when that session is deleted, and are not copied into a fork. Project grants remain until revoked or cleared. Disabling reuse leaves records intact; re-enabling restores their effect. Old development SQLite grants and configuration names are not imported: enable the new setting and approve again. These checks do not provide filesystem sandbox isolation against concurrent changes by other processes.

```bash
icode approvals list                         # all project and saved-session grants
icode approvals list --project /work/demo    # filter by normalized project path
icode approvals list --session SESSION_ID --json
icode approvals revoke GRANT_ID
icode approvals clear --project /work/demo
icode approvals clear --session SESSION_ID
icode approvals clear --all
```

Management works even when reuse is off. Lists show IDs, scope, project and target; a reused call records `grant_ids` in its approval decision and `approval_grant_ids` in tool metadata so it can be traced to the relevant rule.

## Understand automatic approval and safety protections

The following operations usually run without an approval dialog:

- Safe, read-only Shell commands that do not access sensitive targets, such as `ls`, `cat`, and `grep`.
- File writes within the working directory's Git repository that do not access sensitive targets.

Shell commands and file reads or writes that access sensitive targets such as `.env` files, credentials, and private keys still go through approval, even for read-only operations or writes within a Git working directory. This protection applies in manual and automatic modes. Bypass mode skips these approval protections.

Skill scripts run locally with the current user's permissions, without sandbox isolation, so they request approval by default. When the session uses bypass mode, skill scripts run without asking. Before installing or running a third-party skill, review its `SKILL.md`, scripts, and related files to make sure they are trustworthy.

Web tools send requests to outside services, so the `web_search` and `web_fetch` kinds also request approval by default, even when the agent's `approval.default` is `auto`. Explicit `approval.overrides` rules for these kinds or tool names still take precedence. In automatic mode the approval judge model may approve them, and in bypass mode they run without asking. See [Configure web tools](./web-tools.md#approve-web-tool-calls).

## Verify approval modes in the TUI

Select the built-in Code agent, then submit this request:

```text
Use the Shell tool to run icode --version
```

This command only displays the version and does not modify files, but it is not among the read-only Shell commands approved automatically. Expect the following results:

- In manual mode, the **Approval Required** dialog opens. After approval, the iCode version is displayed.
- In automatic mode, the approval judge model will usually judge the command safe and approve it automatically. If the model flags it as suspicious, the dialog shows the evaluation reason and waits for a decision.
- In bypass mode, the command runs and displays the iCode version without an approval dialog.

## Approval modes in other ways of running iCode

Other ways of running iCode use the following approval modes and switching methods:

- **Headless CLI (`icode run`)**: Always bypasses approval and provides no approval-related options.
- **iCode ACP server**: Defaults to manual mode. Use `icode acp --approval manual|auto|bypass` to set the initial mode. ACP clients that support this capability can also switch the current session's mode.
- **Browser-hosted TUI (`icode serve`)**: Use the TUI operations described earlier to switch approval modes and handle approval requests.
