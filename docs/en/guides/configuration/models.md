# Configure models

A model profile contains the information needed to connect to a model service. This guide explains how to create, clone, edit, switch, and delete model profiles in the terminal user interface (TUI). Once saved, profiles let you switch between tasks without entering connection parameters again.

## Open the model configuration window

Open the "Model Configuration" window in any of these ways:

- Press **F4**.
- Click `f4 Models` at the bottom of the interface.
- Enter `/models` in the input field.

The list on the left shows existing model profiles. The form on the right shows the currently selected profile.

While a task is running, the "Model Configuration" window opens in read-only mode. You cannot create, edit, or delete profiles. Wait for the task to finish before changing the configuration.

## Manage model profiles

### Create a model profile

1. Click "New".
2. Enter a "Profile Name" of your choice, such as `My Model`.
3. In the "Model Options" section, select the "Provider" and "API Style". Only OpenAI and DeepSeek (OpenAI) provide an API Style option. Before configuring it, check the model service's documentation for supported APIs and follow its recommendation when choosing "Chat Completions" or "Responses". If the service supports only one API, select its corresponding style; if it supports both, choose according to the service's recommendation or your needs.
4. Enter the model ID in "Model" and, as required by the model service, fill in "Base URL" and "API Key".
5. Set "Max Context Window" and the "Max Output Tokens" field according to the provider's documentation.
6. Adjust "Streaming", "Vision Model", and "Connection Options" as needed.
7. Click "Save".

**Notes:**

- Clicking "New" immediately creates and saves a blank profile. You must click "Save" for any subsequent entries or changes to take effect. Closing the window does not delete the blank profile you created.
- The profile name identifies the model profile in the interface and is not sent to the model service. It cannot be empty or duplicate an existing name (case-insensitive). Names can contain Chinese characters, spaces, letters, numbers, and common symbols.
- A profile with missing required information, such as the model ID or token limits, does not appear in the model selection list.
- If no usable model profile existed before, the new profile automatically becomes the current model profile after you save it and close the window. If another model profile is already in use, saving a new profile does not automatically switch models.
- To set reasoning effort, prompt caching, or other request fields, use "Chat Options" in the "Extra Options" section. See [Chat Options](./chat-options.md), which also lists settings for common model services. When a save makes a profile a Claude profile on the Anthropic protocol without a cache setting, iCode offers to add one.

### Provide an API key through an environment variable

An API key entered directly is saved locally in the model profile file.

To read the key from an environment variable, enter the following in the "API Key" field:

```text
{{PROVIDER_API_KEY}}
```

Replace `PROVIDER_API_KEY` with the name of the environment variable that holds the key. Make sure the variable is set before starting iCode. Environment variable names can contain only letters, numbers, and underscores, and cannot start with a number.

### Clone a model profile

Clone an existing profile when you need another profile with mostly the same parameters. For example, you might use a different model ID from the same provider, or set different context and output limits for the same model.

1. Select the model profile to copy in the list on the left.
2. Click "Clone".
3. iCode creates and selects a copy containing all fields from the original profile, and generates a unique name for it.
4. Edit the profile name and any fields you want to change.
5. Click "Save".

Clicking "Clone" saves the copy immediately, but does not switch the current model to the copy.

### Edit a model profile

1. Select a model profile in the list on the left.
2. Edit the fields you want to change.
3. Click "Save".

If you edit the model profile currently in use, the updated configuration applies to subsequent requests after you save it and close the configuration window.

### Delete a model profile

Deleting a profile removes its local model profile file and **cannot be undone in iCode**. Once you are sure you no longer need the profile:

1. Open the "Model Configuration" window.
2. Select the profile to delete in the list on the left.
3. Click "Delete".
4. In the confirmation window, check the profile name again, then click "Delete".

iCode keeps at least one model profile, so "Delete" is unavailable when only one remains. If you delete the profile currently in use, iCode automatically switches to another selectable profile after you close the configuration window.

## Switch and verify model profiles

### Switch the current model profile

You can switch model profiles by clicking the model profile name or using the input field.

To switch by clicking the model profile name:

1. Click the model profile name in the status bar above the input field to open the model selection window.
2. Click a model profile, or select one with the Up and Down arrow keys and press Enter to confirm.

To switch quickly from the input field:

1. Enter `$` in the input field. The model profile list appears above it.
2. Continue typing the profile name to filter the results.
3. Use the Up and Down arrow keys to select a model profile, then press Enter to confirm.

After switching, the status bar shows the new model profile name. The profile currently in use appears in gray in the list, with a hollow circle before its name. Profiles with missing required fields do not appear in the list.

If the model profile name in the status bar is not clickable, the current agent is bound to a specific model profile. In this case, entering `$` does not display the model profile list either. You must first [change the agent's model settings](./agents.md#set-the-model-an-agent-uses) before you can switch model profiles. You also cannot switch while the agent is running.

### Verify a model profile

After switching models, submit the following in the input field:

```text
Hello
```

iCode should respond with a greeting. If the model service returns an error, first check the model ID, base URL, API key, and other model profile settings.

If the error persists, compare the symptoms against the following cases:

- **The error message contains `The Chat Completions client supports only n=1`**: With the Chat Completions style, iCode processes only one response per request and does not support multiple candidate responses, so the `n` parameter must be omitted or set to the integer `1`. Remove the value or set it to `1` in the profile's ["Chat Options"](./chat-options.md), including when it was set indirectly through `extra_body`.
- **The response has an `HTTP 200` status but is not a valid model reply**: for example a `text/html` Content-Type with an error page, an error JSON body, or an empty body. This means a gateway or proxy wrapped a backend error as a successful response; check that the selected API style matches the model service address, and review the proxy route.

### If iCode can't reach the model service

When iCode can tell why a model request failed, it says so in plain words, with the original error text below it; a paused sub-agent's card shows both as well. While iCode retries, the retry notice shows only the plain-words message. If the device running iCode doesn't seem to have a network connection, iCode says so as well; check that first.

| The message says | What to check |
| --- | --- |
| Can't resolve the address | Your network connection and DNS. If the address is wrong, fix the base URL in the model profile. |
| The host refused the connection | The address and port in the base URL, and that the model service is running. |
| Connecting or waiting for a response timed out | Your network, firewall, or VPN. |
| Can't connect to the proxy | That the proxy is running and set up correctly. To connect without the proxy, turn on **Bypass proxy** in the model profile. |
| The proxy requires authentication | The proxy's user name and password. |
| Couldn't establish a secure connection | On a corporate network, the proxy's certificate setup. |
| The API key is invalid or lacks access | The API key in the model profile. |
| The request exceeds the model's context window | That the context window in the model profile matches the model's real one. |
