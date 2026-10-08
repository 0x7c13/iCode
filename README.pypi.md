# iCode

iCode is a general-purpose, extensible agent platform that runs in your terminal: a full-screen
TUI, a headless command line and an Agent Client Protocol (ACP) server. Screenshots, the full
user guide and the source are on [GitHub](https://github.com/openJiuwen-ai/iCode).

> **Work in progress.** Interfaces, file formats, and defaults still change between releases.

## Install

iCode needs Python 3.14. The simplest way to install it is with
[uv](https://docs.astral.sh/uv/), which downloads Python for you if needed:

```bash
uv tool install iCode-TUI
icode
```

To upgrade later, run `uv tool upgrade iCode-TUI`; to remove it, `uv tool uninstall iCode-TUI`.

What your system needs:

- **Windows (x64), Linux (x86-64 or ARM64, glibc 2.27 or later):** nothing else.
- **macOS on Apple silicon:** the Xcode Command Line Tools (`xcode-select --install`), used to
  build one dependency during installation.
- **Intel Mac, Windows on Arm, older Linux, or no internet access:** use the offline packages
  on the [Releases](https://github.com/openJiuwen-ai/iCode/releases) page instead. They include
  Python and every dependency.

Use a modern terminal, such as [Windows Terminal](https://github.com/microsoft/terminal) on
Windows or [Ghostty](https://ghostty.org) on macOS and Linux.

## First steps

iCode ships no model profiles, so press **F4** on first launch to add one. Press **F8** to open
the built-in user guide, which walks through everything else.

## Privacy

iCode does not collect your data. It has no telemetry, analytics or crash reporting, and by
default it sends nothing, usage data included, to us or to any other third party. The full
statement, including what the optional web tools send, is in the
[README](https://github.com/openJiuwen-ai/iCode#privacy).

## License

iCode is open source under the Apache License 2.0. It also contains and bundles third-party
software, each part under its own license; the NOTICE file lists them.
