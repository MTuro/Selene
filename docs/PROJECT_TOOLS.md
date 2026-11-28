# SELENE project tools

## Architecture audit at 1358229

The stdio MCP server dispatches nine JSON tool schemas through mcp/tools.py. These cover graph and autonomous agent runs, memory, skills, code execution, web search, knowledge queries, delegation and meta statistics. Native tools include workspace files, shell, code, browser, desktop, search, calendar, email, kanban and system operations. ToolRegistry and IsaacTool generate model function schemas. MCP clients can wrap discovered remote tools. Neither discovery nor a wrapper risk level establishes a project grant.

Workspace tools constrain their own roots. Connectors can consume operator capabilities, and high-risk agent actions use an approval callback. No registry binds external project identity to specific operations. Capability tokens support expiry and use limits but alone do not bind a spoken approval to a request or authenticate the speaker.

LangGraph enters Guard, then routes through perception, direct response or planning and execution. The voice worker owns interruption and streaming. Graph state holds execution logs and connector results. Telemetry records node performance. The MetaLearner node accepts a boolean verified_success before recording strategy scores. Episodic recall can involve embeddings and semantic memory stores facts. Neither is a suitable automatic destination for untrusted source text.

## Implementation plan

1. Add a local SQLite registry and operator CLI, with descriptor-relative reads and project identity checks.
2. Expose bounded project tools through MCP. Keep tests behind a local voice approval channel and isolate subprocesses.
3. Route recognized voice commands through LangGraph, preserve interruption, and persist structured observations and explicit feedback in the MetaLearner database.
4. Exercise policy, dispatch, persistence, voice regressions and sandbox availability. Document verified results and deployment limitations.

## Register and use a project

Run these commands from the local operator terminal. Replace the example directory with its canonical absolute path. Each registration gets a fresh ID. Project names are case-insensitive.

```bash
isaac projects register Tapyr /absolute/path/to/Tapyr \
  --allow project_list_files --allow project_read_file \
  --allow project_git_status --allow project_run_tests --runner unittest
isaac projects list
isaac projects inspect Tapyr
isaac voice --hands-free
```

| Spoken request | Response |
| --- | --- |
| “Selene, check the Git status of Tapyr.” | Speaks the change counts and displays branch, staged, modified, untracked and available upstream counts. |
| “List files in Tapyr.” | Speaks the entry count and displays the names. |
| “Read README.md in Tapyr.” | Displays the bounded file excerpt. File contents stay out of speech and model history. |
| “Run the tests for Tapyr.” | Presents the registered name, root and exact command. After the proposal finishes, say “Confirm tests” on the next turn. |
| “Show project history for Tapyr.” | Retrieves recent observations and any operator feedback from disk. |

The local grammar accepts the phrases above. File names use the relative path as transcribed by STT. Pending proposals expire after 60 seconds. Barge-in during the proposal, session exit, rejection, or an intervening utterance discards the grant. Once the sandbox launches, the running job owns its bounded execution and an interrupted turn discards its reply. The microphone and local account are trusted operator interfaces. Spoken approval does not identify the speaker.

```bash
# Replace the complete allowlist, then remove only the registration.
isaac projects permissions Tapyr --allow project_git_status
isaac projects remove Tapyr
```

## MCP and execution boundaries

| MCP name | Arguments in addition to project name or ID | Result |
| --- | --- | --- |
| project_list_files | Optional relative path and depth (1–5) | Up to 200 entries, a truncation flag and count of inaccessible children. |
| project_read_file | Relative path, optional inclusive start and end (1-based) | UTF-8 text, effective bounds and truncation flag. Default window is 200 lines. Input is limited to 1 MiB and output to 16,384 characters. |
| project_git_status | None | NUL-delimited porcelain v2 parsed into fields. Git uses an isolated read-only view, a fixed argv and conventional local upstream refs. |
| project_run_tests | None | Separate stdout and stderr, exit code, timeout and truncation flags. A stdio MCP client receives an authorization error. Only the local voice host can supply a confirmation. |

The registry lives in projects.sqlite3 in the active SELENE profile. Operations hold a registry transaction while validating and using a grant. The filesystem layer opens directory components without following symlinks, checks the registered device/inode, and rejects hard-linked file reads. It excludes common credential paths, Git internals, virtual environments and dependency caches. Redaction covers recognized credential patterns. Operators should grant reads only on trees whose remaining contents they intend to share with local MCP clients.

The graph selects project commands only in the live voice context. Other requests follow the existing Guard and perception route. The project node ends the graph after one operation, displays escaped JSON, and adds an English summary to message history. External MCP results contain untrusted repository data. The project grant applies to this tool family. Independently configured legacy host tools still use their own permission policies.

The in-memory capability binds the originating turn, session, tool, arguments and complete grant configuration. The voice worker transfers it only to the next uninterrupted turn. Execution rechecks the active grant and consumes the capability immediately before launch. The audit trail records grant changes, requests, decisions and outcomes. Raw excerpts and test output are excluded from strategy scores.

## Linux sandbox and limitations

Git and test execution require Linux user namespaces, Bubblewrap with descriptor bind mounts and nested user-namespace disabling, libseccomp, and the system Git and Python under /usr. Verified on Arch with Bubblewrap 0.13.1. SELENE uses an absolute executable path, shell=False, a clean environment, read-only runtime mounts, a private 16 MiB temporary filesystem and a default-deny syscall filter derived from the existing sandbox policy. The filter denies socket and process creation. The child installs hard limits of 15 CPU seconds, 512 MiB address space and 1 MiB per output file. The launcher enforces a 25-second subprocess timeout and returns at most 64 KiB stdout and 8 KiB stderr. Sandbox startup errors stop the operation.

Tests run from a filtered, read-only temporary copy (64 MiB total, 8 MiB per file, 20,000 directory entries). The supported configurations are system Python's unittest and pytest modules. Dependencies must already be available to system Python. Suites requiring subprocesses, threads, sockets, writable checkout files, Git history or a project virtualenv may fail. An exit status describes the configured runner, including its startup failures. Git support targets standard repositories with a .git directory and conventional remote tracking. Linked worktrees, custom refspecs, submodule detail, custom filters and user-level ignore rules are outside this implementation.

## Learning and offline operation

Every authorized tool observation stores project ID, tool, session, duration, execution outcome and a bounded result summary in the MetaLearner SQLite database. Recognized-project errors are also recorded. PerformanceTracker records execution timing. A null fulfilled field means the objective awaits assessment. Only explicit local feedback writes a project_development task label and updates the tool's strategy score. Verification is transactional and single-use.

```bash
isaac projects history Tapyr
isaac projects verify Tapyr 12 --fulfilled --feedback "Confirmed the expected test suite ran"
# For a different observation, use --unfulfilled if the objective was missed.
```

Registry, MCP, history and the deterministic project route work with Ollama offline. Normal model-backed conversation retains its existing preflight diagnostics. Use `isaac doctor` to inspect provider availability. This feature does not start services, pull models, schedule learning or change model weights.

## Verification

```bash
python -m pytest -q tests/test_project_tools.py
ISAAC_HOME="$(mktemp -d)" ISAAC_IDENTITY_PROFILE=isaac HF_HUB_OFFLINE=1 python -m pytest -q -rs tests
```

On 2026-11-28 the full Python 3.12 suite passed 1,391 tests, including all ten project tests and existing STT, TTS streaming, hands-free interruption, cancellation, language and identity coverage. Three unrelated cases skipped for optional SymPy, optional PyArrow and the Windows editor. Real Git and unittest commands ran inside Bubblewrap. Voice integration used simulated transcription and playback. Ruff lint, format and mypy checks passed.
