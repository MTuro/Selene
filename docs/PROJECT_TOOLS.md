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
