@../AGENTS.md

## Code Conventions

Primary language is Python. When writing code, use Python unless explicitly told otherwise. For data work, prefer PySpark/SQL patterns compatible with Databricks runtime.

## Databricks

When working with Databricks notebooks, always create/test in a separate debug notebook before modifying the original. Never overwrite existing notebooks without explicit permission.

For Databricks-related work:
1. Create a separate debug version first and implement the requested changes there.
2. Run and validate the debug version in the target environment. Do not modify the original asset until the debug version has been verified as working.
3. Once validation succeeds, merge only the tested changes back into the original source.

## Tool Usage Priorities

When asked to find files or resources, use available MCP tools and CLI skills (e.g., Databricks CLI, Glean, Trino) BEFORE attempting local bash searches. Ask which environment/cluster to target if unclear.

## Response Style

For infrastructure status checks (clusters, pipelines, jobs), provide concise output first. Only elaborate if asked. When checking Jira/Atlassian, search directly without verbose preamble.

## Data Infrastructure

When encountering catalog/schema/environment mismatches (e.g., test vs prod Trino, EGAP vs EGDP clusters), immediately surface the limitation to the user rather than repeatedly retrying failing queries.
