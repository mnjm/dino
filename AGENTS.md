## style
Keep changes minimal. No large refactors.
Include proper type annotations, no Any, including return types
PEP8 docs - concise, include args and return info. (skip return info if not returning anything ie None)
Each script needs a brief intro.

## uv
Use uv for all tasks:
uv sync
uv run python ...
uv add/remove <pkg>

## checks (after any meaningful change)
uv run ruff check .
uv run ruff format .
uv run python -m compileall -q -x '^./\.venv/'
uv run basedpyright
All must pass.
