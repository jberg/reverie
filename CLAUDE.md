# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Code quality
task lint            # Run linting with ruff
task format          # Format code with ruff
task typecheck       # Run type checking with mypy
```

## Architecture

This is a CLI tool for discovering and exporting Claude Code conversations from `~/.claude/projects/`. The architecture follows a clean pipeline:

1. **Discovery** (`discovery.py`) → Finds all conversation files
2. **Parser** (`parser.py`) → Transforms JSONL to structured `Conversation` objects
3. **Picker** (`picker.py`) → Interactive TUI for browsing conversations
4. **Renderer** (`renderer.py`) → Display conversations in terminal
5. **Exporter** (`exporter.py`) → Export to various formats (currently Markdown)

Key architectural decisions:
- **Streaming-first**: Handle large conversations without loading everything into memory
- **Type-safe**: Strict mypy with Pydantic models for all data structures
- **Read-only**: Never modify original Claude files
- **Error-resilient**: Gracefully handle corrupted JSON lines

## Important Patterns

- Models in `models.py` define the complete JSONL schema with Pydantic
- Use `|` union syntax (Python 3.11+) instead of `Union[]`
- Rich library for all terminal UI components
- Claude encodes `/` as `-` in directory names - use `decode_path()` to reverse this
- Only basic emoji, no ZWJ based. These end up breaking console rendering.
- Use task to run project commands and poetry to run python commands
- Consistent orange/brown color palette (#F6C244 primary color)
