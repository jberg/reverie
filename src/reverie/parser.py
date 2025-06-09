"""
Claude Code conversation parser.

Transforms JSONL conversation streams into structured Conversation objects,
handling tool use aggregation and error recovery for the best user experience.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .discovery import ConversationMetadata, stream_conversation_entries
from .models import (
    AnyEntry,
    BaseMessage,
    Conversation,
    ConversationEntry,
    SystemEntry,
    TextContent,
    ToolResultContent,
    ToolResultEntry,
    ToolUseContent,
)


class ParseError(Exception):
    """Exception raised when conversation parsing fails."""

    def __init__(self, message: str, session_id: str, line_number: int | None = None):
        self.session_id = session_id
        self.line_number = line_number
        super().__init__(message)


def parse_conversation(conv_meta: ConversationMetadata) -> Conversation:
    """
    Parse a complete conversation from JSONL entries.

    Args:
        conv_meta: Conversation metadata with file path and project info

    Returns:
        Parsed Conversation object with all entries and metadata

    Raises:
        ParseError: If conversation cannot be parsed
    """
    try:
        entries: list[AnyEntry] = []
        first_timestamp: datetime | None = None
        last_timestamp: datetime | None = None

        for entry in stream_conversation_entries(conv_meta):
            entries.append(entry)

            entry_timestamp = _extract_timestamp(entry)
            if entry_timestamp:
                if first_timestamp is None:
                    first_timestamp = entry_timestamp
                last_timestamp = entry_timestamp

        # Validate we have at least one entry
        if not entries:
            raise ParseError("No valid entries found in conversation", conv_meta.session_id)

        # Get session ID from first conversation entry
        session_id = conv_meta.session_id
        for entry in entries:
            if isinstance(entry, ConversationEntry):
                session_id = entry.sessionId
                break

        created_at = first_timestamp or conv_meta.first_timestamp or datetime.min
        updated_at = last_timestamp or conv_meta.last_timestamp or datetime.min

        return Conversation(
            session_id=session_id,
            project_path=conv_meta.project_path,
            entries=entries,
            created_at=created_at,
            last_updated=updated_at,
            task_invocations=conv_meta.task_invocations,
        )

    except Exception as e:
        if isinstance(e, ParseError):
            raise
        raise ParseError(f"Failed to parse conversation: {e}", conv_meta.session_id) from e


def _extract_timestamp(entry: AnyEntry) -> datetime | None:
    """Extract timestamp from any entry type."""
    if isinstance(entry, ConversationEntry | SystemEntry | ToolResultEntry):
        return entry.timestamp
    # SummaryEntry doesn't have timestamp
    return None


def _extract_text_from_message(message: BaseMessage) -> str:
    """Extract all text content from a message."""
    if isinstance(message.content, str):
        return message.content

    text_parts = []
    for block in message.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)

    return " ".join(text_parts)


def aggregate_tool_usage(conversation: Conversation) -> dict[str, Any]:
    """
    Aggregate tool usage statistics across the conversation.

    Args:
        conversation: Parsed conversation object

    Returns:
        Dictionary with tool usage statistics and patterns
    """
    stats: dict[str, Any] = {
        "total_tool_calls": 0,
        "unique_tools": set(),
        "tool_frequency": defaultdict(int),
        "tool_sequences": [],
        "error_count": 0,
    }

    current_sequence: list[str] = []

    for entry in conversation.assistant_entries:
        if not entry.message or isinstance(entry.message.content, str):
            continue

        sequence_tools: list[str] = []

        for content_block in entry.message.content:
            if isinstance(content_block, ToolUseContent):
                tool_name = content_block.name
                stats["total_tool_calls"] += 1
                stats["unique_tools"].add(tool_name)
                stats["tool_frequency"][tool_name] += 1
                sequence_tools.append(tool_name)
            elif isinstance(content_block, ToolResultContent):
                if (
                    isinstance(content_block.content, str)
                    and "error" in content_block.content.lower()
                ):
                    stats["error_count"] += 1

        if sequence_tools:
            current_sequence.extend(sequence_tools)
        elif current_sequence:
            stats["tool_sequences"].append(current_sequence.copy())
            current_sequence.clear()

    if current_sequence:
        stats["tool_sequences"].append(current_sequence)

    # Convert set and defaultdict to regular types for JSON serialization
    stats["unique_tools"] = list(stats["unique_tools"])
    stats["tool_frequency"] = dict(stats["tool_frequency"])

    return stats


def get_conversation_summary(conversation: Conversation) -> dict[str, Any]:
    """
    Generate a high-level summary of conversation content and structure.

    Args:
        conversation: Parsed conversation object

    Returns:
        Dictionary with conversation summary statistics
    """
    summary = {
        "session_id": conversation.session_id,
        "project_name": Path(conversation.project_path).name,
        "total_entries": conversation.total_entries,
        "user_messages": len(conversation.user_entries),
        "assistant_messages": len(conversation.assistant_entries),
        "system_messages": len(conversation.system_entries),
        "summary_entries": len(conversation.summary_entries),
        "created_at": conversation.created_at,
        "last_updated": conversation.last_updated,
        "duration_hours": (conversation.last_updated - conversation.created_at).total_seconds()
        / 3600,
        "tool_usage": aggregate_tool_usage(conversation),
    }

    # Add first and last message previews
    if conversation.user_entries:
        first_user = conversation.user_entries[0]
        if first_user.message:
            text = _extract_text_from_message(first_user.message)
            if text:
                summary["first_message_preview"] = text[:100]

    if conversation.assistant_entries:
        last_assistant = conversation.assistant_entries[-1]
        if last_assistant.message:
            text = _extract_text_from_message(last_assistant.message)
            if text:
                summary["last_message_preview"] = text[:100]

    return summary


def parse_multiple_conversations(
    conv_metas: list[ConversationMetadata], max_workers: int = 4
) -> list[tuple[ConversationMetadata, Conversation | ParseError]]:
    """
    Parse multiple conversations efficiently.

    Args:
        conv_metas: List of conversation metadata to parse
        max_workers: Maximum number of concurrent parsing operations

    Returns:
        List of tuples (metadata, result) where result is either Conversation or ParseError
    """
    import concurrent.futures

    def parse_single(
        meta: ConversationMetadata,
    ) -> tuple[ConversationMetadata, Conversation | ParseError]:
        try:
            conversation = parse_conversation(meta)
            return (meta, conversation)
        except Exception as e:
            error = ParseError(str(e), meta.session_id) if not isinstance(e, ParseError) else e
            return (meta, error)

    results: list[tuple[ConversationMetadata, Conversation | ParseError]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {executor.submit(parse_single, meta): meta for meta in conv_metas}

        for future in concurrent.futures.as_completed(future_to_meta):
            results.append(future.result())

    meta_order = {id(meta): i for i, meta in enumerate(conv_metas)}
    results.sort(key=lambda x: meta_order[id(x[0])])

    return results
