"""
Type-safe data models for Claude Code JSONL conversation format.

This module provides Pydantic models that mirror Claude's JSONL conversation structure,
enabling safe parsing and manipulation of conversation data.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage statistics for API calls."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    service_tier: str | None = None


class TextContent(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingContent(BaseModel):
    """Claude's internal reasoning content block."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ImageContent(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    source: dict[str, Any]


class ToolUseContent(BaseModel):
    """Tool invocation request content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    """Tool execution result content block."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | dict[str, Any] | list[dict[str, Any]]


ContentBlock = TextContent | ThinkingContent | ImageContent | ToolUseContent | ToolResultContent


class BaseMessage(BaseModel):
    """Base message structure common to all message types."""

    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]


class UserMessage(BaseMessage):
    """User message structure."""

    role: Literal["user"] = "user"
    content: str | list[ContentBlock]


class AssistantMessage(BaseMessage):
    """Assistant message with API metadata."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[ContentBlock]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: TokenUsage | None = None  # Made optional for compatibility


class StructuredToolUseResult(BaseModel):
    """Structured metadata for tool execution results with todos."""

    oldTodos: list[dict[str, Any]] | None = None
    newTodos: list[dict[str, Any]] | None = None
    # Extensible for other tool result types


ToolUseResult = StructuredToolUseResult | str | list[dict[str, Any]]


class TaskInvocation(BaseModel):
    """Task tool invocation metadata."""

    tool_use_id: str
    description: str
    prompt: str
    timestamp: datetime | None = None
    entry_uuid: str | None = None
    # Task result metadata (populated when result is found)
    total_duration_ms: int | None = None
    total_tokens: int | None = None
    total_tool_use_count: int | None = None
    # Detailed usage breakdown from result
    result_input_tokens: int | None = None
    result_output_tokens: int | None = None
    result_cache_creation_tokens: int | None = None
    result_cache_read_tokens: int | None = None


class SummaryEntry(BaseModel):
    """Summary entry with minimal structure."""

    type: Literal["summary"] = "summary"
    summary: str
    leafUuid: str = Field(alias="leafUuid")

    class Config:
        populate_by_name = True


class SystemEntry(BaseModel):
    """System message entry."""

    type: Literal["system"] = "system"
    content: str
    # Minimal required fields for system messages
    uuid: str | None = None
    timestamp: datetime | None = None
    sessionId: str | None = Field(default=None, alias="sessionId")
    version: str | None = None
    cwd: str | None = None
    userType: str | None = Field(default=None, alias="userType")
    isSidechain: bool | None = Field(default=None, alias="isSidechain")
    parentUuid: str | None = Field(default=None, alias="parentUuid")

    class Config:
        populate_by_name = True
        extra = "allow"  # Allow unknown fields for system messages


class ConversationEntry(BaseModel):
    """
    A single entry in a Claude JSONL conversation file.

    Represents one line of the JSONL file with all metadata and message content.
    """

    # Common metadata for all entries
    uuid: str
    timestamp: datetime
    sessionId: str = Field(alias="sessionId")
    version: str
    cwd: str
    userType: str = Field(alias="userType")
    isSidechain: bool = Field(alias="isSidechain")
    parentUuid: str | None = Field(alias="parentUuid")

    # Message content and type
    type: Literal["user", "assistant"]
    message: UserMessage | AssistantMessage | None = None

    # Optional fields for specific entry types
    isMeta: bool | None = Field(default=None, alias="isMeta")
    requestId: str | None = Field(default=None, alias="requestId")
    costUSD: float | None = Field(default=None, alias="costUSD")
    durationMs: int | None = Field(default=None, alias="durationMs")
    toolUseResult: ToolUseResult | None = Field(default=None, alias="toolUseResult")
    isApiErrorMessage: bool | None = Field(default=None, alias="isApiErrorMessage")

    class Config:
        populate_by_name = True
        extra = "allow"  # Allow unknown fields for extensibility


class ToolResultEntry(BaseModel):
    """
    A tool execution result entry in a Claude JSONL conversation file.

    This represents a "user" type entry that contains tool execution results,
    not actual user messages. Distinguished by the presence of toolUseResult field.
    """

    # Common metadata for all entries
    uuid: str
    timestamp: datetime
    sessionId: str = Field(alias="sessionId")
    version: str
    cwd: str
    userType: str = Field(alias="userType")
    isSidechain: bool = Field(alias="isSidechain")
    parentUuid: str | None = Field(alias="parentUuid")

    # Fixed type for tool results
    type: Literal["user"] = "user"

    # Tool result specific fields
    toolUseResult: ToolUseResult = Field(alias="toolUseResult")

    # Optional fields
    isMeta: bool | None = Field(default=None, alias="isMeta")
    requestId: str | None = Field(default=None, alias="requestId")
    costUSD: float | None = Field(default=None, alias="costUSD")
    durationMs: int | None = Field(default=None, alias="durationMs")
    isApiErrorMessage: bool | None = Field(default=None, alias="isApiErrorMessage")
    message: UserMessage | None = None  # Tool results may or may not have a message

    class Config:
        populate_by_name = True
        extra = "allow"  # Allow unknown fields for extensibility


AnyEntry = ConversationEntry | ToolResultEntry | SummaryEntry | SystemEntry


class Conversation(BaseModel):
    """
    A complete Claude Code conversation.

    Represents a parsed and structured conversation with metadata and entries.
    """

    session_id: str
    project_path: str
    entries: list[AnyEntry]
    created_at: datetime
    last_updated: datetime
    task_invocations: list[TaskInvocation] = Field(default_factory=list)

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def user_entries(self) -> list[ConversationEntry]:
        return [
            entry
            for entry in self.entries
            if isinstance(entry, ConversationEntry)
            and entry.type == "user"
            and not entry.toolUseResult
        ]

    @property
    def assistant_entries(self) -> list[ConversationEntry]:
        return [
            entry
            for entry in self.entries
            if isinstance(entry, ConversationEntry) and entry.type == "assistant"
        ]

    @property
    def tool_result_entries(self) -> list[ToolResultEntry]:
        return [entry for entry in self.entries if isinstance(entry, ToolResultEntry)]

    @property
    def summary_entries(self) -> list[SummaryEntry]:
        return [entry for entry in self.entries if isinstance(entry, SummaryEntry)]

    @property
    def system_entries(self) -> list[SystemEntry]:
        return [entry for entry in self.entries if isinstance(entry, SystemEntry)]

    @property
    def tool_usage_count(self) -> int:
        count = 0
        for entry in self.assistant_entries:
            if entry.message and isinstance(entry.message.content, list):
                for content_block in entry.message.content:
                    if isinstance(content_block, ToolUseContent):
                        count += 1
        return count


class ProjectConversations(BaseModel):
    """
    All conversations for a specific project.

    Groups conversations by project directory with metadata.
    """

    project_path: str
    project_name: str
    conversations: list[Conversation]

    @property
    def total_conversations(self) -> int:
        return len(self.conversations)

    @property
    def latest_conversation(self) -> Conversation | None:
        if not self.conversations:
            return None
        return max(self.conversations, key=lambda c: c.last_updated)

    @property
    def total_entries(self) -> int:
        return sum(conv.total_entries for conv in self.conversations)
