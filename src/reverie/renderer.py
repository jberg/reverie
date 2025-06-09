"""
Terminal renderer for Claude Code conversations.

Transforms parsed Conversation objects into beautiful terminal displays
with syntax highlighting, formatted messages, and tool usage summaries.
"""

from datetime import datetime
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .models import (
    AnyEntry,
    AssistantMessage,
    ContentBlock,
    Conversation,
    ConversationEntry,
    ImageContent,
    SummaryEntry,
    SystemEntry,
    TaskInvocation,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)


class ConversationRenderer:
    """Renders conversations in the terminal with Rich formatting."""

    def __init__(self, console: Console | None = None):
        """Initialize renderer with optional console."""
        self.console = console or Console()

    def render(self, conversation: Conversation) -> None:
        """
        Render a complete conversation to the terminal.

        Args:
            conversation: Parsed conversation object to render
        """
        self._render_header(conversation)

        # Create a map of task invocations by their entry UUID for inline display
        tasks_by_entry: dict[str, list[TaskInvocation]] = {}
        for task in conversation.task_invocations:
            if task.entry_uuid:
                if task.entry_uuid not in tasks_by_entry:
                    tasks_by_entry[task.entry_uuid] = []
                tasks_by_entry[task.entry_uuid].append(task)

        for entry in conversation.entries:
            self._render_entry(entry)

            # Check if this entry has any Task invocations
            if isinstance(entry, ConversationEntry) and entry.uuid in tasks_by_entry:
                for task in tasks_by_entry[entry.uuid]:
                    self._render_task_invocation(task)

            self.console.print()

    def _render_header(self, conversation: Conversation) -> None:
        """Render conversation header with metadata."""
        header = Text()
        header.append("Conversation Details\n", style="bold blue")
        header.append(f"Project: {conversation.project_path}\n")
        header.append(f"Created: {self._format_timestamp(conversation.created_at)}\n")
        header.append(f"Last Updated: {self._format_timestamp(conversation.last_updated)}\n")
        header.append(f"Total Entries: {conversation.total_entries}")

        if conversation.tool_usage_count > 0:
            header.append(f" | Tool Uses: {conversation.tool_usage_count}")

        self.console.print(
            Panel(header, title="[bold]Claude Code Conversation[/bold]", border_style="blue")
        )
        self.console.print()

    def _render_entry(self, entry: AnyEntry) -> None:
        """Render a single conversation entry."""
        if isinstance(entry, ConversationEntry):
            self._render_conversation_entry(entry)
        elif isinstance(entry, SummaryEntry):
            self._render_summary_entry(entry)
        elif isinstance(entry, SystemEntry):
            self._render_system_entry(entry)

    def _render_conversation_entry(self, entry: ConversationEntry) -> None:
        """Render a conversation entry (user or assistant message)."""
        if not entry.message:
            return

        if entry.type == "user":
            self._render_user_message(entry)
        else:
            self._render_assistant_message(entry)

    def _render_user_message(self, entry: ConversationEntry) -> None:
        """Render a user message."""
        timestamp = self._format_timestamp(entry.timestamp)

        content = self._extract_content(entry.message.content) if entry.message else Text("")

        # Create panel with user styling
        panel = Panel(
            content,
            title=f"[bold green]User[/bold green] • {timestamp}",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _render_assistant_message(self, entry: ConversationEntry) -> None:
        """Render an assistant message."""
        if not entry.message or not isinstance(entry.message, AssistantMessage):
            return

        timestamp = self._format_timestamp(entry.timestamp)
        model = entry.message.model

        content_parts = []
        for block in entry.message.content:
            content_parts.append(self._render_content_block(block))

        content = (
            Group(*content_parts)
            if len(content_parts) > 1
            else (content_parts[0] if content_parts else "")
        )

        # Create panel with assistant styling
        title = f"[bold blue]Assistant[/bold blue] • {model} • {timestamp}"
        panel = Panel(
            content,
            title=title,
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _render_content_block(self, block: ContentBlock) -> Any:
        """Render a single content block."""
        if isinstance(block, TextContent):
            return self._render_text_content(block)
        elif isinstance(block, ThinkingContent):
            return self._render_thinking_content(block)
        elif isinstance(block, ToolUseContent):
            return self._render_tool_use_content(block)
        elif isinstance(block, ToolResultContent):
            return self._render_tool_result_content(block)
        elif isinstance(block, ImageContent):
            return Text("[Image]", style="dim italic")
        else:
            return Text(str(block), style="dim")

    def _render_text_content(self, block: TextContent) -> RenderableType:
        """Render text content with code block detection."""
        text = block.text

        # Simple code block detection
        if "```" in text:
            parts: list[RenderableType] = []
            segments = text.split("```")

            for i, segment in enumerate(segments):
                if i % 2 == 0:
                    if segment.strip():
                        parts.append(Markdown(segment))
                else:
                    lines = segment.split("\n", 1)
                    if len(lines) >= 2:
                        language = lines[0].strip()
                        code = lines[1]
                        parts.append(
                            Syntax(code, language or "text", theme="monokai", line_numbers=True)
                        )
                    else:
                        parts.append(Syntax(segment, "text", theme="monokai", line_numbers=True))

            return Group(*parts) if len(parts) > 1 else (parts[0] if parts else "")
        else:
            return Markdown(text)

    def _render_thinking_content(self, block: ThinkingContent) -> Any:
        """Render thinking content."""
        return Panel(
            Text(block.thinking, style="italic dim"),
            title="[dim]Thinking[/dim]",
            border_style="dim",
            padding=(0, 1),
        )

    def _render_tool_use_content(self, block: ToolUseContent) -> Any:
        """Render tool use content."""
        content = Text()
        content.append(f"Tool: {block.name}\n", style="bold yellow")
        content.append(f"ID: {block.id}\n", style="dim")

        if block.input:
            content.append("Parameters:\n", style="bold")
            for key, value in block.input.items():
                content.append(f"  {key}: ", style="cyan")
                content.append(f"{value}\n")

        return Panel(content, title="[yellow]Tool Use[/yellow]", border_style="yellow")

    def _render_tool_result_content(self, block: ToolResultContent) -> Any:
        """Render tool result content."""
        content = Text()
        content.append(f"Tool ID: {block.tool_use_id}\n", style="dim")

        if isinstance(block.content, str):
            content.append(block.content)
        else:
            content.append(str(block.content))

        return Panel(content, title="[green]Tool Result[/green]", border_style="green")

    def _render_summary_entry(self, entry: SummaryEntry) -> None:
        """Render a summary entry."""
        panel = Panel(
            Text(entry.summary, style="italic"),
            title="[bold magenta]Summary[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _render_system_entry(self, entry: SystemEntry) -> None:
        """Render a system entry."""
        content = Text(entry.content, style="dim")

        if entry.timestamp:
            timestamp = self._format_timestamp(entry.timestamp)
            title = f"[dim]System • {timestamp}[/dim]"
        else:
            title = "[dim]System[/dim]"

        panel = Panel(
            content,
            title=title,
            border_style="dim",
            padding=(0, 1),
        )
        self.console.print(panel)

    def _extract_content(self, content: str | list[ContentBlock]) -> Any:
        """Extract renderable content from message content."""
        if isinstance(content, str):
            return Markdown(content)
        elif isinstance(content, list):
            parts = []
            for block in content:
                parts.append(self._render_content_block(block))
            return Group(*parts) if len(parts) > 1 else (parts[0] if parts else "")
        else:
            return Text(str(content))

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display."""
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def _render_task_invocation(self, task: TaskInvocation) -> None:
        """Render a Task tool invocation with its results."""
        # Create a separator line
        self.console.print("═" * 60, style="dim blue")

        # Build the task content
        content = Text()
        content.append("🌱 Task Launched: ", style="bold yellow")
        content.append(f'"{task.description}"\n', style="yellow")
        content.append(f"   ID: {task.tool_use_id[:16]}...\n", style="dim")

        # Show runtime if available
        if task.total_duration_ms is not None:
            duration_sec = task.total_duration_ms / 1000
            content.append(f"   Runtime: {duration_sec:.1f}s\n", style="cyan")

        # Show token usage if available
        if task.total_tokens is not None:
            content.append(f"   Tokens: {task.total_tokens:,}", style="green")
            if task.result_input_tokens and task.result_output_tokens:
                content.append(
                    f" (in: {task.result_input_tokens:,}, out: {task.result_output_tokens:,})",
                    style="dim green",
                )
            content.append("\n")

        # Show tool count if available
        if task.total_tool_use_count is not None:
            content.append(f"   Tool Calls: {task.total_tool_use_count}\n", style="blue")

        # Show prompt preview (first 100 chars)
        if task.prompt:
            prompt_preview = task.prompt[:100] + "..." if len(task.prompt) > 100 else task.prompt
            content.append("\n   Prompt: ", style="dim")
            content.append(f'"{prompt_preview}"\n', style="dim italic")

        self.console.print(content)
        self.console.print("═" * 60, style="dim blue")
