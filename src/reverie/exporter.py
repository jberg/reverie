"""Export Claude Code conversations to various formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .discovery import ConversationMetadata
from .models import (
    AssistantMessage,
    BaseMessage,
    Conversation,
    ConversationEntry,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
    UserMessage,
)
from .pricing import CostMode, PricingFetcher


class ExportOptions:
    """Configuration options for export formatting."""

    def __init__(
        self,
        include_tool_results: bool = True,
        include_thinking: bool = True,
        include_cost: bool = True,
        include_timestamps: bool = True,
        include_statistics: bool = True,
        max_tool_result_lines: int = 100,
        collapsible_tool_results: bool = True,
        syntax_highlighting: bool = True,
        cost_mode: CostMode = CostMode.AUTO,
    ):
        self.include_tool_results = include_tool_results
        self.include_thinking = include_thinking
        self.include_cost = include_cost
        self.include_timestamps = include_timestamps
        self.include_statistics = include_statistics
        self.max_tool_result_lines = max_tool_result_lines
        self.collapsible_tool_results = collapsible_tool_results
        self.syntax_highlighting = syntax_highlighting
        self.cost_mode = cost_mode


class Exporter(ABC):
    """Abstract base class for conversation exporters."""

    @abstractmethod
    def export(
        self, conversation: Conversation, metadata: ConversationMetadata, output_path: Path
    ) -> None:
        pass


class MarkdownExporter(Exporter):
    """Export conversations to beautiful, comprehensive markdown documents."""

    def __init__(self, options: ExportOptions | None = None):
        self.options = options or ExportOptions()
        self.pricing_fetcher = PricingFetcher()

    def export(
        self, conversation: Conversation, metadata: ConversationMetadata, output_path: Path
    ) -> None:
        """Export a conversation to markdown format."""
        markdown = self._generate_markdown(conversation, metadata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    def _generate_markdown(self, conversation: Conversation, metadata: ConversationMetadata) -> str:
        """Generate the complete markdown document."""
        sections = []

        sections.append(self._generate_frontmatter(conversation, metadata))

        if len(conversation.entries) > 20:
            sections.append(self._generate_table_of_contents(conversation))

        sections.append(self._generate_messages(conversation))

        if self.options.include_statistics:
            sections.append(self._generate_summary(conversation))

        return "\n\n".join(sections)

    def _generate_frontmatter(
        self, conversation: Conversation, metadata: ConversationMetadata
    ) -> str:
        """Generate YAML frontmatter with conversation metadata."""
        meta = metadata
        stats = self._calculate_statistics(conversation)

        total_cost = 0.0
        if self.options.include_cost:
            for entry in conversation.entries:
                if (
                    isinstance(entry, ConversationEntry)
                    and entry.type == "assistant"
                    and entry.costUSD
                ):
                    total_cost += entry.costUSD

        unique_tools = set()
        for entry in conversation.entries:
            if isinstance(entry, ConversationEntry) and entry.message:
                msg = entry.message
                if isinstance(msg, AssistantMessage) and msg.content:
                    for block in msg.content:
                        if isinstance(block, ToolUseContent):
                            unique_tools.add(block.name)

        title = meta.summary if meta.summary else f"Session {meta.session_id[:12]}..."

        frontmatter = [
            "---",
            f'title: "Conversation: {title}"',
            f'session_id: "{meta.session_id}"',
            "project:",
            f'  name: "{meta.project_name}"',
            f'  path: "{meta.project_path}"',
            f'created_at: "{conversation.created_at.isoformat()}"',
            f'updated_at: "{conversation.last_updated.isoformat()}"',
            f"duration_hours: {stats['duration_hours']:.2f}",
            "statistics:",
            f"  total_messages: {stats['total_messages']}",
            f"  user_messages: {stats['user_messages']}",
            f"  assistant_messages: {stats['assistant_messages']}",
            f"  tool_calls: {stats['tool_calls']}",
        ]

        if unique_tools:
            tools_list = sorted(unique_tools)
            frontmatter.append(f"  unique_tools: {tools_list}")

        if self.options.include_cost and total_cost > 0:
            frontmatter.extend(
                [
                    "cost:",
                    f"  total_usd: {total_cost:.4f}",
                    f"  input_tokens: {stats['total_input_tokens']}",
                    f"  output_tokens: {stats['total_output_tokens']}",
                    f"  cache_read_tokens: {stats['total_cache_tokens']}",
                ]
            )

        models = set()
        for entry in conversation.entries:
            if isinstance(entry, ConversationEntry) and entry.message:
                msg = entry.message
                if isinstance(msg, AssistantMessage) and msg.model:
                    models.add(msg.model)

        if models:
            if len(models) == 1:
                frontmatter.append(f'  model: "{list(models)[0]}"')
            else:
                frontmatter.append(f"  models: {sorted(models)}")

        frontmatter.append("---")
        return "\n".join(frontmatter)

    def _generate_table_of_contents(self, conversation: Conversation) -> str:
        """Generate a table of contents for long conversations."""
        toc = ["## 📋 Table of Contents", ""]

        message_num = 0
        for i, entry in enumerate(conversation.entries):
            if isinstance(entry, ConversationEntry) and entry.message:
                msg = entry.message
                if isinstance(msg, UserMessage):
                    message_num += 1
                    preview = self._get_message_preview(msg)
                    anchor = self._generate_anchor(i, msg)
                    toc.append(f"{message_num}. [{preview}](#{anchor})")

        return "\n".join(toc)

    def _generate_messages(self, conversation: Conversation) -> str:
        """Generate formatted messages section."""
        messages = []

        for i, entry in enumerate(conversation.entries):
            if isinstance(entry, ConversationEntry) and entry.message:
                message_block = self._format_message(entry.message, entry, i)
                if message_block:
                    messages.append(message_block)

        return "\n\n---\n\n".join(messages)

    def _format_message(
        self, message: BaseMessage, entry: ConversationEntry, entry_index: int
    ) -> str:
        """Format a single message with all its content."""
        if isinstance(message, UserMessage):
            return self._format_user_message(message, entry, entry_index)
        elif isinstance(message, AssistantMessage):
            return self._format_assistant_message(message, entry, entry_index)
        return ""

    def _format_user_message(
        self, message: UserMessage, entry: ConversationEntry, entry_index: int
    ) -> str:
        """Format a user message."""
        lines = []

        anchor = self._generate_anchor(entry_index, message)
        lines.append(f'<a id="{anchor}"></a>\n')
        lines.append("## 👤 User")

        if self.options.include_timestamps and entry.timestamp:
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"> *{timestamp}*")

        lines.append("")

        if isinstance(message.content, str):
            lines.append(message.content)
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextContent):
                    lines.append(block.text)

        return "\n".join(lines)

    def _format_assistant_message(
        self, message: AssistantMessage, entry: ConversationEntry, _entry_index: int
    ) -> str:
        """Format an assistant message with all content blocks."""
        lines = []

        model_info = f" ({message.model})" if message.model else ""
        lines.append(f"## 🤖 Assistant{model_info}")

        metadata_parts = []
        if self.options.include_timestamps and entry.timestamp:
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            metadata_parts.append(f"*{timestamp}*")

        if self.options.include_cost and entry.costUSD:
            metadata_parts.append(f"💰 ${entry.costUSD:.4f}")

        if entry.durationMs:
            duration_s = entry.durationMs / 1000
            metadata_parts.append(f"⏱️ {duration_s:.1f}s")

        if metadata_parts:
            lines.append("> " + " | ".join(metadata_parts))
            lines.append("")

        if message.content:
            content_sections = []
            thinking_blocks = []

            for block in message.content:
                if isinstance(block, TextContent):
                    content_sections.append(block.text)
                elif isinstance(block, ThinkingContent) and self.options.include_thinking:
                    thinking_blocks.append(block.thinking)
                elif isinstance(block, ToolUseContent):
                    content_sections.append(self._format_tool_use(block))
                elif isinstance(block, ToolResultContent) and self.options.include_tool_results:
                    content_sections.append(self._format_tool_result(block))

            if thinking_blocks:
                lines.append("<details>")
                lines.append("<summary>🧠 Assistant's Thinking Process</summary>")
                lines.append("")
                for thinking in thinking_blocks:
                    lines.append("> " + thinking.replace("\n", "\n> "))
                lines.append("")
                lines.append("</details>")
                lines.append("")

            for section in content_sections:
                lines.append(section)
                lines.append("")

        return "\n".join(lines).rstrip()

    def _format_tool_use(self, tool: ToolUseContent) -> str:
        """Format tool usage block."""
        lines = ["### 🛠️ Using Tool: " + tool.name]

        # Format parameters as YAML for readability
        lines.append("```yaml")
        lines.append(f"tool: {tool.name}")
        lines.append(f"id: {tool.id}")

        if tool.input:
            lines.append("parameters:")
            for key, value in tool.input.items():
                if isinstance(value, str) and "\n" in value:
                    lines.append(f"  {key}: |")
                    for line in value.split("\n"):
                        lines.append(f"    {line}")
                else:
                    lines.append(f"  {key}: {repr(value)}")

        lines.append("```")
        return "\n".join(lines)

    def _format_tool_result(self, result: ToolResultContent) -> str:
        """Format tool result block."""
        lines = []

        lines.append(f"#### ✅ Tool Result (id: {result.tool_use_id})")

        content_text = ""
        if isinstance(result.content, str):
            content_text = result.content
        elif isinstance(result.content, dict):
            if "error" in result.content:
                lines[0] = f"#### ❌ Tool Error (id: {result.tool_use_id})"
                content_text = str(result.content.get("error", "Unknown error"))
            else:
                content_text = str(result.content)
        elif isinstance(result.content, list):
            content_text = str(result.content)

        content_lines = content_text.split("\n") if content_text else []

        max_lines = self.options.max_tool_result_lines
        if self.options.collapsible_tool_results and len(content_lines) > max_lines:
            lines.append("```")
            lines.extend(content_lines[:max_lines])
            remaining = len(content_lines) - max_lines
            lines.append(f"... ({remaining} more lines)")
            lines.append("```")

            lines.append("")
            lines.append("<details>")
            lines.append("<summary>📄 Show full output</summary>")
            lines.append("")
            lines.append("```")
            lines.extend(content_lines)
            lines.append("```")
            lines.append("</details>")
        else:
            lines.append("```")
            lines.extend(content_lines)
            lines.append("```")

        return "\n".join(lines)

    def _generate_summary(self, conversation: Conversation) -> str:
        """Generate conversation summary with statistics."""
        stats = self._calculate_statistics(conversation)

        lines = [
            "---",
            "",
            "## 📊 Conversation Summary",
            "",
        ]

        # Cost breakdown
        if self.options.include_cost and stats["total_cost"] > 0:
            lines.extend(
                [
                    "### 💰 Cost Breakdown",
                    f"- **Total Cost:** ${stats['total_cost']:.4f} USD",
                    f"- **Input Tokens:** {stats['total_input_tokens']:,} tokens",
                    f"- **Output Tokens:** {stats['total_output_tokens']:,} tokens",
                ]
            )

            if stats["total_cache_tokens"] > 0:
                cache_savings = stats["total_cache_tokens"] * 0.9
                cache_tokens = stats["total_cache_tokens"]
                lines.append(
                    f"- **Cache Read Tokens:** {cache_tokens:,} tokens "
                    f"(saved ~{cache_savings:,.0f} tokens)"
                )

            lines.append("")

        # Performance metrics
        duration_hours = stats["duration_hours"]
        duration_minutes = int((duration_hours % 1) * 60)
        lines.extend(
            [
                "### 📈 Performance Metrics",
                f"- **Total Duration:** {duration_hours:.0f}h {duration_minutes}m",
                f"- **Total Messages:** {stats['total_messages']}",
                f"- **User Messages:** {stats['user_messages']}",
                f"- **Assistant Messages:** {stats['assistant_messages']}",
            ]
        )

        if stats["avg_response_time"] > 0:
            avg_time = stats["avg_response_time"]
            lines.append(f"- **Average Response Time:** {avg_time:.1f}s")

        if stats["tool_calls"] > 0:
            tool_calls = stats["tool_calls"]
            tool_success = stats["tool_success"]
            tool_errors = stats["tool_errors"]
            lines.append(
                f"- **Tool Executions:** {tool_calls} "
                f"({tool_success} successful, {tool_errors} errors)"
            )

        lines.append("")

        # Tool usage statistics
        if stats["tool_usage"]:
            lines.extend(
                [
                    "### 📊 Tool Usage Statistics",
                    "| Tool | Calls | Success Rate | Avg Duration |",
                    "|------|-------|--------------|--------------|",
                ]
            )

            for tool_name, tool_stats in sorted(stats["tool_usage"].items()):
                total_calls = tool_stats["total"]
                success_rate = 0
                if total_calls > 0:
                    success_rate = tool_stats["success"] / total_calls * 100

                avg_duration = 0
                if total_calls > 0:
                    avg_duration = tool_stats["total_duration"] / total_calls

                lines.append(
                    f"| {tool_name} | {total_calls} | {success_rate:.0f}% | {avg_duration:.1f}s |"
                )

        return "\n".join(lines)

    def _calculate_statistics(self, conversation: Conversation) -> dict[str, Any]:
        """Calculate comprehensive statistics for the conversation."""
        stats: dict[str, Any] = {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "tool_calls": 0,
            "tool_success": 0,
            "tool_errors": 0,
            "tool_usage": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_tokens": 0,
            "total_cost": 0.0,
            "duration_hours": 0.0,
            "avg_response_time": 0.0,
        }

        if conversation.entries and conversation.created_at and conversation.last_updated:
            # Calculate duration from conversation metadata
            duration = conversation.last_updated - conversation.created_at
            stats["duration_hours"] = duration.total_seconds() / 3600

        response_times = []

        for entry in conversation.entries:
            if not isinstance(entry, ConversationEntry):
                continue

            # Count messages
            if entry.message:
                stats["total_messages"] += 1
                if isinstance(entry.message, UserMessage):
                    stats["user_messages"] += 1
                elif isinstance(entry.message, AssistantMessage):
                    stats["assistant_messages"] += 1

                    # Track tool usage
                    if entry.message.content:
                        for block in entry.message.content:
                            if isinstance(block, ToolUseContent):
                                stats["tool_calls"] += 1
                                tool_name = block.name

                                tool_usage = stats["tool_usage"]
                                if tool_name not in tool_usage:
                                    tool_usage[tool_name] = {
                                        "total": 0,
                                        "success": 0,
                                        "errors": 0,
                                        "total_duration": 0.0,
                                    }

                                tool_usage[tool_name]["total"] += 1

                            elif isinstance(block, ToolResultContent):
                                for use_block in entry.message.content:
                                    if (
                                        isinstance(use_block, ToolUseContent)
                                        and use_block.id == block.tool_use_id
                                    ):
                                        tool_name = use_block.name
                                        has_error = (
                                            isinstance(block.content, dict)
                                            and "error" in block.content
                                        )
                                        if has_error:
                                            stats["tool_errors"] += 1
                                            tool_usage = stats["tool_usage"]
                                            if tool_name in tool_usage:
                                                tool_usage[tool_name]["errors"] += 1
                                        else:
                                            stats["tool_success"] += 1
                                            tool_usage = stats["tool_usage"]
                                            if tool_name in tool_usage:
                                                tool_usage[tool_name]["success"] += 1
                                        break

            # Track response times
            if entry.durationMs:
                response_times.append(entry.durationMs / 1000)

            # Token usage from message
            if isinstance(entry.message, AssistantMessage) and entry.message.usage:
                usage = entry.message.usage
                stats["total_input_tokens"] += usage.input_tokens
                stats["total_output_tokens"] += usage.output_tokens
                stats["total_cache_tokens"] += usage.cache_read_input_tokens or 0

            # Cost
            if entry.costUSD:
                stats["total_cost"] += entry.costUSD

        # Calculate average response time
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            stats["avg_response_time"] = avg_response_time

        # Estimate tool durations (rough approximation based on response times)
        tool_usage = stats["tool_usage"]
        if tool_usage and response_times:
            # Assume 30% of time is tool execution
            avg_tool_duration = stats["avg_response_time"] * 0.3
            for tool_stats in tool_usage.values():
                total_duration = tool_stats["total"] * avg_tool_duration
                tool_stats["total_duration"] = total_duration

        return stats

    def _get_message_preview(self, message: UserMessage, max_length: int = 60) -> str:
        """Get a preview of a message for TOC."""
        text = ""
        if isinstance(message.content, str):
            text = message.content
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextContent):
                    text = block.text
                    break

        preview = text.strip().split("\n")[0]
        if len(preview) > max_length:
            preview = preview[: max_length - 3] + "..."
        return preview

    def _generate_anchor(self, entry_index: int, message: BaseMessage) -> str:
        """Generate a unique anchor for a message."""
        msg_type = "user" if isinstance(message, UserMessage) else "assistant"
        return f"{msg_type}-{entry_index}"
