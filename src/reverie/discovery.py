"""
Claude Code conversation discovery system.

Finds and catalogs all conversations across Claude projects, handling path encoding
and providing metadata for the picker interface.
"""

import json
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    AnyEntry,
    ConversationEntry,
    SummaryEntry,
    SystemEntry,
    TaskInvocation,
    ToolResultEntry,
)
from .pricing import CostMode, PricingFetcher, TokenUsage, calculate_cost_for_entry

_pricing_fetcher = PricingFetcher()
_pricing_data: dict[str, Any] | None = None


def _get_pricing_data() -> dict[str, Any]:
    """Get cached pricing data, fetching if needed."""
    global _pricing_data
    if _pricing_data is None:
        _pricing_data = _pricing_fetcher.fetch_model_pricing_sync()
    return _pricing_data


@dataclass
class TimestampedCostEntry:
    """A cost/token entry with timestamp for filtering."""

    timestamp: datetime
    cost_usd: float = 0.0
    calculated_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: str | None = None
    entry_type: str = "assistant"  # "user" or "assistant"


class ConversationMetadata:
    """Lightweight metadata for a single conversation file."""

    def __init__(
        self,
        session_id: str,
        file_path: Path,
        project_path: str,
        project_name: str,
    ):
        self.session_id = session_id
        self.file_path = file_path
        self.project_path = project_path
        self.project_name = project_name
        self.entry_count = 0
        self.first_timestamp: datetime | None = None
        self.last_timestamp: datetime | None = None
        self.has_errors = False
        self.error_message: str | None = None
        self.summary: str | None = None
        self.total_cost_usd: float = 0.0
        self.calculated_cost_usd: float = 0.0

        # Token usage statistics
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_creation_input_tokens: int = 0
        self.total_cache_read_input_tokens: int = 0

        # Model usage tracking
        self.models_used: set[str] = set()
        self.assistant_turns: int = 0
        self.user_turns: int = 0
        self.tool_result_turns: int = 0

        # Additional interesting metadata
        self.total_duration_seconds: float = 0.0
        self.working_directories: set[str] = set()

        # Timestamped entries for timeframe filtering
        self.timestamped_entries: list[TimestampedCostEntry] = []

        # Task tool invocations (for parent conversations)
        self.task_invocations: list[TaskInvocation] = []

    def clone(self) -> "ConversationMetadata":
        """Create a deep copy of this metadata object."""
        new_conv = ConversationMetadata(
            self.session_id, self.file_path, self.project_path, self.project_name
        )

        # Copy all simple attributes
        new_conv.entry_count = self.entry_count
        new_conv.first_timestamp = self.first_timestamp
        new_conv.last_timestamp = self.last_timestamp
        new_conv.has_errors = self.has_errors
        new_conv.error_message = self.error_message
        new_conv.summary = self.summary
        new_conv.total_cost_usd = self.total_cost_usd
        new_conv.calculated_cost_usd = self.calculated_cost_usd
        new_conv.total_input_tokens = self.total_input_tokens
        new_conv.total_output_tokens = self.total_output_tokens
        new_conv.total_cache_creation_input_tokens = self.total_cache_creation_input_tokens
        new_conv.total_cache_read_input_tokens = self.total_cache_read_input_tokens
        new_conv.assistant_turns = self.assistant_turns
        new_conv.user_turns = self.user_turns
        new_conv.tool_result_turns = self.tool_result_turns
        new_conv.total_duration_seconds = self.total_duration_seconds

        # Deep copy collections
        new_conv.models_used = self.models_used.copy()
        new_conv.working_directories = self.working_directories.copy()
        new_conv.timestamped_entries = self.timestamped_entries.copy()
        new_conv.task_invocations = self.task_invocations.copy()

        return new_conv

    def __repr__(self) -> str:
        return (
            f"ConversationMetadata(session_id='{self.session_id}', project='{self.project_name}')"
        )


class ProjectMetadata:
    """Metadata for all conversations in a project."""

    def __init__(self, project_path: str, project_name: str):
        self.project_path = project_path
        self.project_name = project_name
        self.conversations: list[ConversationMetadata] = []

    @property
    def conversation_count(self) -> int:
        return len(self.conversations)

    @property
    def total_entries(self) -> int:
        return sum(conv.entry_count for conv in self.conversations)

    @property
    def latest_conversation(self) -> ConversationMetadata | None:
        if not self.conversations:
            return None
        conversations_with_timestamps = [conv for conv in self.conversations if conv.last_timestamp]
        if not conversations_with_timestamps:
            return None
        return max(conversations_with_timestamps, key=lambda c: c.last_timestamp or datetime.min)

    @property
    def total_cost_usd(self) -> float:
        return sum(conv.total_cost_usd for conv in self.conversations)

    @property
    def calculated_cost_usd(self) -> float:
        return sum(conv.calculated_cost_usd for conv in self.conversations)

    @property
    def total_input_tokens(self) -> int:
        return sum(conv.total_input_tokens for conv in self.conversations)

    @property
    def total_output_tokens(self) -> int:
        return sum(conv.total_output_tokens for conv in self.conversations)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(conv.total_cache_creation_input_tokens for conv in self.conversations)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(conv.total_cache_read_input_tokens for conv in self.conversations)

    @property
    def all_models_used(self) -> set[str]:
        models = set()
        for conv in self.conversations:
            models.update(conv.models_used)
        return models

    @property
    def total_assistant_turns(self) -> int:
        return sum(conv.assistant_turns for conv in self.conversations)

    @property
    def total_user_turns(self) -> int:
        return sum(conv.user_turns for conv in self.conversations)

    @property
    def total_duration_seconds(self) -> float:
        return sum(conv.total_duration_seconds for conv in self.conversations)

    @property
    def all_working_directories(self) -> set[str]:
        dirs = set()
        for conv in self.conversations:
            dirs.update(conv.working_directories)
        return dirs

    def __repr__(self) -> str:
        return (
            f"ProjectMetadata(name='{self.project_name}', conversations={self.conversation_count})"
        )


def decode_project_path(encoded_path: str) -> str:
    """Decode Claude's project path encoding (- back to /)."""
    return encoded_path.replace("-", "/")


def _get_fallback_project_info(project_dir: Path) -> dict[str, str]:
    """Get fallback project info from encoded directory name."""
    decoded_path = decode_project_path(project_dir.name)
    return {"path": decoded_path, "name": Path(decoded_path).name}


def find_claude_projects_dir() -> Path:
    """Find the Claude projects directory."""
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        raise FileNotFoundError(
            f"Claude projects directory not found at {claude_dir}. "
            "Is Claude Code installed and has it created any projects?"
        )
    return claude_dir


def discover_projects() -> list[ProjectMetadata]:
    """
    Discover all Claude projects and their conversations.

    Returns:
        List of ProjectMetadata objects, sorted by most recent activity.
    """
    projects_dir = find_claude_projects_dir()
    projects: list[ProjectMetadata] = []

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue

        project_info = _get_fallback_project_info(project_dir)
        project_meta = ProjectMetadata(project_info["path"], project_info["name"])

        real_project_path = None
        for jsonl_file in project_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            conv_meta = ConversationMetadata(
                session_id=session_id,
                file_path=jsonl_file,
                project_path=project_info["path"],
                project_name=project_info["name"],
            )

            found_project_path = _extract_conversation_metadata(conv_meta)
            if real_project_path is None and found_project_path:
                real_project_path = found_project_path

            project_meta.conversations.append(conv_meta)

        # Update all project and conversation metadata once if we found the real path
        if real_project_path:
            real_project_name = Path(real_project_path).name
            project_meta.project_path = real_project_path
            project_meta.project_name = real_project_name
            for conv in project_meta.conversations:
                conv.project_path = real_project_path
                conv.project_name = real_project_name

        if project_meta.conversations:  # Only include projects with conversations
            # Sort conversations by most recent activity (newest first)
            project_meta.conversations.sort(
                key=lambda c: (
                    c.last_timestamp.replace(tzinfo=None) if c.last_timestamp else datetime.min
                ),
                reverse=True,
            )
            projects.append(project_meta)

    # Sort by most recent activity
    return sorted(
        projects,
        key=lambda p: (
            p.latest_conversation.last_timestamp
            if p.latest_conversation and p.latest_conversation.last_timestamp
            else datetime.min
        ),
        reverse=True,
    )


def _extract_conversation_metadata(conv_meta: ConversationMetadata) -> str | None:
    """
    Extract metadata from a conversation JSONL file.

    Reads the file to get entry count, timestamps, summary, token usage,
    and other interesting metadata. Returns the real project path if found.
    Does not load the full conversation into memory.
    """
    real_project_path = None

    # Get pricing data for cost calculation
    pricing_data = _get_pricing_data()

    try:
        with conv_meta.file_path.open("r", encoding="utf-8") as f:
            entry_count = 0
            first_timestamp = None
            last_timestamp = None
            summary = None
            total_cost = 0.0
            calculated_cost = 0.0

            # Token counters
            total_input_tokens = 0
            total_output_tokens = 0
            total_cache_creation_tokens = 0
            total_cache_read_tokens = 0

            # Tracking counters and sets
            models_used = set()
            assistant_turns = 0
            user_turns = 0
            tool_result_turns = 0
            total_duration = 0.0
            working_directories = set()

            # Tracking for single-pass Task result collection
            task_results: dict[str, dict[str, Any]] = {}  # tool_use_id -> result data
            task_id_to_model: dict[str, str] = {}  # tool_use_id -> model used
            current_model: str | None = None  # Track current model for each entry

            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    entry_count += 1
                    entry_type = data.get("type", "")

                    # Extract real project info from cwd field
                    if "cwd" in data:
                        cwd = data["cwd"]
                        if cwd and cwd != ".":
                            if real_project_path is None:
                                real_project_path = cwd
                            working_directories.add(cwd)

                    # Extract timestamp
                    timestamp = None
                    if "timestamp" in data:
                        timestamp_str = data["timestamp"]
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            if first_timestamp is None:
                                first_timestamp = timestamp
                            last_timestamp = timestamp
                        except (ValueError, TypeError):
                            timestamp = None

                    if entry_type == "summary" and "summary" in data:
                        summary = data["summary"]

                    if entry_type == "user":
                        # Check if this is a tool result or actual user message
                        if "toolUseResult" in data:
                            # This is a tool result, not a user message
                            tool_result_turns += 1
                        else:
                            # This is an actual user message
                            user_turns += 1
                            # Create timestamped entry for user messages (no cost/tokens)
                            # Use timestamp if available, fallback to first_timestamp
                            entry_timestamp = timestamp or first_timestamp
                            if entry_timestamp:
                                user_entry = TimestampedCostEntry(
                                    timestamp=entry_timestamp, entry_type="user"
                                )
                                conv_meta.timestamped_entries.append(user_entry)

                        # Check for Task tool results in user messages
                        if "toolUseResult" in data:
                            message = data.get("message", {})
                            if isinstance(message, dict) and "content" in message:
                                content = message.get("content", [])
                                if isinstance(content, list):
                                    for content_block in content:
                                        if (
                                            isinstance(content_block, dict)
                                            and content_block.get("type") == "tool_result"
                                        ):
                                            tool_use_id = content_block.get("tool_use_id")
                                            if tool_use_id:
                                                # Store the result for later matching
                                                task_results[tool_use_id] = data["toolUseResult"]

                    if entry_type == "assistant":
                        assistant_turns += 1

                        if "costUSD" in data:
                            total_cost += data.get("costUSD", 0.0)

                        if "durationSeconds" in data:
                            total_duration += data.get("durationSeconds", 0.0)

                        # Analyze message content for artifacts and detailed token usage
                        if "message" in data and isinstance(data["message"], dict):
                            message = data["message"]

                            model = message.get("model")
                            if model:
                                models_used.add(model)
                                current_model = model  # Track current model for Task associations

                            # Track Task tool invocations
                            if "content" in message and isinstance(message["content"], list):
                                for content_block in message["content"]:
                                    if (
                                        isinstance(content_block, dict)
                                        and content_block.get("type") == "tool_use"
                                        and content_block.get("name") == "Task"
                                    ):
                                        # Extract Task tool invocation details
                                        task_input = content_block.get("input", {})
                                        tool_use_id = content_block.get("id", "")
                                        task_invocation = TaskInvocation(
                                            tool_use_id=tool_use_id,
                                            description=task_input.get("description", ""),
                                            prompt=task_input.get("prompt", ""),
                                            timestamp=timestamp,
                                            entry_uuid=data.get("uuid"),
                                        )
                                        conv_meta.task_invocations.append(task_invocation)

                                        # Store model association for this task
                                        if current_model and tool_use_id:
                                            task_id_to_model[tool_use_id] = current_model

                            # Extract detailed token usage
                            if "usage" in message and isinstance(message.get("usage"), dict):
                                usage_data = message["usage"]

                                input_tokens = usage_data.get("input_tokens", 0)
                                output_tokens = usage_data.get("output_tokens", 0)
                                cache_creation = usage_data.get("cache_creation_input_tokens", 0)
                                cache_read = usage_data.get("cache_read_input_tokens", 0)

                                total_input_tokens += input_tokens
                                total_output_tokens += output_tokens
                                total_cache_creation_tokens += cache_creation or 0
                                total_cache_read_tokens += cache_read or 0

                                # Create usage object for cost calculation
                                usage = TokenUsage(
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    cache_creation_input_tokens=cache_creation,
                                    cache_read_input_tokens=cache_read,
                                )

                                # Calculate cost (always calculate for comparison)
                                calc_cost = calculate_cost_for_entry(
                                    cost_usd=data.get("costUSD"),
                                    model=model,
                                    usage=usage,
                                    mode=CostMode.CALCULATE,  # Always calculate
                                    pricing_fetcher=_pricing_fetcher,
                                    pricing_data=pricing_data,
                                )
                                calculated_cost += calc_cost

                                # Create timestamped entry for this assistant response
                                # Use timestamp if available, fallback to first_timestamp
                                # if we have cost data
                                entry_timestamp = timestamp
                                if not entry_timestamp and (
                                    data.get("costUSD", 0.0) > 0 or calc_cost > 0
                                ):
                                    # Use first_timestamp as fallback for entries with cost
                                    # but no timestamp
                                    entry_timestamp = first_timestamp

                                if entry_timestamp:
                                    assistant_entry = TimestampedCostEntry(
                                        timestamp=entry_timestamp,
                                        cost_usd=data.get("costUSD", 0.0),
                                        calculated_cost_usd=calc_cost,
                                        input_tokens=input_tokens,
                                        output_tokens=output_tokens,
                                        cache_creation_tokens=cache_creation or 0,
                                        cache_read_tokens=cache_read or 0,
                                        model=model,
                                        entry_type="assistant",
                                    )
                                    conv_meta.timestamped_entries.append(assistant_entry)

                except json.JSONDecodeError as e:
                    conv_meta.has_errors = True
                    conv_meta.error_message = f"JSON error on line {line_num}: {e}"
                    continue

            # Store all the extracted metadata
            conv_meta.entry_count = entry_count
            conv_meta.first_timestamp = first_timestamp
            conv_meta.last_timestamp = last_timestamp
            conv_meta.summary = summary
            conv_meta.total_cost_usd = total_cost
            conv_meta.calculated_cost_usd = calculated_cost

            # Store token usage statistics
            conv_meta.total_input_tokens = total_input_tokens
            conv_meta.total_output_tokens = total_output_tokens
            conv_meta.total_cache_creation_input_tokens = total_cache_creation_tokens
            conv_meta.total_cache_read_input_tokens = total_cache_read_tokens

            # Store tracking information
            conv_meta.models_used = models_used
            conv_meta.assistant_turns = assistant_turns
            conv_meta.user_turns = user_turns
            conv_meta.tool_result_turns = tool_result_turns
            conv_meta.total_duration_seconds = total_duration
            conv_meta.working_directories = working_directories

            # Match Task tool results with invocations (O(1) dictionary lookup)
            if conv_meta.task_invocations and task_results:
                for task in conv_meta.task_invocations:
                    if task.tool_use_id in task_results:
                        # Extract result metadata
                        result = task_results[task.tool_use_id]
                        if isinstance(result, dict):
                            task.total_duration_ms = result.get("totalDurationMs")
                            task.total_tokens = result.get("totalTokens")
                            task.total_tool_use_count = result.get("totalToolUseCount")

                            # Extract usage details
                            if "usage" in result and isinstance(result["usage"], dict):
                                usage_data = result["usage"]
                                task.result_input_tokens = usage_data.get("input_tokens", 0)
                                task.result_output_tokens = usage_data.get("output_tokens", 0)
                                task.result_cache_creation_tokens = usage_data.get(
                                    "cache_creation_input_tokens", 0
                                )
                                task.result_cache_read_tokens = usage_data.get(
                                    "cache_read_input_tokens", 0
                                )

    except (OSError, UnicodeDecodeError) as e:
        conv_meta.has_errors = True
        conv_meta.error_message = f"File read error: {e}"

    return real_project_path


def find_conversation_by_session_id(session_id: str) -> ConversationMetadata | None:
    """
    Find a specific conversation by session ID across all projects.

    Args:
        session_id: The UUID session identifier

    Returns:
        ConversationMetadata if found, None otherwise
    """
    projects = discover_projects()
    for project in projects:
        for conversation in project.conversations:
            if conversation.session_id == session_id:
                return conversation
    return None


def stream_conversation_entries(conv_meta: ConversationMetadata) -> Generator[AnyEntry, None, None]:
    """
    Stream entries from a conversation file without loading everything into memory.

    Args:
        conv_meta: Conversation metadata with file path

    Yields:
        Parsed conversation entries (ConversationEntry, SummaryEntry, or SystemEntry)

    Raises:
        FileNotFoundError: If the conversation file doesn't exist
        ValueError: If entries cannot be parsed
    """
    with conv_meta.file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Determine entry type and parse accordingly
                entry_type = data.get("type")
                if entry_type == "summary":
                    yield SummaryEntry.model_validate(data)
                elif entry_type == "system":
                    yield SystemEntry.model_validate(data)
                elif entry_type == "user" and "toolUseResult" in data:
                    # User entries with toolUseResult are tool execution results
                    yield ToolResultEntry.model_validate(data)
                else:
                    # Default to ConversationEntry for user/assistant messages
                    yield ConversationEntry.model_validate(data)

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            except Exception as e:
                raise ValueError(f"Failed to parse entry on line {line_num}: {e}") from e


def filter_projects_by_timeframe(
    projects: list[ProjectMetadata], timeframe_days: int | None
) -> list[ProjectMetadata]:
    """
    Filter projects and conversations by timeframe.

    Args:
        projects: List of all projects to filter
        timeframe_days: Number of days to look back, or None for all time

    Returns:
        Filtered list of projects with recalculated costs for the timeframe
    """
    if timeframe_days is None:
        return projects[:]

    # Calculate cutoff date
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    cutoff = now - timedelta(days=timeframe_days)

    # Filter projects and conversations
    filtered_projects = []

    for project in projects:
        # Create a new project metadata with filtered conversations
        filtered_project = ProjectMetadata(project.project_path, project.project_name)

        # Filter conversations by timestamp
        for conv in project.conversations:
            # Check if conversation has any activity within the timeframe
            if conv.last_timestamp and conv.last_timestamp >= cutoff:
                # Use the robust clone method
                filtered_conv = conv.clone()

                # Filter task invocations by timestamp
                filtered_conv.task_invocations = [
                    task
                    for task in conv.task_invocations
                    if not task.timestamp or task.timestamp >= cutoff
                ]

                # Recalculate totals based on timestamped entries within the timeframe
                filtered_conv.total_cost_usd = 0.0
                filtered_conv.calculated_cost_usd = 0.0
                filtered_conv.total_input_tokens = 0
                filtered_conv.total_output_tokens = 0
                filtered_conv.total_cache_creation_input_tokens = 0
                filtered_conv.total_cache_read_input_tokens = 0
                filtered_conv.assistant_turns = 0
                filtered_conv.user_turns = 0
                # Keep original duration
                filtered_conv.total_duration_seconds = conv.total_duration_seconds

                # Filter timestamped entries and recalculate
                for entry in conv.timestamped_entries:
                    # Use entry timestamp if available, otherwise check if we should
                    # include based on conversation
                    entry_timestamp = entry.timestamp

                    # Include entry if it has a timestamp within range OR if it has cost data
                    # (cost data without timestamp uses conversation start as fallback)
                    if entry_timestamp >= cutoff or (
                        entry.entry_type == "assistant"
                        and (entry.cost_usd > 0 or entry.calculated_cost_usd > 0)
                        and conv.first_timestamp
                        and conv.first_timestamp >= cutoff
                    ):
                        filtered_conv.total_cost_usd += entry.cost_usd
                        filtered_conv.calculated_cost_usd += entry.calculated_cost_usd
                        filtered_conv.total_input_tokens += entry.input_tokens
                        filtered_conv.total_output_tokens += entry.output_tokens
                        filtered_conv.total_cache_creation_input_tokens += (
                            entry.cache_creation_tokens
                        )
                        filtered_conv.total_cache_read_input_tokens += entry.cache_read_tokens

                        if entry.entry_type == "assistant":
                            filtered_conv.assistant_turns += 1
                        elif entry.entry_type == "user":
                            filtered_conv.user_turns += 1

                # Copy the filtered timestamped entries
                filtered_conv.timestamped_entries = [
                    entry
                    for entry in conv.timestamped_entries
                    if entry.timestamp >= cutoff
                    or (
                        entry.entry_type == "assistant"
                        and (entry.cost_usd > 0 or entry.calculated_cost_usd > 0)
                        and conv.first_timestamp
                        and conv.first_timestamp >= cutoff
                    )
                ]

                filtered_project.conversations.append(filtered_conv)

        # Only include projects with conversations after filtering
        if filtered_project.conversations:
            filtered_projects.append(filtered_project)

    return filtered_projects
