"""
Interactive conversation picker with Rich TUI.

Terminal interface for browsing and selecting Claude Code conversations
across all projects with smooth keyboard navigation and instant preview.
"""

import contextlib
import sys
import termios
import tty
from datetime import datetime
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .discovery import (
    ConversationMetadata,
    ProjectMetadata,
    discover_projects,
    filter_projects_by_timeframe,
    stream_conversation_entries,
)
from .models import ConversationEntry, TaskInvocation

# ===== COLOR PALETTE CONSTANTS =====
COLOR_BRIGHT_ORANGE = "#F6C244"  # Primary orange accent
COLOR_ORANGE = "#CA7B5E"  # Darker orange for less emphasis
COLOR_LIGHT_BROWN = "#D6BDA0"  # Light brown
COLOR_WARM_BROWN = "#C49F81"  # Medium brown
COLOR_DARK_BROWN = "#A16343"  # Dark brown
COLOR_DEEP_BROWN = "#995434"  # Darker brown
COLOR_OFF_WHITE = "#E5E5E5"  # Light text color
COLOR_TEXT_DARK = "#0A0A0A"  # Very dark background
COLOR_ACCENT_BLUE = "#B6D6FB"  # Muted blue-gray for actions
COLOR_LIGHT_GRAY = "#999999"  # Light gray
COLOR_MEDIUM_GRAY = "#666666"  # Medium gray
COLOR_DARK_GRAY = "#4A4A4A"  # Dark gray
COLOR_DEEP_GRAY = "#333333"  # Darker gray
COLOR_GREY11 = "#1A1A1A"  # Dark surface background

# Style modifiers
STYLE_BOLD = "bold"
STYLE_DIM = "dim"
STYLE_ITALIC = "italic"

# Background colors
BG_ORANGE = f"on {COLOR_ORANGE}"
BG_WARM_BROWN = f"on {COLOR_WARM_BROWN}"
BG_DARK = f"on {COLOR_GREY11}"

# ===== COMPOSITE STYLE CONSTANTS =====

# Selection highlights
PROJECT_HIGHLIGHT_STYLE = f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}"
CONVERSATION_HIGHLIGHT_STYLE = f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}"

# Text styles
SUMMARY_STYLE = f"{STYLE_ITALIC} {COLOR_ACCENT_BLUE}"
USER_MESSAGE_STYLE = f"{STYLE_BOLD} {COLOR_OFF_WHITE}"
TIMESTAMP_STYLE = f"{STYLE_DIM} {COLOR_ORANGE}"
METADATA_LABEL_STYLE = f"{STYLE_BOLD} {COLOR_ACCENT_BLUE}"
METADATA_VALUE_STYLE = COLOR_OFF_WHITE

# Border styles - prominent but monochromatic
HEADER_BORDER_STYLE = COLOR_BRIGHT_ORANGE  # Orange for header prominence
PROJECTS_BORDER_STYLE = COLOR_DARK_GRAY  # Subtle gray
PREVIEW_BORDER_STYLE = COLOR_DARK_GRAY  # Subtle gray
FOOTER_BORDER_STYLE = "#6A5ACD"

# Table headers - monochromatic
TABLE_HEADER_PROJECTS = f"{STYLE_BOLD} {COLOR_OFF_WHITE}"
TABLE_HEADER_CONVERSATIONS = f"{STYLE_BOLD} {COLOR_OFF_WHITE}"

# Key display styles
KEY_STYLE = f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}"
QUIT_KEY_STYLE = f"{STYLE_BOLD} {COLOR_WARM_BROWN}"
SELECT_KEY_STYLE = f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}"
ACTION_TEXT_STYLE = COLOR_OFF_WHITE

# Error styles
ERROR_STYLE = COLOR_ORANGE
ERROR_BOLD_STYLE = f"{COLOR_BRIGHT_ORANGE} {STYLE_BOLD}"

# Info styles - monochromatic
INFO_STYLE = COLOR_OFF_WHITE
SUCCESS_STYLE = COLOR_ACCENT_BLUE
WARNING_STYLE = COLOR_BRIGHT_ORANGE

# Separator/dim styles
SEPARATOR_STYLE = STYLE_DIM
BORDER_DIM_STYLE = STYLE_DIM

# Panel background style
PANEL_BG_STYLE = BG_DARK

COLOR_BACKGROUND = COLOR_TEXT_DARK  # Main background (#0A0A0A)
COLOR_SURFACE = COLOR_GREY11  # Panel/surface background (#1A1A1A)

# Enhanced color theming
COST_COLOR_SCALE = [
    (0.01, f"{STYLE_DIM} {COLOR_LIGHT_BROWN}"),  # < 1¢
    (2.50, COLOR_LIGHT_BROWN),  # < $2.50
    (5.00, COLOR_WARM_BROWN),  # < $5
    (10.00, COLOR_DARK_BROWN),  # < $10
    (20.00, COLOR_BRIGHT_ORANGE),  # < $20
    (float("inf"), f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}"),  # $20+
]

TOKEN_COLOR_SCALE = [
    (1_000, f"{STYLE_DIM} {COLOR_LIGHT_BROWN}"),  # < 1K
    (10_000, COLOR_LIGHT_BROWN),  # < 10K
    (25_000, COLOR_WARM_BROWN),  # < 25K
    (75_000, COLOR_DARK_BROWN),  # < 75K
    (150_000, COLOR_BRIGHT_ORANGE),  # < 150K
    (float("inf"), f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}"),  # 150K+
]

# Layout constants
HEADER_SIZE = 5  # Increased to accommodate timeframe line
FOOTER_SIZE = 3
MAIN_MIN_SIZE = 15
PROJECTS_RATIO = 2
PREVIEW_RATIO = 3

# UI overhead constants (for viewport sizing calculations)
PANEL_BORDER_HEIGHT = 4  # top/bottom borders for panels (2 panels × 2 borders)
TABLE_HEADER_HEIGHT = 2  # projects table header + conversations table header
SCROLL_INDICATOR_HEIGHT = 2  # one for projects, one for conversations
UI_MARGINS_HEIGHT = 2  # spacing between elements

# Scrolling constants
MAX_CONVERSATIONS_VIEWPORT_SIZE = 10  # Max conversations to show at once
MAX_PROJECTS_VIEWPORT_SIZE = 8  # Max projects to show at once

# Key mappings for cleaner code
KEY_MAPPINGS = {
    "quit": ("q", "\x1b"),
    "help": ("?",),
    "select": ("\r", "\n"),
    "down": ("j", "\x1b[B"),
    "up": ("k", "\x1b[A"),
    "left": ("h", "\x1b[D"),
    "right": ("l", "\x1b[C"),
    "top": ("g",),
    "bottom": ("G",),
    "refresh": ("r",),
    "timeframe": ("t", "T"),
}

# Timeframe options in days (None = all time)
TIMEFRAME_OPTIONS = [1, 7, 30, None]
TIMEFRAME_LABELS = {
    1: "Last 24 hours",
    7: "Last 7 days",
    30: "Last 30 days",
    None: "All time",
}


class ConversationPicker:
    """Interactive TUI for selecting Claude Code conversations."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.all_projects: list[ProjectMetadata] = []  # Unfiltered projects
        self.projects: list[ProjectMetadata] = []  # Filtered projects
        self.current_project_index = 0
        self.current_conversation_index = 0
        self.show_help = False
        self.ui_dirty = True
        self._conversations_viewport_size = MAX_CONVERSATIONS_VIEWPORT_SIZE
        self._projects_viewport_size = MAX_PROJECTS_VIEWPORT_SIZE
        self._first_message_cache: dict[str, list[str]] = {}
        self._loading_conversation: str | None = None
        # Timeframe filtering
        self.current_timeframe_index = 1  # Default to 7 days
        self.timeframe_days = TIMEFRAME_OPTIONS[self.current_timeframe_index]

    def run(self) -> ConversationMetadata | None:
        """Run the interactive picker and return selected conversation."""
        try:
            if not self._discover_conversations():
                return None
            return self._run_interactive_loop()
        except KeyboardInterrupt:
            return None

    def _discover_conversations(self) -> bool:
        """Discover conversations and show error if none found."""
        self.console.print("🔍 Discovering Claude Code conversations...", style=INFO_STYLE)
        self.all_projects = discover_projects()

        if not self.all_projects:
            self.console.print("❌ No Claude Code conversations found!", style=ERROR_BOLD_STYLE)
            self.console.print("Make sure Claude Code has created some conversations first.")
            return False

        # Apply initial timeframe filter
        self._apply_timeframe_filter()
        return True

    def _run_interactive_loop(self) -> ConversationMetadata | None:
        """Run the main interactive event loop."""
        with Live(
            self._build_layout(),
            console=self.console,
            auto_refresh=False,
            screen=True,
        ) as live:
            while True:
                key = self._get_key()
                result = self._handle_keypress(key)

                if result != "continue":  # Either selection or quit
                    return result if isinstance(result, ConversationMetadata) else None

                if self.ui_dirty:
                    live.update(self._build_layout())
                    live.refresh()
                    self.ui_dirty = False

    def _handle_keypress(self, key: str) -> ConversationMetadata | None | str:
        """Handle keypress and return conversation if selected, None to quit,
        or 'continue'."""
        if key in KEY_MAPPINGS["quit"]:
            return None
        elif key in KEY_MAPPINGS["help"]:
            self.show_help = not self.show_help
            self.ui_dirty = True
        elif key in KEY_MAPPINGS["select"]:
            return self._select_current_conversation()
        elif key in KEY_MAPPINGS["down"]:
            self._move_down()
        elif key in KEY_MAPPINGS["up"]:
            self._move_up()
        elif key in KEY_MAPPINGS["left"]:
            self._move_to_previous_project()
        elif key in KEY_MAPPINGS["right"]:
            self._move_to_next_project()
        elif key in KEY_MAPPINGS["top"]:
            self._move_to_top()
        elif key in KEY_MAPPINGS["bottom"]:
            self._move_to_bottom()
        elif key in KEY_MAPPINGS["refresh"]:
            self._refresh_conversations()
        elif key in KEY_MAPPINGS["timeframe"]:
            self._cycle_timeframe()

        return "continue"  # Continue loop

    def _build_layout(self) -> Layout:
        """Build the main TUI layout."""
        # Adjust viewport size based on terminal height
        self._adjust_viewport_for_terminal_height()

        layout = Layout()

        # Split into header, main content, and footer
        layout.split(
            Layout(name="spacer", size=1),  # Add spacer at top
            Layout(name="header", size=HEADER_SIZE),
            Layout(name="main", minimum_size=MAIN_MIN_SIZE),
            Layout(name="footer", size=FOOTER_SIZE),
        )

        # Main content - split into projects and preview
        layout["main"].split_row(
            Layout(name="projects", ratio=PROJECTS_RATIO),
            Layout(name="preview", ratio=PREVIEW_RATIO),
        )

        # Update all sections
        layout["spacer"].update("")  # Empty spacer
        layout["header"].update(self._build_header())
        layout["main"]["projects"].update(self._build_projects_panel())
        layout["main"]["preview"].update(self._build_preview_panel())
        layout["footer"].update(self._build_footer())

        return layout

    def _adjust_viewport_for_terminal_height(self) -> None:
        """Adjust viewport sizes based on available terminal height."""
        terminal_height = self.console.size.height

        # Calculate UI overhead using constants
        ui_overhead = (
            1  # spacer
            + HEADER_SIZE
            + FOOTER_SIZE
            + PANEL_BORDER_HEIGHT
            + TABLE_HEADER_HEIGHT
            + SCROLL_INDICATOR_HEIGHT
            + UI_MARGINS_HEIGHT
        )
        available = max(4, terminal_height - ui_overhead)

        # Be very conservative - ensure scroll indicators always have space
        if available <= 8:  # Small terminal - be very aggressive
            self._projects_viewport_size = min(2, max(1, available // 3))
            self._conversations_viewport_size = min(
                3, max(2, available - self._projects_viewport_size - 2)
            )  # Reserve 2 lines for indicators
        elif available <= 15:  # Medium terminal
            self._projects_viewport_size = min(3, available // 4)
            self._conversations_viewport_size = min(
                6, available - self._projects_viewport_size - 2
            )  # Reserve 2 lines for indicators
        else:  # Large terminal
            # Even in large terminals, be conservative if not truly large
            if available < 20:
                self._projects_viewport_size = min(MAX_PROJECTS_VIEWPORT_SIZE, available // 3)
                self._conversations_viewport_size = min(
                    MAX_CONVERSATIONS_VIEWPORT_SIZE,
                    available - self._projects_viewport_size - 2,
                )
            else:
                self._projects_viewport_size = MAX_PROJECTS_VIEWPORT_SIZE
                self._conversations_viewport_size = MAX_CONVERSATIONS_VIEWPORT_SIZE

    def _format_relative_date(self, conversation: ConversationMetadata | None) -> str:
        """Format conversation timestamp as relative date."""
        if not conversation or not conversation.last_timestamp:
            return "Never"

        latest = conversation.last_timestamp.astimezone()
        now_local = datetime.now().astimezone()
        days_ago = (now_local.date() - latest.date()).days

        if days_ago == 0:
            return "Today"
        elif days_ago == 1:
            return "Yesterday"
        elif days_ago < 7:
            return f"{days_ago}d ago"
        else:
            return latest.strftime("%m/%d")

    def _format_cost(self, cost: float) -> str:
        """Format cost in a sleek, compact way."""
        if cost < 0.01:
            return "< 1¢"
        elif cost < 1.00:
            return f"{int(cost * 100)}¢"
        elif cost < 100:
            return f"${cost:.2f}"
        elif cost < 1000:
            return f"${cost:.0f}"
        else:
            return f"${cost / 1000:.1f}k"

    def _get_cost_color(self, cost: float) -> str:
        """Get color style based on cost amount."""
        for threshold, color in COST_COLOR_SCALE:
            if cost < threshold:
                return color
        return COST_COLOR_SCALE[-1][1]

    def _get_token_color(self, tokens: int) -> str:
        """Get color style based on token count."""
        for threshold, color in TOKEN_COLOR_SCALE:
            if tokens < threshold:
                return color
        return TOKEN_COLOR_SCALE[-1][1]

    def _get_title_colors(self) -> list[str]:
        return [
            COLOR_BRIGHT_ORANGE,
            "#F5A85D",
            "#E88B7D",
            "#C97AA8",
            "#9B72CF",
            "#7B68EE",
            "#6A5ACD",
        ]

    def _build_header(self) -> Panel:
        """Build header panel with beautiful gradient effect."""
        title_colors = self._get_title_colors()
        title_text = Text()
        title_text.append("r", style=f"{STYLE_BOLD} {title_colors[0]}")
        title_text.append("e", style=f"{STYLE_BOLD} {title_colors[1]}")
        title_text.append("v", style=f"{STYLE_BOLD} {title_colors[2]}")
        title_text.append("e", style=f"{STYLE_BOLD} {title_colors[3]}")
        title_text.append("r", style=f"{STYLE_BOLD} {title_colors[4]}")
        title_text.append("i", style=f"{STYLE_BOLD} {title_colors[5]}")
        title_text.append("e", style=f"{STYLE_BOLD} {title_colors[6]}")

        # Stats line with icons - consistent spacing
        total_convs = sum(p.conversation_count for p in self.projects)
        total_cost = sum(p.calculated_cost_usd for p in self.projects)

        stats_text = Text()
        stats_text.append("📁 ", style=STYLE_BOLD)
        stats_text.append(f"{len(self.projects)} projects", style=COLOR_ACCENT_BLUE)
        stats_text.append("   •   ", style=SEPARATOR_STYLE)  # Consistent spacing
        stats_text.append("💬 ", style=STYLE_BOLD)
        stats_text.append(f"{total_convs} conversations", style=COLOR_WARM_BROWN)

        # Add total tokens with cache breakdown
        total_all_tokens = sum(p.total_input_tokens + p.total_output_tokens for p in self.projects)
        total_cached = sum(
            p.total_cache_creation_tokens + p.total_cache_read_tokens for p in self.projects
        )

        if total_all_tokens > 0:
            stats_text.append("   •   ", style=SEPARATOR_STYLE)  # Consistent spacing
            stats_text.append("🔶 ", style=STYLE_BOLD)

            # Format total
            if total_all_tokens >= 1_000_000:
                total_str = f"{total_all_tokens / 1_000_000:.1f}M"
            elif total_all_tokens >= 1_000:
                total_str = f"{total_all_tokens / 1_000:.0f}K"
            else:
                total_str = str(total_all_tokens)

            stats_text.append(f"{total_str} total", style=COLOR_BRIGHT_ORANGE)

            # Add cache breakdown if present
            if total_cached > 0:
                if total_cached >= 1_000_000:
                    cache_str = f"{total_cached / 1_000_000:.1f}M"
                elif total_cached >= 1_000:
                    cache_str = f"{total_cached / 1_000:.0f}K"
                else:
                    cache_str = str(total_cached)
                stats_text.append(
                    f" (💾 {cache_str} cached)", style=f"{STYLE_DIM} {COLOR_ACCENT_BLUE}"
                )

        stats_text.append("   •   ", style=SEPARATOR_STYLE)  # Consistent spacing
        stats_text.append("💰 ", style=STYLE_BOLD)
        stats_text.append(self._format_cost(total_cost), style=f"{STYLE_BOLD} {COLOR_ORANGE}")

        # Add timeframe indicator
        timeframe_text = Text()
        timeframe_text.append("📅 ", style=STYLE_BOLD)
        timeframe_label = TIMEFRAME_LABELS[self.timeframe_days]
        timeframe_text.append(timeframe_label, style=f"{STYLE_BOLD} {COLOR_ACCENT_BLUE}")

        # Center each line individually
        header_content = Group(
            Align.center(title_text), Align.center(stats_text), Align.center(timeframe_text)
        )

        return Panel(
            header_content,
            box=box.DOUBLE,
            border_style=HEADER_BORDER_STYLE,
            padding=(0, 2),
        )

    def _build_projects_panel(self) -> Panel:
        """Build the projects listing panel."""
        if not self.projects:
            timeframe_label = TIMEFRAME_LABELS[self.timeframe_days]
            return Panel(
                f"No conversations found in {timeframe_label.lower()}",
                title="Projects",
                box=box.ROUNDED,
            )

        # Add projects table with viewport scrolling
        content: list[Table | Text | Align] = [self._build_projects_table()]

        if self.projects and self.current_project_index < len(self.projects):
            current_project = self.projects[self.current_project_index]
            content.append(Text(""))  # Consistent single-line spacing

            # Add project stats (total entries, tokens, and cost if available)
            stats_text = Text()
            stats_text.append("💬 ")
            stats_text.append(f"{current_project.total_entries:,} entries", style=COLOR_WARM_BROWN)

            # Add token information if available
            total_tokens = current_project.total_input_tokens + current_project.total_output_tokens
            if total_tokens > 0:
                stats_text.append("   ·   ", style=SEPARATOR_STYLE)  # Softer separator
                # Use the new format helper but without breakdown for project
                # level
                token_display, _ = self._format_tokens(total_tokens, show_breakdown=False)
                stats_text.append("🔶 ")
                stats_text.append(token_display, style=COLOR_BRIGHT_ORANGE)

            # Add cost if available - make it more prominent
            if current_project.calculated_cost_usd > 0:
                stats_text.append("   ·   ", style=SEPARATOR_STYLE)  # Softer separator
                stats_text.append("💰 ")
                stats_text.append(
                    self._format_cost(current_project.calculated_cost_usd),
                    style=f"{STYLE_BOLD} {COLOR_ORANGE}",
                )

            content.append(Align.center(stats_text))
            content.append(Text(""))  # Consistent single-line spacing

            content.append(
                self._build_conversations_table()
                if current_project.conversations
                else Text("No conversations in this project", style=WARNING_STYLE)
            )

        return Panel(
            Group(*content),
            title=(f"[{TABLE_HEADER_PROJECTS}]🌍 Navigation[/{TABLE_HEADER_PROJECTS}]"),
            box=box.DOUBLE,  # Use DOUBLE for consistency with header
            border_style=PROJECTS_BORDER_STYLE,
            padding=(0, 1),
        )

    def _build_projects_table(self) -> Table:
        """Build the projects table with viewport scrolling."""
        # Calculate viewport bounds for scrolling
        start_idx, end_idx = self._calculate_viewport_bounds_generic(
            self.current_project_index, len(self.projects), self._projects_viewport_size
        )
        viewport_projects = self.projects[start_idx:end_idx]

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style=TABLE_HEADER_PROJECTS,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("📁 Project", style=COLOR_ACCENT_BLUE, no_wrap=True, width=None)
        table.add_column("💬 Count", justify="right", style=COLOR_WARM_BROWN, width=8)
        table.add_column("🕰️  Latest", style=COLOR_ORANGE, no_wrap=True, width=12)

        # Add projects in current viewport
        for i, project in enumerate(viewport_projects):
            actual_index = start_idx + i
            is_selected = actual_index == self.current_project_index
            style = PROJECT_HIGHLIGHT_STYLE if is_selected else ""
            latest_text = self._format_relative_date(project.latest_conversation)

            # Format count with proper alignment
            count_text = Text(str(project.conversation_count), justify="right")

            # Add selection indicator
            if is_selected:
                project_display = Text()
                project_display.append("▸ ", style=f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}")
                project_display.append(project.project_name)
            else:
                project_display = Text("  " + project.project_name)

            table.add_row(project_display, count_text, latest_text, style=style)

        # Add scroll indicator if needed
        scroll_info = self._get_scroll_indicator_generic(
            start_idx, end_idx, len(self.projects), self._projects_viewport_size
        )

        if scroll_info:
            indicator_text = f"[{scroll_info}]"
        else:
            # Show actual viewport range when not scrolling
            indicator_text = f"[• {start_idx + 1}-{end_idx}/{len(self.projects)}]"

        # Add indicator as a table row - right align in first column for
        # near-center appearance
        right_aligned_indicator = Text(indicator_text, justify="right")
        table.add_row(
            right_aligned_indicator, "", "", style=COLOR_LIGHT_BROWN
        )  # Remove dim for better visibility

        return table

    def _calculate_viewport_bounds_generic(
        self, current_index: int, total_items: int, viewport_size: int
    ) -> tuple[int, int]:
        """Calculate start and end indices for any viewport."""
        if total_items <= viewport_size:
            return 0, total_items

        # Keep current selection visible with context
        if current_index < viewport_size // 2:
            start_idx = 0
        elif current_index >= total_items - (viewport_size // 2):
            start_idx = total_items - viewport_size
        else:
            start_idx = current_index - (viewport_size // 2)

        end_idx = min(total_items, start_idx + viewport_size)
        return max(0, start_idx), end_idx

    def _get_scroll_indicator_generic(
        self, start_idx: int, end_idx: int, total: int, viewport_size: int
    ) -> str:
        """Get scroll position indicator for any viewport."""
        if total > viewport_size:
            if start_idx > 0 and end_idx < total:
                return f"↕ {start_idx + 1}-{end_idx}/{total}"
            elif start_idx > 0:
                return f"↑ {start_idx + 1}-{end_idx}/{total}"
            elif end_idx < total:
                return f"↓ {start_idx + 1}-{end_idx}/{total}"
            # This case should never happen if total > viewport_size,
            # but just in case:
            return f"• {start_idx + 1}-{end_idx}/{total}"
        return ""

    def _build_conversations_table(self) -> Table:
        """Build the conversations table with viewport scrolling."""
        project = self.projects[self.current_project_index]
        conversations = project.conversations

        # Calculate viewport bounds for scrolling
        start_idx, end_idx = self._calculate_viewport_bounds_generic(
            self.current_conversation_index, len(conversations), self._conversations_viewport_size
        )
        viewport_conversations = conversations[start_idx:end_idx]

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style=TABLE_HEADER_CONVERSATIONS,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("💬 Conversation", style=COLOR_ACCENT_BLUE, no_wrap=True, width=None)
        table.add_column("🔶 Tokens", justify="right", style=COLOR_WARM_BROWN, width=9)
        table.add_column("🕰️  Updated", style=COLOR_ORANGE, no_wrap=True, width=12)

        # Add conversations in current viewport
        for i, conv in enumerate(viewport_conversations):
            actual_index = start_idx + i
            is_selected = actual_index == self.current_conversation_index
            style = CONVERSATION_HIGHLIGHT_STYLE if is_selected else ""

            # Format timestamp consistently
            if conv.last_timestamp:
                timestamp = conv.last_timestamp.astimezone()
                # Show time only for today, date+time for recent, date only
                # for old
                now = datetime.now().astimezone()
                days_diff = (now.date() - timestamp.date()).days

                if days_diff == 0:
                    updated_text = f"Today {timestamp.strftime('%H:%M')}"
                elif days_diff < 7:
                    updated_text = timestamp.strftime("%m/%d %H:%M")
                else:
                    updated_text = timestamp.strftime("%m/%d")
            else:
                updated_text = "Unknown"

            # Use summary if available, otherwise session ID
            conv_text = self._get_conversation_display_text(conv)

            # Add selection indicator
            if is_selected:
                display_text = Text()
                display_text.append("▸ ", style=f"{STYLE_BOLD} {COLOR_BRIGHT_ORANGE}")
                display_text.append(conv_text)
            else:
                display_text = Text("  " + conv_text, style=f"{COLOR_ACCENT_BLUE}")

            # Format tokens compactly for table display
            total_tokens = conv.total_input_tokens + conv.total_output_tokens
            if total_tokens >= 1_000_000:
                token_str = f"{total_tokens / 1_000_000:.1f}M"
            elif total_tokens >= 1_000:
                token_str = f"{total_tokens / 1_000:.0f}K"
            else:
                token_str = str(total_tokens)

            # Apply color to token count if not highlighted
            if style != CONVERSATION_HIGHLIGHT_STYLE:
                token_display = Text(token_str, style=self._get_token_color(total_tokens))
                table.add_row(
                    display_text,
                    token_display,
                    updated_text,
                    style=style,
                )
            else:
                table.add_row(
                    display_text,
                    token_str,
                    updated_text,
                    style=style,
                )

        # Add scroll indicator if needed
        scroll_info = self._get_scroll_indicator_generic(
            start_idx, end_idx, len(conversations), self._conversations_viewport_size
        )

        if scroll_info:
            indicator_text = f"[{scroll_info}]"
        else:
            # Show actual viewport range when not scrolling
            indicator_text = f"[• {start_idx + 1}-{end_idx}/{len(conversations)}]"

        # Add indicator as a table row - right align in first column for
        # near-center appearance
        right_aligned_indicator = Text(indicator_text, justify="right")
        table.add_row(
            right_aligned_indicator, "", "", style=COLOR_ORANGE
        )  # Remove dim for better visibility

        return table

    def _build_preview_panel(self) -> Panel:
        """Build the conversation preview panel."""
        if not self.projects or self.current_project_index >= len(self.projects):
            return Panel("No project selected", title="Preview", box=box.ROUNDED)

        project = self.projects[self.current_project_index]
        if not project.conversations or self.current_conversation_index >= len(
            project.conversations
        ):
            return Panel("No conversation selected", title="Preview", box=box.ROUNDED)

        conv = project.conversations[self.current_conversation_index]
        preview_content = self._get_conversation_preview(conv)

        # Create a beautiful title with summary or session ID
        title_text = self._get_conversation_display_text(conv, max_length=50)

        return Panel(
            preview_content,
            title=(
                f"[{STYLE_BOLD} {COLOR_WARM_BROWN}]💬 {title_text}"
                f"[/{STYLE_BOLD} {COLOR_WARM_BROWN}]"
            ),
            box=box.DOUBLE,  # Use DOUBLE for consistency
            border_style=PREVIEW_BORDER_STYLE,
        )

    def _get_conversation_preview(self, conv: ConversationMetadata) -> Group:
        """Get basic conversation preview content."""
        from typing import Any

        content: list[Any] = []

        # Beautiful info table with clean design
        info_table = Table(
            box=box.SIMPLE,  # Simple lines for subtle structure
            show_header=False,
            padding=(0, 1),
            expand=False,  # Don't expand to prevent center alignment
        )
        info_table.add_column("Field", style=METADATA_LABEL_STYLE, width=13)
        info_table.add_column("Value", style=METADATA_VALUE_STYLE)

        # Add rows with icons for visual hierarchy
        rows: list[tuple[str, Any]] = [
            ("📁 Project:", conv.project_name),
        ]

        # Add turn counts
        if conv.assistant_turns > 0 or conv.user_turns > 0:
            turns_text = f"👤 {conv.user_turns} ↔️  {conv.assistant_turns} 🤖"
            # Add task count if available
            if conv.task_invocations:
                turns_text += f" [🌱 {len(conv.task_invocations)}]"
            # Add tool result count if available
            if conv.tool_result_turns > 0:
                turns_text += f" [🔧 {conv.tool_result_turns}]"
            rows.append(("💬 Turns:", turns_text))

        # Add token usage information if available
        if conv.total_input_tokens > 0 or conv.total_output_tokens > 0:
            total_tokens = conv.total_input_tokens + conv.total_output_tokens
            total_str, breakdown_str = self._format_tokens(
                total_tokens,
                conv.total_input_tokens,
                conv.total_output_tokens,
                conv.total_cache_creation_input_tokens,
                conv.total_cache_read_input_tokens,
            )

            # Apply token-based coloring using _get_token_color
            token_color = self._get_token_color(total_tokens)

            # Create separate Text objects
            total_text = Text(total_str, style=token_color)

            # Combine them into a new Text
            total_display = Text()
            total_display.append(total_text)

            # Add breakdown_str as unstyled text
            if breakdown_str:
                breakdown_text = Text(" " + breakdown_str)
                total_display.append(breakdown_text)

            rows.append(("🔶 Tokens:", total_display))

        # Add model information if available
        if conv.models_used:
            # Filter out '<synthetic>' from models
            filtered_models = [m for m in conv.models_used if m != "<synthetic>"]
            if filtered_models:
                models_text = ", ".join(sorted(filtered_models))
                if len(models_text) > 40:
                    models_text = models_text[:37] + "..."
                rows.append(("🤖 Models:", models_text))

        # Add duration if available (remove duplicate, handled later)
        if conv.total_duration_seconds > 0:
            duration_mins = conv.total_duration_seconds / 60
            if duration_mins < 1:
                duration_text = f"{conv.total_duration_seconds:.0f} seconds"
            elif duration_mins < 60:
                duration_text = f"{duration_mins:.0f} minutes"
            else:
                hours = duration_mins / 60
                duration_text = f"{hours:.1f} hours"
            rows.append(("⏲️  Duration:", duration_text))

        # Add cost information with smart formatting
        if conv.calculated_cost_usd > 0 or conv.total_cost_usd > 0:
            # Determine if we should show both costs
            show_both = False
            if conv.total_cost_usd > 0 and conv.calculated_cost_usd > 0:
                # Show both if they differ by more than 10%
                diff_percent = (
                    abs(conv.total_cost_usd - conv.calculated_cost_usd) / conv.total_cost_usd
                )
                show_both = diff_percent > 0.1

            # Determine the primary cost for coloring
            primary_cost = (
                conv.calculated_cost_usd if conv.calculated_cost_usd > 0 else conv.total_cost_usd
            )

            if show_both:
                # Show both costs with clear labels
                cost_str = (
                    f"{self._format_cost(conv.calculated_cost_usd)} "
                    f"(original: {self._format_cost(conv.total_cost_usd)})"
                )
            elif conv.calculated_cost_usd > 0:
                # Show only calculated cost
                cost_str = self._format_cost(conv.calculated_cost_usd)
            else:
                # Show only original cost
                cost_str = self._format_cost(conv.total_cost_usd)

            # Apply cost-based coloring using _get_cost_color
            cost_color = self._get_cost_color(primary_cost)
            cost_display = Text(cost_str, style=cost_color)
            rows.append(("💰 Cost:", cost_display))

        # Add timestamp rows if available
        if conv.first_timestamp:
            rows.append(
                ("📍 Created:", conv.first_timestamp.astimezone().strftime("%b %d, %Y at %I:%M %p"))
            )

        if conv.last_timestamp:
            rows.append(
                ("✏️  Updated:", conv.last_timestamp.astimezone().strftime("%b %d, %Y at %I:%M %p"))
            )

            # Calculate duration if both timestamps exist (skip if already
            # added)
            if conv.first_timestamp and conv.total_duration_seconds == 0:
                duration = conv.last_timestamp - conv.first_timestamp
                hours = duration.total_seconds() / 3600

                # Format duration nicely
                if hours < 1:
                    duration_str = f"{int(hours * 60)} minutes"
                elif hours < 24:
                    duration_str = f"{hours:.1f} hours"
                else:
                    days = int(hours / 24)
                    remaining_hours = hours % 24
                    duration_str = f"{days}d {remaining_hours:.0f}h"

                rows.append(("⏱️  Duration:", duration_str))

        # Add all rows to table
        for field, value in rows:
            info_table.add_row(field, value)

        # Wrap metadata in a subtle panel
        metadata_panel = Panel(
            info_table,
            box=box.ROUNDED,
            border_style=COLOR_LIGHT_BROWN,  # Use warm brown for better cohesion
            padding=(0, 1),
        )
        content.append(metadata_panel)

        # Get first 2 messages and first 2 tasks, then sort by timestamp
        user_messages_with_timestamps = self._get_first_user_messages_with_timestamps(
            conv, max_messages=2
        )
        first_tasks = self._get_first_tasks(conv, max_tasks=2)

        # Combine and sort by timestamp
        all_items = user_messages_with_timestamps + first_tasks
        all_items.sort(
            key=lambda x: (
                x["timestamp"] if x["timestamp"] else (conv.first_timestamp or datetime.min)
            )
        )

        for item in all_items:
            if item["type"] == "message":
                # Render message
                msg_text = self._truncate_to_lines(item["content"], max_lines=4)

                msg_panel = Panel(
                    Text(msg_text, style=USER_MESSAGE_STYLE, overflow="ellipsis"),
                    title=f"[{METADATA_LABEL_STYLE}]🗨️  Message[/{METADATA_LABEL_STYLE}]",
                    box=box.HEAVY,
                    border_style=COLOR_ACCENT_BLUE,
                    padding=(0, 1),
                    height=6,  # Fixed height: 4 lines content + 2 for borders
                )
                content.append(msg_panel)

            elif item["type"] == "task":
                # Render task
                task_bubble = self._create_task_bubble_compact(item["content"])
                content.append(task_bubble)

        # Add action hint with refined style
        hint = Text()
        hint.append("→ ", style=f"{STYLE_BOLD} {COLOR_WARM_BROWN}")
        hint.append("Press ", style=SEPARATOR_STYLE)
        hint.append("Enter", style=f"{STYLE_BOLD} {COLOR_WARM_BROWN}")
        hint.append(" to view full conversation", style=SEPARATOR_STYLE)
        content.append(Align.center(hint))

        return Group(*content)

    def _build_footer(self) -> Panel:
        """Build the footer with controls."""
        return self._build_help_panel() if self.show_help else self._build_controls_panel()

    def _build_help_panel(self) -> Panel:
        """Build the help panel with better visual design."""
        help_text = Text()

        # Single line of controls with better spacing
        help_text.append("↑↓", style=KEY_STYLE)
        help_text.append(" Navigate", style=ACTION_TEXT_STYLE)
        help_text.append("   ")

        help_text.append("←→", style=KEY_STYLE)
        help_text.append(" Switch project", style=ACTION_TEXT_STYLE)
        help_text.append("   ")

        help_text.append("⏎", style=SELECT_KEY_STYLE)
        help_text.append(" Select", style=ACTION_TEXT_STYLE)
        help_text.append("   ")

        help_text.append("t", style=KEY_STYLE)
        help_text.append(" Timeframe", style=ACTION_TEXT_STYLE)
        help_text.append("   ")

        help_text.append("r", style=KEY_STYLE)
        help_text.append(" Refresh", style=ACTION_TEXT_STYLE)
        help_text.append("   ")

        help_text.append("q", style=QUIT_KEY_STYLE)
        help_text.append(" Quit", style=ACTION_TEXT_STYLE)

        return Panel(
            Align.center(help_text), box=box.DOUBLE, style=FOOTER_BORDER_STYLE, padding=(0, 1)
        )

    def _build_controls_panel(self) -> Panel:
        """Build the controls panel with elegant design."""
        controls = Text()

        # Show main actions with consistent spacing
        controls.append("?", style=KEY_STYLE)
        controls.append(" Help", style=ACTION_TEXT_STYLE)
        controls.append("   ")

        # Make select action slightly more prominent
        controls.append("⏎", style=SELECT_KEY_STYLE)
        controls.append(" Select", style=f"{STYLE_BOLD} {COLOR_OFF_WHITE}")
        controls.append("   ")

        controls.append("q", style=QUIT_KEY_STYLE)
        controls.append(" Quit", style=ACTION_TEXT_STYLE)

        return Panel(
            Align.center(controls), box=box.DOUBLE, style=FOOTER_BORDER_STYLE, padding=(0, 1)
        )

    def _get_key(self) -> str:
        """Get a single keypress from the user."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)

            # Handle escape sequences (arrow keys)
            if key == "\x1b":
                with contextlib.suppress(Exception):
                    key += sys.stdin.read(2)

            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _move_down(self) -> None:
        """Move selection down."""
        project = self.projects[self.current_project_index]
        if (
            project.conversations
            and self.current_conversation_index < len(project.conversations) - 1
        ):
            self.current_conversation_index += 1
            self.ui_dirty = True

    def _move_up(self) -> None:
        """Move selection up."""
        if self.current_conversation_index > 0:
            self.current_conversation_index -= 1
            self.ui_dirty = True

    def _move_to_next_project(self) -> None:
        """Move to next project."""
        if self.current_project_index < len(self.projects) - 1:
            self.current_project_index += 1
            self.current_conversation_index = 0
            self.ui_dirty = True

    def _move_to_previous_project(self) -> None:
        """Move to previous project."""
        if self.current_project_index > 0:
            self.current_project_index -= 1
            self.current_conversation_index = 0
            self.ui_dirty = True

    def _move_to_top(self) -> None:
        """Move to top of current list."""
        if self.current_conversation_index != 0:
            self.current_conversation_index = 0
            self.ui_dirty = True

    def _move_to_bottom(self) -> None:
        """Move to bottom of current list."""
        project = self.projects[self.current_project_index]
        if project.conversations:
            new_index = len(project.conversations) - 1
            if new_index != self.current_conversation_index:
                self.current_conversation_index = new_index
                self.ui_dirty = True

    def _select_current_conversation(self) -> ConversationMetadata | None:
        """Select the currently highlighted conversation."""
        project = self.projects[self.current_project_index]
        if not project.conversations or self.current_conversation_index >= len(
            project.conversations
        ):
            return None
        return project.conversations[self.current_conversation_index]

    def _get_conversation_display_text(
        self, conv: ConversationMetadata, max_length: int = 30
    ) -> str:
        """Get display text for conversation - summary if available, else
        session ID."""
        if conv.summary:
            # Clean and truncate summary intelligently
            summary = conv.summary.strip()
            # Remove newlines and excessive whitespace
            summary = " ".join(summary.split())

            if len(summary) > max_length:
                # Try to break at word boundary
                truncated = summary[: max_length - 3]
                last_space = truncated.rfind(" ")
                if last_space > max_length * 0.7:  # If we have a reasonable break point
                    truncated = truncated[:last_space]
                return truncated + "..."
            return summary
        else:
            # Fallback to session ID with icon
            return f"ID {conv.session_id[:8]}"

    def _get_first_user_messages(
        self, conv: ConversationMetadata, max_messages: int = 2
    ) -> list[str]:
        """Get the first user message(s) from the conversation."""
        # Check cache first
        cache_key = f"{conv.session_id}_messages"
        if cache_key in self._first_message_cache:
            return self._first_message_cache[cache_key]

        try:
            # Parse to get first user messages (regardless of what's in between)
            user_messages = []

            for entry in stream_conversation_entries(conv):
                if isinstance(entry, ConversationEntry) and entry.type == "user" and entry.message:
                    # Extract text from message
                    if isinstance(entry.message.content, str):
                        text = entry.message.content
                    else:
                        # Extract from content blocks
                        text_parts = []
                        for block in entry.message.content:
                            if hasattr(block, "text"):
                                text_parts.append(block.text)
                        text = " ".join(text_parts)

                    text = text.strip()
                    if text:
                        user_messages.append(text)
                        if len(user_messages) >= max_messages:
                            # Stop after reaching max_messages
                            break

            # Format and truncate messages
            formatted_messages = []
            for msg in user_messages:
                if len(msg) > 300:
                    msg = msg[:297] + "..."
                formatted_messages.append(msg)

            self._first_message_cache[cache_key] = formatted_messages
            return formatted_messages

        except Exception:
            # Don't let preview errors break the UI
            self._first_message_cache[cache_key] = []
            return []

    def _get_first_user_messages_with_timestamps(
        self, conv: ConversationMetadata, max_messages: int = 2
    ) -> list[dict[str, Any]]:
        """Get the first user messages with their timestamps."""
        try:
            messages_with_timestamps = []

            for entry in stream_conversation_entries(conv):
                if isinstance(entry, ConversationEntry) and entry.type == "user" and entry.message:
                    # Extract text from message
                    if isinstance(entry.message.content, str):
                        text = entry.message.content
                    else:
                        # Extract from content blocks
                        text_parts = []
                        for block in entry.message.content:
                            if hasattr(block, "text"):
                                text_parts.append(block.text)
                        text = " ".join(text_parts)

                    text = text.strip()
                    if text:
                        if len(text) > 300:
                            text = text[:297] + "..."

                        messages_with_timestamps.append(
                            {"type": "message", "content": text, "timestamp": entry.timestamp}
                        )

                        if len(messages_with_timestamps) >= max_messages:
                            break

            return messages_with_timestamps

        except Exception:
            return []

    def _get_first_tasks(
        self, conv: ConversationMetadata, max_tasks: int = 2
    ) -> list[dict[str, Any]]:
        """Get the first X tasks with timestamps."""
        tasks_with_timestamps = []

        for task in conv.task_invocations[:max_tasks]:
            tasks_with_timestamps.append(
                {
                    "type": "task",
                    "content": task,
                    "timestamp": (
                        task.timestamp or conv.first_timestamp
                    ),  # Fallback to conversation first_timestamp
                }
            )

        return tasks_with_timestamps

    def _refresh_conversations(self) -> None:
        """Refresh all conversations by re-discovering from disk."""
        # Clear cache
        self._first_message_cache.clear()

        # Save current position
        current_project_name = None
        current_conv_id = None
        if self.projects and self.current_project_index < len(self.projects):
            current_project = self.projects[self.current_project_index]
            current_project_name = current_project.project_name
            if current_project.conversations and self.current_conversation_index < len(
                current_project.conversations
            ):
                current_conv_id = current_project.conversations[
                    self.current_conversation_index
                ].session_id

        # Re-discover all projects and conversations
        self.console.print("🔄 Refreshing conversations...", style=INFO_STYLE)
        self.all_projects = discover_projects()

        # Apply timeframe filter
        self._apply_timeframe_filter()

        # Try to restore position
        self.current_project_index = 0
        self.current_conversation_index = 0

        if current_project_name:
            # Find the project
            for i, project in enumerate(self.projects):
                if project.project_name == current_project_name:
                    self.current_project_index = i
                    # Find the conversation
                    if current_conv_id and project.conversations:
                        for j, conv in enumerate(project.conversations):
                            if conv.session_id == current_conv_id:
                                self.current_conversation_index = j
                                break
                    break

        # Mark UI as dirty to force redraw
        self.ui_dirty = True

    def _cycle_timeframe(self) -> None:
        """Cycle through timeframe options."""
        # Move to next timeframe option
        self.current_timeframe_index = (self.current_timeframe_index + 1) % len(TIMEFRAME_OPTIONS)
        self.timeframe_days = TIMEFRAME_OPTIONS[self.current_timeframe_index]

        # Apply the new filter
        self._apply_timeframe_filter()

        # Reset conversation index to avoid index out of bounds
        self.current_project_index = 0
        self.current_conversation_index = 0

        # Mark UI as dirty
        self.ui_dirty = True

    def _apply_timeframe_filter(self) -> None:
        """Apply timeframe filter to all projects."""
        self.projects = filter_projects_by_timeframe(self.all_projects, self.timeframe_days)

    def _format_tokens(
        self,
        total: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation: int = 0,
        cache_read: int = 0,
        show_breakdown: bool = True,
    ) -> tuple[str, str]:
        # Format total with K/M abbreviations
        if total >= 1_000_000:
            total_str = f"{total / 1_000_000:.1f}M"
        elif total >= 1_000:
            total_str = f"{total / 1_000:.1f}K"
        else:
            total_str = str(total)

        total_string = f"{total_str} tokens"

        if not show_breakdown or (input_tokens == 0 and output_tokens == 0):
            return total_string, ""

        # Build breakdown
        parts = ["("]

        # Add input/output
        if input_tokens >= 1_000:
            parts.append(f"↓ {input_tokens / 1_000:.1f}K")
        else:
            parts.append(f"↓ {input_tokens}")

        if output_tokens >= 1_000:
            parts.append(f" ↑ {output_tokens / 1_000:.1f}K")
        else:
            parts.append(f" ↑ {output_tokens}")

        # Add cache info if present with input/output split
        if cache_creation > 0 or cache_read > 0:
            parts.append(" 💾 ")

            # Cache creation (input)
            if cache_creation >= 1_000:
                parts.append(f"↓ {cache_creation / 1_000:.1f}K")
            else:
                parts.append(f"↓ {cache_creation}")

            # Cache read (output)
            if cache_read >= 1_000:
                parts.append(f" ↑ {cache_read / 1_000:.1f}K")
            else:
                parts.append(f" ↑ {cache_read}")

            # Calculate cache efficiency
            total_cache = cache_creation + cache_read
            cache_efficiency = (
                int((total_cache / (input_tokens + total_cache)) * 100)
                if input_tokens + total_cache > 0
                else 0
            )
            parts.append(f"・{cache_efficiency}%")

        parts.append(")")
        breakdown_string = "".join(parts)

        return total_string, breakdown_string

    def _create_task_bubble_compact(self, task: TaskInvocation) -> Panel:
        """Create a compact task panel with fixed height."""
        content = Text()

        # Truncate description to fit in 2 lines max
        desc_lines = task.description.split("\n")
        if len(desc_lines) > 2:
            truncated_desc = "\n".join(desc_lines[:2]) + "..."
        else:
            truncated_desc = task.description

        # Further truncate if individual lines are too long
        truncated_desc = self._truncate_to_lines(truncated_desc, max_lines=2)
        content.append(truncated_desc, style="bold")

        # Add compact metadata on one line with emojis
        metadata_parts = []
        if task.total_duration_ms:
            duration = f"{task.total_duration_ms / 1000:.1f}s"
            metadata_parts.append(f"⏱️  {duration}")

        # Show full token breakdown for tasks
        if task.total_tokens:
            total_str, breakdown_str = self._format_tokens(
                task.total_tokens,
                task.result_input_tokens or 0,
                task.result_output_tokens or 0,
                task.result_cache_creation_tokens or 0,
                task.result_cache_read_tokens or 0,
                show_breakdown=True,  # Show full breakdown for tasks
            )
            # Combine total and breakdown for full display
            if breakdown_str:
                metadata_parts.append(f"🔶 {total_str} {breakdown_str}")
            else:
                metadata_parts.append(f"🔶 {total_str}")

        if metadata_parts:
            content.append("\n")
            content.append(" • ".join(metadata_parts), style="dim")

        return Panel(
            content,
            title="[bold yellow]🌱 Task[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
            height=4,  # Fixed height: 2 lines description + 1 metadata + 1 border
        )

    def _truncate_to_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to max_lines, breaking only on newlines. Let Rich handle wrapping."""
        lines = text.split("\n")

        if len(lines) <= max_lines:
            return text

        # Take first max_lines and add ellipsis to the last one
        truncated_lines = lines[:max_lines]
        if truncated_lines:
            truncated_lines[-1] += "..."

        return "\n".join(truncated_lines)


def pick_conversation(console: Console | None = None) -> ConversationMetadata | None:
    """Launch the interactive conversation picker."""
    return ConversationPicker(console).run()


if __name__ == "__main__":
    # Test the picker
    selected = pick_conversation()
    if selected:
        print(f"Selected: {selected.session_id} from {selected.project_name}")
    else:
        print("No conversation selected")
