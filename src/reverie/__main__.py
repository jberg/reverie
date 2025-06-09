"""
reverie - Main entry point.

Export and view Claude Code conversations.
"""

import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from .exporter import ExportOptions, MarkdownExporter
from .parser import parse_conversation
from .picker import pick_conversation
from .pricing import CostMode
from .renderer import ConversationRenderer


def main() -> None:
    """Main entry point for reverie."""
    console = Console()

    try:
        console.print("🔶 reverie", style="#F6C244")
        console.print()

        # Launch the picker
        selected = pick_conversation(console)

        if selected:
            console.print()
            console.print("✅ Conversation Selected!", style="bold green")
            console.print(f"   📝 Session: [yellow]{selected.session_id[:12]}...[/yellow]")
            console.print(f"   📁 Project: [cyan]{selected.project_name}[/cyan]")
            console.print(f"   📊 Entries: [green]{selected.entry_count}[/green]")

            # Show cost if available
            if selected.calculated_cost_usd > 0 or selected.total_cost_usd > 0:
                cost = (
                    selected.calculated_cost_usd
                    if selected.calculated_cost_usd > 0
                    else selected.total_cost_usd
                )
                console.print(f"   💰 Cost: [yellow]${cost:.4f}[/yellow]")

            console.print()

            # Ask what to do with the conversation
            choices = [
                "[bold green]1[/bold green]) 📄 Export to Markdown",
                "[bold blue]2[/bold blue]) 🖥️  View in Console",
                "[bold red]3[/bold red]) ❌ Cancel",
            ]

            console.print("What would you like to do?")
            for choice in choices:
                console.print(f"   {choice}")

            console.print()
            action = Prompt.ask("Choose an option", choices=["1", "2", "3"], default="1")

            if action == "1":
                # Export to markdown
                console.print()
                console.print("📄 Exporting to Markdown...", style="cyan")

                # Parse the full conversation
                conversation = parse_conversation(selected)

                # Generate output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_project_name = selected.project_name.replace("/", "_").replace(" ", "_")
                filename = f"reverie_claude_code_export_{safe_project_name}_{timestamp}.md"

                # Default to Desktop if it exists, otherwise current directory
                desktop = Path.home() / "Desktop"
                output_dir = desktop if desktop.exists() else Path.cwd()
                output_path = output_dir / filename

                # Create exporter with beautiful options
                options = ExportOptions(
                    include_tool_results=True,
                    include_thinking=True,
                    include_cost=True,
                    include_timestamps=True,
                    include_statistics=True,
                    max_tool_result_lines=100,
                    collapsible_tool_results=True,
                    syntax_highlighting=True,
                    cost_mode=CostMode.AUTO,
                )

                exporter = MarkdownExporter(options)
                exporter.export(conversation, selected, output_path)

                console.print()
                console.print("✅ Successfully exported!", style="bold green")
                console.print(f"   📄 File: [blue]{output_path}[/blue]")

                # Show non-cached tokens instead of file size
                non_cached_input = selected.total_input_tokens
                non_cached_output = selected.total_output_tokens
                total_non_cached = non_cached_input + non_cached_output

                if total_non_cached >= 1_000_000:
                    token_str = f"{total_non_cached / 1_000_000:.1f}M"
                elif total_non_cached >= 1_000:
                    token_str = f"{total_non_cached / 1_000:.0f}K"
                else:
                    token_str = str(total_non_cached)

                tokens_detail = f"(↓{non_cached_input:,} ↑{non_cached_output:,})"
                console.print(f"   🔶 Tokens: [yellow]{token_str} {tokens_detail}[/yellow]")
            elif action == "2":
                # View in console
                console.print()
                console.print("Loading conversation...", style="dim")

                # Parse the full conversation
                conversation = parse_conversation(selected)

                # Clear screen and render
                console.clear()
                renderer = ConversationRenderer(console)
                renderer.render(conversation)

        else:
            console.print()
            console.print("No conversation selected", style="yellow")

    except KeyboardInterrupt:
        console.print("\nCancelled", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="bold red")
        sys.exit(1)


if __name__ == "__main__":
    main()
