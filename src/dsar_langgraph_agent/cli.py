from __future__ import annotations

import argparse
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from dsar_langgraph_agent.triage_graph import build_triage_graph
from dsar_langgraph_agent.triage_schemas import (
    HumanReviewDecision,
    LLMFallbackWarning,
    ReasoningEntry,
    TriageOutput,
)

console = Console()

# ── Source-type display config ────────────────────────────────────────────────
_SOURCE_META = {
    "llm":           ("🤖", "AI Thought",           "cyan"),
    "deterministic": ("⚙️ ", "Deterministic Rule",   "green"),
    "llm_fallback":  ("⚠️ ", "LLM Fallback",         "yellow"),
    "human":         ("👤", "Human Decision",        "magenta"),
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _print_health_alerts(warnings: List[LLMFallbackWarning]) -> None:
    """Render llm_warnings as bold alert panel at the top of the report."""
    if not warnings:
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
    table.add_column("Agent",  style="bold", no_wrap=True)
    table.add_column("Cause",  style="bold yellow", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    table.add_column("When",   style="dim", no_wrap=True)

    for w in warnings:
        table.add_row(w.agent, w.cause.upper(), w.detail, w.timestamp)

    console.print(
        Panel(
            table,
            title="[bold red]⚠  System Health Alerts[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
    )


def _print_reasoning_history(history: List[ReasoningEntry]) -> None:
    """Print the reasoning history as a linear audit trail."""
    if not history:
        console.print("[dim]  (no reasoning history recorded)[/dim]")
        return

    for idx, entry in enumerate(history, 1):
        icon, label, colour = _SOURCE_META.get(
            entry.source, ("❓", entry.source, "white")
        )
        header = (
            f"[{colour}]{icon}  [{idx}/{len(history)}] "
            f"[bold]{label}[/bold] — [italic]{entry.agent}[/italic][/{colour}]"
        )
        body = (
            f"[bold]Decision:[/bold] {entry.decision_made}\n"
            f"[bold]Rationale:[/bold] {entry.rationale}"
        )
        console.print(
            Panel(
                body,
                title=header,
                border_style=colour,
                padding=(0, 1),
            )
        )


def _print_triage_report(state_values: Dict[str, Any], iteration: int) -> None:
    """Full structured report shown each time the graph pauses at human_checkpoint."""
    classification = state_values.get("classification")
    scope          = state_values.get("scope")
    risk           = state_values.get("risk")
    warnings: List[LLMFallbackWarning] = list(state_values.get("llm_warnings") or [])
    history: List[ReasoningEntry]      = list(state_values.get("reasoning_history") or [])

    pass_label = f"Review pass #{iteration}"
    console.print()
    console.print(Rule(f"[bold blue]  DSAR Triage Report — {pass_label}  [/bold blue]", style="blue"))

    # 1 · Health alerts (only if any)
    _print_health_alerts(warnings)

    # 2 · Core decisions table
    decisions = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    decisions.add_column("Field",  style="bold dim", no_wrap=True)
    decisions.add_column("Value")

    if classification:
        decisions.add_row(
            "Request Type",
            f"[bold]{classification.request_type.value.upper()}[/bold]"
            f"  (confidence: {classification.confidence:.0%})",
        )
        decisions.add_row("Rationale", classification.rationale)

    if scope:
        systems_str = ", ".join(scope.systems) if scope.systems else "[dim]none[/dim]"
        decisions.add_row("Scoped Systems", systems_str)
        if scope.rationale:
            decisions.add_row("Scope Rationale", scope.rationale)

    if risk:
        if risk.flags:
            flags_str = "  ".join(f"[yellow]⚑ {f.value}[/yellow]" for f in risk.flags)
        else:
            flags_str = "[green]✓ No risk flags[/green]"
        decisions.add_row("Risk Flags", flags_str)
        for note in (risk.notes or []):
            decisions.add_row("", f"[dim]• {note}[/dim]")

    console.print(
        Panel(
            decisions,
            title="[bold]📋  Proposed Decisions[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
    )

    # 3 · Reasoning history
    console.print(
        Panel(
            "",
            title="[bold]🔍  Reasoning History[/bold]",
            border_style="dim",
            padding=(0, 0),
        )
    )
    _print_reasoning_history(history)
    console.print()


def _print_final_summary(output: TriageOutput) -> None:
    """Compact final summary printed after the reviewer approves."""
    c = output.classification
    s = output.scope
    r = output.risk

    # Decisions table
    summary = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    summary.add_column("Field", style="bold dim", no_wrap=True)
    summary.add_column("Value")

    summary.add_row(
        "Request Type",
        f"[bold green]{c.request_type.value.upper()}[/bold green]"
        f"  (confidence: {c.confidence:.0%})",
    )
    summary.add_row("Scoped Systems", ", ".join(s.systems) or "none")

    if r.flags:
        summary.add_row(
            "Risk Flags",
            "  ".join(f"[yellow]⚑ {f.value}[/yellow]" for f in r.flags),
        )
    else:
        summary.add_row("Risk Flags", "[green]✓ None[/green]")

    if output.human_review:
        hr = output.human_review
        summary.add_row(
            "Reviewer",
            f"{hr.reviewer}  ({'approved' if hr.approved else 'revised'})",
        )
        if hr.notes:
            summary.add_row("Reviewer Notes", hr.notes)

    history_len = len(output.reasoning_history)
    warn_len    = len(output.llm_warnings)

    summary.add_row("Audit Trail", f"{history_len} reasoning step(s) recorded")
    if warn_len:
        summary.add_row(
            "LLM Warnings",
            f"[yellow]{warn_len} fallback event(s) — review llm_warnings for details[/yellow]",
        )

    console.print()
    console.print(Rule("[bold green]  ✅  Final DSAR Triage Summary  [/bold green]", style="green"))
    console.print(
        Panel(summary, title="[bold]Final Decisions[/bold]", border_style="green", padding=(0, 1))
    )

    # Audit trail mini-timeline
    if output.reasoning_history:
        console.print("[bold dim]Audit trail:[/bold dim]")
        for idx, entry in enumerate(output.reasoning_history, 1):
            icon, _, colour = _SOURCE_META.get(entry.source, ("❓", "", "white"))
            console.print(
                f"  [{colour}]{icon}[/{colour}] "
                f"[dim]{idx:02d}.[/dim] "
                f"[bold]{entry.agent}[/bold] → {entry.decision_made}"
            )
    console.print()


def _prompt_human_decision(reviewer: str) -> HumanReviewDecision:
    """
    Interactive prompt: [a]pprove or [r]evise with feedback.
    Loops until the user gives a valid answer.
    """
    while True:
        console.print(
            "\n[bold]What would you like to do?[/bold]  "
            "[[bold green]a[/bold green]] Approve  "
            "[[bold yellow]r[/bold yellow]] Revise with feedback  "
            "[[bold red]q[/bold red]] Quit"
        )
        choice = input("  → ").strip().lower()

        if choice in ("q", "quit", "exit"):
            raise SystemExit(0)

        if choice in ("a", "approve"):
            notes = input("  Notes (optional, press Enter to skip): ").strip()
            return HumanReviewDecision(
                approved=True,
                reviewer=reviewer,
                notes=notes,
            )

        if choice in ("r", "revise"):
            console.print(
                "  [yellow]Describe what needs to be corrected.[/yellow]\n"
                "  [dim]Tip: mention the correct request type "
                "(e.g. 'This is portability, not access.')[/dim]"
            )
            feedback = input("  Feedback: ").strip()
            if not feedback:
                console.print("  [red]Feedback cannot be empty. Please try again.[/red]")
                continue
            return HumanReviewDecision(
                approved=False,
                reviewer=reviewer,
                human_feedback=feedback,
            )

        console.print("  [red]Unknown choice — please press 'a', 'r', or 'q'.[/red]")


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DSAR triage LangGraph workflow (with human checkpoint)."
    )
    parser.add_argument("--text",        required=True,  help="Incoming DSAR request text.")
    parser.add_argument("--thread-id",   default="triage_cli", help="LangGraph thread id.")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve (no prompt).")
    parser.add_argument("--use-llm",     action="store_true",  help="Use Ollama LLM agents.")
    parser.add_argument("--model",       default="llama3.1",            help="Ollama model name.")
    parser.add_argument("--base-url",    default="http://localhost:11434/v1", help="Ollama base URL.")
    parser.add_argument("--api-key",     default="ollama",   help="OpenAI SDK api-key field.")
    parser.add_argument("--reviewer",    default="human",    help="Reviewer name shown in audit.")
    args = parser.parse_args()

    llm_config = {"model": args.model, "base_url": args.base_url, "api_key": args.api_key}
    app    = build_triage_graph(use_llm=args.use_llm, llm_config=llm_config if args.use_llm else None)
    config = {"configurable": {"thread_id": args.thread_id}}

    from langgraph.types import Command

    # ── First invocation ──────────────────────────────────────────────────────
    try:
        result = app.invoke({"request_text": args.text}, config=config)
        if "output" in result:
            # Graph completed without pausing (e.g. auto-run mode without checkpoint)
            _print_final_summary(result["output"])
            return 0
    except Exception as exc:
        if "interrupt" not in exc.__class__.__name__.lower():
            raise

    # ── Revision loop ─────────────────────────────────────────────────────────
    iteration = 0
    while True:
        iteration += 1

        # Read current state from the checkpointed graph
        graph_state  = app.get_state(config)  # type: ignore[attr-defined]
        state_values: Dict[str, Any] = graph_state.values

        # Check whether there is actually an interrupt pending
        interrupts = getattr(graph_state, "interrupts", None)
        if not interrupts:
            # Graph ran to completion (possible if a prior resume triggered finalize)
            if "output" in state_values:
                _print_final_summary(state_values["output"])
            return 0

        # Print the structured triage report
        _print_triage_report(state_values, iteration)

        # Collect decision
        if args.auto_approve:
            decision = HumanReviewDecision(
                approved=True, reviewer="auto", notes="auto-approved by CLI"
            )
            console.print("[dim]Auto-approving…[/dim]")
        else:
            decision = _prompt_human_decision(args.reviewer)

        # Resume the graph with the human decision
        resumed = app.invoke(Command(resume=decision.model_dump()), config=config)

        # If the graph produced a final output we're done; otherwise loop again
        if "output" in resumed:
            _print_final_summary(resumed["output"])
            return 0

        # Output not yet present → the graph cycled back (revise path) and
        # paused at the checkpoint again; loop back to show the new report


if __name__ == "__main__":
    raise SystemExit(main())
