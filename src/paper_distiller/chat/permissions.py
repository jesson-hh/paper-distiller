"""Permission modes for the agent loop — borrowed from Claude Code's model.

Five modes affecting how tools are gated before execution:

  default      → show plan-mode preview for costly tools (>= threshold)
  auto         → skip plan-mode preview, run everything
  bypass       → like auto, but ALSO skip even rare "destructive" prompts
                 (currently equivalent to auto in our codebase; reserved
                 for v1.12 when we add file-edit / vault-write gates)
  plan         → ALWAYS show plan-mode preview, no auto-proceed timeout
                 (user must explicitly Enter or 'q'); good for first-time
                 users who want to see every tool call before it fires
  safe         → like plan, but threshold for triggering is ¥0 so even
                 cheap tools show a preview. Maximum caution.

The mode is held on AgentLoop.permission_mode and surfaced in the status
line. Slash command `/mode <name>` switches at runtime.
"""

from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    DEFAULT = "default"
    AUTO = "auto"
    BYPASS = "bypass"
    PLAN = "plan"
    SAFE = "safe"


# Friendly display labels for the status line
LABELS = {
    PermissionMode.DEFAULT: "default",
    PermissionMode.AUTO: "auto",
    PermissionMode.BYPASS: "bypass",
    PermissionMode.PLAN: "plan",
    PermissionMode.SAFE: "safe",
}

# Color hint for the status line — render each mode with appropriate weight
STATUS_COLORS = {
    PermissionMode.DEFAULT: "dim",
    PermissionMode.AUTO: "yellow",
    PermissionMode.BYPASS: "bold red",      # signal "dangerous"
    PermissionMode.PLAN: "cyan",
    PermissionMode.SAFE: "bold green",
}


def should_show_plan_for_mode(
    mode: PermissionMode,
    tool_name: str,
    arguments: dict,
    threshold_cny: float,
) -> bool:
    """Decide whether to show plan-mode preview for a tool call.

    Args:
        mode: current PermissionMode
        tool_name / arguments: the tool call about to execute
        threshold_cny: cost threshold (PD_PLAN_THRESHOLD_CNY env)
    """
    from .cost_estimator import estimate_tool_cost_cny

    if mode == PermissionMode.AUTO or mode == PermissionMode.BYPASS:
        return False
    if mode == PermissionMode.PLAN:
        return True  # always show, no threshold
    if mode == PermissionMode.SAFE:
        return True  # always show + no auto-proceed
    # DEFAULT: threshold-based
    return estimate_tool_cost_cny(tool_name, arguments) >= threshold_cny


def confirm_timeout_seconds(mode: PermissionMode, default: int = 5) -> int:
    """How long the plan-mode card waits before auto-proceeding.

    `plan` and `safe` modes wait forever (require explicit input).
    `default` mode uses the env's countdown (5s by default).
    """
    if mode in (PermissionMode.PLAN, PermissionMode.SAFE):
        return 0  # 0 = no auto-proceed
    return default


def parse_mode(s: str) -> PermissionMode | None:
    """Parse a user string into PermissionMode. Returns None if invalid."""
    s = (s or "").strip().lower()
    for m in PermissionMode:
        if m.value == s:
            return m
    return None
