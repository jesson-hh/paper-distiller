"""Tests for chat.permissions — PermissionMode enum + plan-mode gating."""

from __future__ import annotations


def test_parse_mode_valid():
    from paper_distiller.chat.permissions import parse_mode, PermissionMode

    assert parse_mode("default") == PermissionMode.DEFAULT
    assert parse_mode("auto") == PermissionMode.AUTO
    assert parse_mode("bypass") == PermissionMode.BYPASS
    assert parse_mode("plan") == PermissionMode.PLAN
    assert parse_mode("safe") == PermissionMode.SAFE


def test_parse_mode_case_insensitive():
    from paper_distiller.chat.permissions import parse_mode, PermissionMode

    assert parse_mode("AUTO") == PermissionMode.AUTO
    assert parse_mode("  Default  ") == PermissionMode.DEFAULT


def test_parse_mode_invalid_returns_none():
    from paper_distiller.chat.permissions import parse_mode

    assert parse_mode("") is None
    assert parse_mode("nope") is None
    assert parse_mode(None) is None


def test_auto_mode_skips_plan():
    from paper_distiller.chat.permissions import (
        PermissionMode, should_show_plan_for_mode,
    )
    # Even for a costly tool, auto mode returns False
    assert not should_show_plan_for_mode(
        PermissionMode.AUTO, "research", {"max_cost_cny": 50}, threshold_cny=10,
    )


def test_bypass_mode_skips_plan():
    from paper_distiller.chat.permissions import (
        PermissionMode, should_show_plan_for_mode,
    )
    assert not should_show_plan_for_mode(
        PermissionMode.BYPASS, "research", {"max_cost_cny": 50}, threshold_cny=10,
    )


def test_plan_mode_always_shows():
    from paper_distiller.chat.permissions import (
        PermissionMode, should_show_plan_for_mode,
    )
    # Even cheap tools show plan
    assert should_show_plan_for_mode(
        PermissionMode.PLAN, "show", {"slug": "x"}, threshold_cny=10,
    )


def test_safe_mode_always_shows():
    from paper_distiller.chat.permissions import (
        PermissionMode, should_show_plan_for_mode,
    )
    assert should_show_plan_for_mode(
        PermissionMode.SAFE, "show", {"slug": "x"}, threshold_cny=10,
    )


def test_default_mode_respects_threshold():
    from paper_distiller.chat.permissions import (
        PermissionMode, should_show_plan_for_mode,
    )
    # Below threshold → no plan
    assert not should_show_plan_for_mode(
        PermissionMode.DEFAULT, "search", {"topic": "x"}, threshold_cny=10,
    )
    # Above threshold (research defaults to ¥30) → plan
    assert should_show_plan_for_mode(
        PermissionMode.DEFAULT, "research", {"max_cost_cny": 30}, threshold_cny=10,
    )


def test_confirm_timeout_for_plan_mode_is_zero():
    from paper_distiller.chat.permissions import (
        PermissionMode, confirm_timeout_seconds,
    )
    # 0 means "wait forever" — must explicitly confirm
    assert confirm_timeout_seconds(PermissionMode.PLAN) == 0
    assert confirm_timeout_seconds(PermissionMode.SAFE) == 0


def test_confirm_timeout_for_default_is_normal():
    from paper_distiller.chat.permissions import (
        PermissionMode, confirm_timeout_seconds,
    )
    # DEFAULT uses the provided default (5s by env)
    assert confirm_timeout_seconds(PermissionMode.DEFAULT, default=5) == 5


def test_mode_labels_and_colors_present():
    from paper_distiller.chat.permissions import LABELS, STATUS_COLORS, PermissionMode
    for m in PermissionMode:
        assert m in LABELS
        assert m in STATUS_COLORS
        # Bypass should be visually loud (red)
        if m == PermissionMode.BYPASS:
            assert "red" in STATUS_COLORS[m]
