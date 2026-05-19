"""Tests for paper-distiller-chat 'browse' subcommand."""

from paper_distiller.chat.cli import _parse_picks


def test_parse_picks_simple_list():
    assert _parse_picks("1,3,5", 10) == [1, 3, 5]


def test_parse_picks_range():
    assert _parse_picks("2-5", 10) == [2, 3, 4, 5]


def test_parse_picks_mixed():
    assert _parse_picks("1,3-5,7", 10) == [1, 3, 4, 5, 7]


def test_parse_picks_all():
    assert _parse_picks("all", 5) == [1, 2, 3, 4, 5]


def test_parse_picks_cancel():
    assert _parse_picks("q", 10) == []
    assert _parse_picks("quit", 10) == []
    assert _parse_picks("", 10) == []


def test_parse_picks_invalid():
    assert _parse_picks("abc", 10) is None
    assert _parse_picks("11", 10) is None  # out of range (only 10 candidates)
    assert _parse_picks("0", 10) is None   # 1-based, 0 invalid


def test_parse_picks_dedup_and_sort():
    assert _parse_picks("3,1,2,1", 10) == [1, 2, 3]


def test_parse_picks_range_too_wide():
    assert _parse_picks("5-3", 10) is None   # reversed range
    assert _parse_picks("8-12", 10) is None  # out of range


def test_browse_cli_parses_args():
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args(["browse", "--vault", "/tmp/v", "--topic", "X", "--n", "8"])
    assert args.subcommand == "browse"
    assert args.topic == "X"
    assert args.n == 8
