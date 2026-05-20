"""CLI for managing the local arxiv mirror.

Commands:
  bootstrap   one-time bulk download + ingest
  sync        incremental OAI-PMH update since last sync
  search      query local FTS5 (no API calls)
  stats       db size, paper count, last sync
  doctor      diagnose connectivity, integrity, schema
"""

from __future__ import annotations

import argparse
import sys

# UTF-8 stdout reconfigure for Windows GBK terminals (mirrors chat/cli.py)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from .store import DEFAULT_DIR, Store  # noqa: E402


def _open_store() -> Store:
    return Store(DEFAULT_DIR / "arxiv.db")


def _cmd_bootstrap(args) -> int:
    from .bootstrap import bootstrap, BootstrapError
    store = _open_store()
    if store.paper_count() > 0 and not args.force:
        print(
            f"Store already has {store.paper_count():,} papers. "
            "Use --force to re-bootstrap, or --since <DATE> to top up via OAI.",
            file=sys.stderr,
        )
        store.close()
        return 2
    try:
        result = bootstrap(store, source=args.source, since=args.since)
    except BootstrapError as e:
        print(f"bootstrap failed: {e}", file=sys.stderr)
        store.close()
        return 3
    print(
        f"bootstrap complete: {result.n_inserted:,} papers inserted, "
        f"{result.n_malformed} malformed, {result.n_skipped} skipped."
    )
    store.close()
    return 0


def _cmd_sync(args) -> int:
    from .incremental import sync
    store = _open_store()
    try:
        result = sync(store, since=args.since)
    except Exception as e:
        print(f"sync failed: {e}", file=sys.stderr)
        store.close()
        return 3
    print(
        f"sync complete: {result.n_seen:,} seen, "
        f"{result.n_inserted:,} inserted, {result.n_deleted:,} deleted, "
        f"{result.duration_sec:.1f}s elapsed."
    )
    store.close()
    return 0


def _cmd_search(args) -> int:
    from .search import search
    store = _open_store()
    results = search(
        store, args.query, n=args.n, sort=args.sort,
        primary_category=args.category,
    )
    for i, p in enumerate(results, start=1):
        print(f"[{i}] {p.arxiv_id}  {p.published}")
        print(f"    {p.title}")
        if args.verbose:
            print(f"    {', '.join(p.authors[:3])}")
            print(f"    {p.abstract[:200]}")
        print()
    if not results:
        print("(no matches)")
    store.close()
    return 0


def _cmd_stats(args) -> int:
    store = _open_store()
    n = store.paper_count()
    state = store.load_state()
    db_path = DEFAULT_DIR / "arxiv.db"
    db_size = db_path.stat().st_size / (1024 ** 2) if db_path.exists() else 0
    print(f"papers:            {n:,}")
    print(f"db size:           {db_size:.1f} MB")
    print(f"schema version:    {state['schema_version']}")
    print(f"bootstrap source:  {state.get('bootstrap_source') or '(never)'}")
    print(f"last sync:         {state.get('last_sync') or '(never)'}")
    store.close()
    return 0


def _cmd_doctor(args) -> int:
    import httpx
    store = _open_store()
    db_path = DEFAULT_DIR / "arxiv.db"

    print("== paper-distiller-arxiv doctor ==")
    print(f"db path:       {db_path}")
    print(f"db exists:     {db_path.exists()}")
    print(f"paper count:   {store.paper_count():,}")

    try:
        store._conn.execute("PRAGMA integrity_check").fetchone()
        print("integrity:     OK")
    except Exception as e:
        print(f"integrity:     FAIL — {e}")

    try:
        r = httpx.get(
            "https://export.arxiv.org/oai2?verb=Identify", timeout=10.0,
        )
        print(f"oai-pmh:       reachable ({r.status_code})")
    except Exception as e:
        print(f"oai-pmh:       UNREACHABLE — {e}")

    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paper-distiller-arxiv")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser(
        "bootstrap",
        help="One-time bulk download + ingest. Default chain tries OAI-PMH "
             "first (no auth required). With --since X, only fetches papers "
             "from date X onwards (much faster than full catalog).",
    )
    b.add_argument(
        "--source",
        choices=["auto", "internet_archive", "kaggle", "oai_pmh"],
        default="auto",
    )
    b.add_argument(
        "--since",
        help="ISO date like '2024-01-01'. Only honored by oai_pmh source. "
             "Omit for full catalog (~6-7h, 1.7M papers).",
    )
    b.add_argument(
        "--force", action="store_true",
        help="Re-bootstrap even if DB already has papers",
    )

    s = sub.add_parser("sync", help="OAI-PMH incremental update")
    s.add_argument(
        "--since", help="ISO date; default uses last_sync from state",
    )

    sr = sub.add_parser("search", help="Local FTS5 search (no API calls)")
    sr.add_argument("query")
    sr.add_argument("--n", type=int, default=10)
    sr.add_argument(
        "--sort", choices=["relevance", "date"], default="relevance",
    )
    sr.add_argument(
        "--category", help="Filter by primary_category (e.g. cs.LG)",
    )
    sr.add_argument("--verbose", "-v", action="store_true")

    sub.add_parser("stats", help="Show DB stats")
    sub.add_parser("doctor", help="Diagnose mirror health")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "bootstrap": _cmd_bootstrap,
        "sync": _cmd_sync,
        "search": _cmd_search,
        "stats": _cmd_stats,
        "doctor": _cmd_doctor,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 2
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
