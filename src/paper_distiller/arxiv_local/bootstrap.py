"""Bootstrap the local arxiv mirror from a bulk metadata dump.

Source chain (auto mode):
  1. Internet Archive mirror — no auth, fast CDN
  2. Kaggle (Cornell-University/arxiv) — needs API token
  3. OAI-PMH from scratch — no auth, slow (~6-7 hours)

The dump format is line-delimited JSON (`*.json` or `*.json.gz`). We stream-
parse to avoid loading the full ~1 GB into memory.
"""

from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx

from .store import PaperRow, Store


_INTERNET_ARCHIVE_URL = (
    "https://archive.org/download/arxiv-bulk-metadata/"
    "arxiv-metadata-oai-snapshot.json.gz"
)


class BootstrapError(RuntimeError):
    pass


@dataclass
class IngestResult:
    n_inserted: int = 0
    n_malformed: int = 0
    n_skipped: int = 0


def parse_kaggle_record(record: dict) -> PaperRow:
    """Convert one Kaggle/arxiv-metadata JSONL record into a PaperRow.

    Kaggle dump uses these keys:
      id, title, authors_parsed, abstract, categories (space-separated),
      doi, comments, journal-ref, update_date, versions
    """
    authors_parsed = record.get("authors_parsed") or []
    authors = []
    for a in authors_parsed:
        if not isinstance(a, list) or len(a) < 2:
            continue
        last = str(a[0]).strip()
        first = str(a[1]).strip()
        full = f"{first} {last}".strip()
        if full:
            authors.append(full)

    categories_raw = record.get("categories", "") or ""
    categories = [c for c in categories_raw.split() if c]
    primary = categories[0] if categories else None

    update_date = record.get("update_date") or ""

    return PaperRow(
        arxiv_id=str(record.get("id", "")),
        title=(record.get("title") or "").strip(),
        authors=authors,
        abstract=(record.get("abstract") or "").strip(),
        categories=categories,
        primary_category=primary,
        published=update_date,
        updated=None,
        doi=record.get("doi") or None,
        comment=record.get("comments") or None,
        journal_ref=record.get("journal-ref") or None,
        source="bootstrap",
    )


def _open_jsonl(path: Path):
    """Open .jsonl, .json, or .json.gz transparently."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def ingest_jsonl(
    jsonl_path: Path,
    store: Store,
    batch_size: int = 10_000,
    progress_cb=None,
) -> IngestResult:
    """Stream-parse a Kaggle-format JSONL into the store."""
    result = IngestResult()
    batch: list[PaperRow] = []
    line_no = 0

    with _open_jsonl(jsonl_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                result.n_malformed += 1
                continue
            try:
                paper = parse_kaggle_record(rec)
            except Exception:
                result.n_malformed += 1
                continue
            if not paper.arxiv_id or not paper.title:
                result.n_skipped += 1
                continue
            batch.append(paper)
            if len(batch) >= batch_size:
                store.upsert_many(batch)
                result.n_inserted += len(batch)
                batch.clear()
                if progress_cb:
                    progress_cb(line_no, result.n_inserted)

    if batch:
        store.upsert_many(batch)
        result.n_inserted += len(batch)
        if progress_cb:
            progress_cb(line_no, result.n_inserted)

    return result


def _download_to_file(url: str, dest: Path, timeout: float = 600.0) -> None:
    """Stream-download `url` → `dest`. 10-minute default timeout."""
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)


def _bootstrap_internet_archive(work_dir: Path, store: Store) -> IngestResult:
    """Pull arxiv bulk metadata from Internet Archive's mirror (no auth)."""
    dump_path = work_dir / "arxiv-metadata.json.gz"
    print(f"[bootstrap] downloading from internet_archive → {dump_path}")
    try:
        _download_to_file(_INTERNET_ARCHIVE_URL, dump_path)
    except (httpx.HTTPError, OSError) as e:
        raise BootstrapError(f"internet_archive download failed: {e}") from e
    print(f"[bootstrap] ingesting {dump_path}")
    return ingest_jsonl(dump_path, store)


def _bootstrap_kaggle(work_dir: Path, store: Store) -> IngestResult:
    """Pull from Kaggle. Requires KAGGLE_USERNAME + KAGGLE_KEY env or
    ~/.kaggle/kaggle.json."""
    try:
        import kaggle  # type: ignore
    except ImportError:
        raise BootstrapError(
            "kaggle python package not installed; run `pip install kaggle`"
        )
    print("[bootstrap] downloading from kaggle (Cornell-University/arxiv)")
    try:
        kaggle.api.dataset_download_files(
            "Cornell-University/arxiv",
            path=str(work_dir),
            unzip=True,
        )
    except Exception as e:
        raise BootstrapError(f"kaggle download failed: {e}") from e

    candidates = list(work_dir.glob("*.json")) + list(work_dir.glob("*.jsonl"))
    if not candidates:
        raise BootstrapError("kaggle download produced no JSONL file")
    return ingest_jsonl(candidates[0], store)


def _bootstrap_oai_pmh(work_dir: Path, store: Store, since: str | None) -> IngestResult:
    """Pull from arxiv's OAI-PMH endpoint, optionally bounded by `since` date.

    No auth, no dump file — streams directly from the server in 1000-record
    pages. `since=None` means full harvest (~6-7h, 1.7M papers). `since` set
    to e.g. '2024-01-01' bounds it to ~2 years (~2h, ~600k papers).

    work_dir is unused (we don't materialize a file) but kept for signature
    compatibility with the other bootstrap_X helpers.
    """
    _ = work_dir
    from .incremental import sync as oai_sync
    print(
        f"[bootstrap] starting OAI-PMH harvest "
        f"({'from ' + since if since else 'full catalog'}); "
        "this may take hours."
    )
    sync_result = oai_sync(store, since=since)
    return IngestResult(n_inserted=sync_result.n_inserted)


def bootstrap(
    store: Store,
    source: str = "auto",
    since: str | None = None,
    work_dir: Path | None = None,
    keep_dump: bool = False,
) -> IngestResult:
    """Run a full bootstrap. `source` ∈ auto / internet_archive / kaggle / oai_pmh.

    `since`: ISO date (e.g. '2024-01-01'). Only honored by `oai_pmh` source
    — internet_archive and kaggle deliver pre-built dumps with their own
    snapshot dates.

    Default chain (`source='auto'`): try OAI-PMH first because it's the only
    source guaranteed to work without auth and with current data. Internet
    Archive's bulk-metadata item currently only has 2017-2018 XML dumps;
    Kaggle requires API credentials. Both stay in the chain as last-resort
    fallbacks for future when those paths get fixed.
    """
    from datetime import datetime, timezone

    work_dir = Path(work_dir or (store.path.parent / "dump.tmp"))
    work_dir.mkdir(parents=True, exist_ok=True)

    sources_to_try = (
        ["oai_pmh", "internet_archive", "kaggle"]
        if source == "auto"
        else [source]
    )

    last_err = None
    chosen = None
    result = None
    for src in sources_to_try:
        try:
            if src == "internet_archive":
                result = _bootstrap_internet_archive(work_dir, store)
            elif src == "kaggle":
                result = _bootstrap_kaggle(work_dir, store)
            elif src == "oai_pmh":
                result = _bootstrap_oai_pmh(work_dir, store, since=since)
            else:
                raise BootstrapError(f"unknown source: {src!r}")
            chosen = src
            break
        except BootstrapError as e:
            last_err = e
            print(f"[bootstrap] {src} failed: {e}; trying next source")
            continue

    if result is None:
        raise BootstrapError(
            f"all bootstrap sources exhausted; last error: {last_err}"
        )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.save_state({
        "bootstrap_source": chosen,
        "bootstrap_completed_at": now,
        "last_sync": now,
    })

    if not keep_dump:
        shutil.rmtree(work_dir, ignore_errors=True)

    return result
