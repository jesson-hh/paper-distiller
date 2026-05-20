"""OAI-PMH incremental sync against export.arxiv.org/oai2.

arxiv's OAI server gives 1000 records per resumption-token request, with an
implicit ~15s server-side throttle between batches. Sickle handles resumption
tokens automatically; we just iterate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .oai_record import record_to_paper
from .store import Store


_OAI_ENDPOINT = "https://export.arxiv.org/oai2"


@dataclass
class SyncResult:
    n_seen: int = 0
    n_inserted: int = 0
    n_deleted: int = 0
    duration_sec: float = 0.0


def sync(
    store: Store,
    since: str | None = None,
    metadata_prefix: str = "arXiv",
    set_spec: str | None = None,
    progress_cb=None,
    sickle_client=None,
) -> SyncResult:
    """Pull all records updated since `since` (ISO date) into the store.

    Arguments:
        store: open Store
        since: ISO date string (e.g. "2024-01-15") or None to use last_sync from state
        metadata_prefix: 'arXiv' for arxiv-specific schema; 'oai_dc' for Dublin Core
        set_spec: optional category filter (e.g. "cs"); None = all
        sickle_client: optional injected client for tests
    """
    import time

    state = store.load_state()
    from_date = since or state.get("last_sync")

    if sickle_client is None:
        from sickle import Sickle
        sickle_client = Sickle(_OAI_ENDPOINT)

    kwargs = {"metadataPrefix": metadata_prefix}
    if from_date:
        kwargs["from"] = from_date.split("T")[0] if "T" in from_date else from_date
    if set_spec:
        kwargs["set"] = set_spec

    t0 = time.monotonic()
    result = SyncResult()
    batch: list = []
    BATCH = 1000

    records = sickle_client.ListRecords(**kwargs)
    for record in records:
        result.n_seen += 1
        if record.deleted:
            result.n_deleted += 1
            continue
        paper = record_to_paper(record)
        if paper is None:
            continue
        batch.append(paper)
        if len(batch) >= BATCH:
            store.upsert_many(batch)
            result.n_inserted += len(batch)
            batch.clear()
            store.save_state({
                "last_sync": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            if progress_cb:
                progress_cb(result.n_seen, result.n_inserted)

    if batch:
        store.upsert_many(batch)
        result.n_inserted += len(batch)

    store.save_state({
        "last_sync": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    result.duration_sec = time.monotonic() - t0
    return result
