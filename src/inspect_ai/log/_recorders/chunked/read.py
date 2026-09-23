"""Read a chunked sample back as the `EvalSample` its monolith would give."""

import json
import re
from collections.abc import Callable
from functools import partial
from typing import Any

import anyio

from inspect_ai._util._async import tg_collect
from inspect_ai._util.async_zip import AsyncZipReader
from inspect_ai._util.constants import get_deserializing_context
from inspect_ai._util.hash import mm3_hash
from inspect_ai._util.url import is_data_uri
from inspect_ai.event._pool import _expand_refs, collect_pool_ref_positions

from ..._condense import (
    ATTACHMENT_PROTOCOL,
    WalkContext,
    walk_chat_messages,
    walk_input,
)
from ..._log import EvalSample
from .format import (
    ATTACHMENTS_SEQUENCE,
    CALLS_SEQUENCE,
    EVENTS_SEQUENCE,
    MESSAGES_SEQUENCE,
    chunk_entry_name,
    events_uuids_entry_name,
    metadata_entry_name,
    sample_prefix,
    shell_entry_name,
)

# Inverse of the writer's `_attachment_ref_renumberer`, which rewrites
# `attachment://<32 hex>` anywhere in the serialized JSON. Matching the same
# scope is what keeps a ref the writer rewrote mid-string from staying
# dangling. The rewrite is not injective, so the inverse cannot always be
# exact. Any text beginning `attachment://` followed by digits is
# indistinguishable from a renumbered ref, whatever follows the digits — so
# `attachment://0abc` is rewritten and `attachment://2024-01-01/report.pdf`
# refers to attachment 2024 — and a renumbered ref with a digit written right
# after it reads as one larger index. Both losses belong to the conversion,
# and nothing in the archive can tell them apart.
_INDEX_REF = re.compile(rb"attachment://(\d+)")

# concurrent chunk reads, bounded as the journal-summary reads in `eval.py` are
_MAX_CHUNK_READS = 25


async def read_chunked_sample(
    reader: AsyncZipReader,
    names: set[str],
    id: str | int,
    epoch: int,
    location: str,
    exclude_fields: set[str] | None = None,
) -> EvalSample:
    """Inverse of `convert._write_chunked_sample`.

    Refs go back from `attachment://<index>` to `attachment://<hash>`, the final
    conversation is expanded from `message_refs` into the messages sequence, and
    the text the writer extracted from it and from `input` is inlined again (a
    monolith writer extracts only images from those two).

    An attachment's identity is `mm3_hash` of its content, as `_condense`
    assigns it; the chunked shape stores the contents alone and the map is keyed
    by rehashing them, so a writer that keyed attachments any other way would
    not round-trip.

    A sample whose chunk sequences do not reconstruct — a gap in a sequence, a
    ref past its end, a shell without `message_refs` — raises `ValueError`
    rather than returning a shorter or differently ordered sample.

    Args:
       reader: Reader for the log archive.
       names: Entry names in the archive.
       id: Sample id.
       epoch: Sample epoch.
       location: Log location, for error messages.
       exclude_fields: Sample fields to leave out of the result.
    """
    exclude = exclude_fields or set()
    sequence = partial(_read_sequence, reader, names, id, epoch, location)

    # read even when `attachments` is excluded: the exclusion says what the
    # caller gets back, and these contents are what the other sequences' index
    # refs resolve to
    attachments: list[str] = await sequence(ATTACHMENTS_SEQUENCE)
    hashes = [mm3_hash(content).encode() for content in attachments]
    contents = {hash.decode(): content for hash, content in zip(hashes, attachments)}

    def rekey(data: bytes) -> bytes:
        def hash_ref(match: re.Match[bytes]) -> bytes:
            index = int(match.group(1))
            if index >= len(hashes):
                raise ValueError(
                    f"Chunked sample id {id} for epoch {epoch} in log {location} "
                    f"refers to attachment {index} but its attachments sequence "
                    f"holds {len(hashes)}: either the sequence is incomplete, or "
                    f"the sample carried text beginning attachment:// followed "
                    f"by digits, which the conversion cannot tell from a reference"
                )
            return ATTACHMENT_PROTOCOL.encode() + hashes[index]

        return _INDEX_REF.sub(hash_ref, data)

    data: dict[str, Any] = json.loads(
        rekey(await reader.read_member_fully(shell_entry_name(id, epoch)))
    )
    message_refs = data.pop("message_refs", None)
    if message_refs is None:
        raise ValueError(
            f"Chunked sample id {id} for epoch {epoch} in log {location} has no "
            f"'message_refs' in {shell_entry_name(id, epoch)}"
        )
    pool: list[Any] = []
    if not {"messages", "events"} <= exclude:
        pool = await sequence(MESSAGES_SEQUENCE, rekey)
    check = partial(_check_ref_bounds, id=id, epoch=epoch, location=location)
    if "messages" not in exclude:
        check(
            lowest=min((start for start, _ in message_refs), default=0),
            past_end=max((end for _, end in message_refs), default=0),
            size=len(pool),
            sequence=MESSAGES_SEQUENCE,
        )
        data["messages"] = _expand_refs(message_refs, pool)
    if "events" not in exclude:
        events = await sequence(
            EVENTS_SEQUENCE, rekey, events_uuids_entry_name(id, epoch)
        )
        calls = await sequence(CALLS_SEQUENCE, rekey)
        # the events' own refs into both pools; the shell's message_refs cover
        # only the final conversation, which may reference neither pool fully
        positions = collect_pool_ref_positions(events)
        for referenced, size, name in (
            (positions.message_positions, len(pool), MESSAGES_SEQUENCE),
            (positions.call_positions, len(calls), CALLS_SEQUENCE),
        ):
            check(
                lowest=min(referenced, default=0),
                past_end=max(referenced, default=-1) + 1,
                size=size,
                sequence=name,
            )
        data["events"] = events
        data["events_data"] = {"messages": pool, "calls": calls}
    metadata_entry = metadata_entry_name(id, epoch)
    if "metadata" not in exclude and metadata_entry in names:
        data["metadata"] = json.loads(
            rekey(await reader.read_member_fully(metadata_entry))
        )
    for field in exclude:
        data.pop(field, None)
    data["attachments"] = {} if "attachments" in exclude else contents
    sample = EvalSample.model_validate(data, context=get_deserializing_context())
    return _inline_extracted_text(sample, contents)


def _inline_extracted_text(sample: EvalSample, contents: dict[str, str]) -> EvalSample:
    """Put back the text the converter extracted from `input` and `messages`.

    The attachment map is passed in because it is needed even when the caller
    excluded the field and `sample.attachments` is therefore empty.
    """

    def content_fn(text: str) -> str:
        if not text.startswith(ATTACHMENT_PROTOCOL):
            return text
        content = contents.get(text.removeprefix(ATTACHMENT_PROTOCOL))
        return text if content is None or is_data_uri(content) else content

    context = WalkContext(message_cache={}, only_core=False)
    return sample.model_copy(
        update={
            "input": walk_input(sample.input, content_fn, context),
            "messages": walk_chat_messages(sample.messages, content_fn, context),
        }
    )


def _check_ref_bounds(
    *,
    lowest: int,
    past_end: int,
    size: int,
    sequence: str,
    id: str | int,
    epoch: int,
    location: str,
) -> None:
    """Raise unless every referenced position exists in the sequence.

    `_expand_refs` slices, so a ref past the end would silently yield fewer
    items than were written — a short conversation, or a recorded prompt
    missing its last turns.
    """
    if lowest < 0 or past_end > size:
        raise ValueError(
            f"Chunked sample id {id} for epoch {epoch} in log {location} refers "
            f"to {sequence} [{lowest}, {past_end}) but its {sequence} sequence "
            f"holds {size}"
        )


async def _read_sequence(
    reader: AsyncZipReader,
    names: set[str],
    id: str | int,
    epoch: int,
    location: str,
    sequence: str,
    rekey: Callable[[bytes], bytes] | None = None,
    count_entry: str | None = None,
) -> list[Any]:
    """A sequence's items, from chunk entries each named by its first item's index.

    The names are what makes a gap detectable: a chunk starting anywhere but at
    the running item count means an entry between them is missing, and
    concatenating regardless would renumber every item after it. They say
    nothing about a missing last chunk, so `count_entry` names a sidecar with
    one element per item whose length settles the total.
    """
    prefix = f"{sample_prefix(id, epoch)}/{sequence}/"
    starts = sorted(
        int(stem)
        for name in names
        if name.startswith(prefix)
        and (stem := name[len(prefix) :].removesuffix(".json")).isdigit()
    )
    semaphore = anyio.Semaphore(_MAX_CHUNK_READS)

    async def read_chunk(start: int) -> bytes:
        async with semaphore:
            return await reader.read_member_fully(
                chunk_entry_name(id, epoch, sequence, start)
            )

    chunks = await tg_collect([partial(read_chunk, start) for start in starts])
    items: list[Any] = []
    for start, chunk in zip(starts, chunks):
        if start != len(items):
            raise ValueError(
                f"Chunked sample id {id} for epoch {epoch} in log {location} is "
                f"missing {sequence} items {len(items)} through {start - 1}"
            )
        items.extend(json.loads(chunk if rekey is None else rekey(chunk)))
    if count_entry is not None and count_entry in names:
        recorded = len(json.loads(await reader.read_member_fully(count_entry)))
        if len(items) != recorded:
            raise ValueError(
                f"Chunked sample id {id} for epoch {epoch} in log {location} "
                f"holds {len(items)} {sequence} items where {count_entry} "
                f"records {recorded}"
            )
    return items
