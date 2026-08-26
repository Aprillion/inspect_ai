"""The find-messages endpoint: request/response models and the search over one sample."""

import json
import re
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from inspect_ai._util.async_zip import AsyncZipReader
from inspect_ai._util.asyncfiles import AsyncFilesystem
from inspect_ai._util.textsearch import FoldedText, compile_query
from inspect_ai.event._pool import _expand_refs, materialize_pooled_events
from inspect_ai.event._validate import validate_chat_messages
from inspect_ai.log._condense import (
    ATTACHMENT_PROTOCOL,
    WalkContext,
    resolve_events_attachments,
    walk_chat_messages,
)
from inspect_ai.log._file import read_eval_log_sample_async
from inspect_ai.log._recorders.buffer import sample_buffer
from inspect_ai.log._recorders.chunked.format import (
    ATTACHMENTS_SEQUENCE,
    MESSAGES_SEQUENCE,
    chunk_entry_name,
    classify_sample_shape,
    sample_prefix,
    shell_entry_name,
)
from inspect_ai.model._chat_message import ChatMessage

from ._projection import DisplayMode, ProjectionOptions, ToolCallStyle, project_row
from ._rows import MessageRow, message_rows, messages_from_events, row_anchors

MAX_LIMIT = 1000  # a guess: bounds response size, the client pages


class FindMessagesCursor(BaseModel):
    anchor: str


class FindMessagesProjection(BaseModel):
    """How the viewer currently shows the rows; a host sends only what it changed."""

    unlabeled_roles: list[str] = []
    tool_call_style: ToolCallStyle = "complete"
    display_mode: DisplayMode = "rendered"
    """`raw` shows markdown source (link URLs included), so nothing is stripped."""


class FindMessagesRequest(BaseModel):
    sample_id: str | int
    epoch: int
    text: str
    direction: Literal["forward", "backward"]
    cursor: FindMessagesCursor | None = None
    """Resume strictly after (forward) or before (backward) this row; None starts at the near edge."""
    limit: int = Field(ge=1, le=MAX_LIMIT)
    """Maximum number of rows returned; larger values are rejected (422), not clamped."""
    projection: FindMessagesProjection = FindMessagesProjection()


class FindMessagesRow(BaseModel):
    anchor: str
    index: int
    """0-based position of the row in the sample's row list."""
    count: int
    """Non-overlapping matches in the row's projection; the client corrects it against the DOM."""
    texts: list[str]
    """The distinct source substrings matched, in first-appearance order."""


class FindMessagesTotal(BaseModel):
    rows: int
    occurrences: int
    relation: Literal["eq", "gte"]
    """inspect_ai always sends "eq" (the whole sample is scanned); a host that
    did not scan its whole universe (chunked or remote sources) sends "gte"."""


class FindMessagesResponse(BaseModel):
    rows: list[FindMessagesRow]
    """In the direction of travel: rows[0] is the next matching row from the cursor."""
    total: FindMessagesTotal
    complete: bool
    """Whether the universe was fully observed: false for a sample still being written."""


class _SampleIndex:
    """A sample's rows, folded once per projection options."""

    def __init__(self, messages: list[ChatMessage], complete: bool) -> None:
        self.complete = complete
        self.rows: list[MessageRow] = message_rows(messages)
        self.anchors: list[str] = row_anchors(self.rows)
        self.anchor_index = {anchor: i for i, anchor in enumerate(self.anchors)}
        self.roles = frozenset(row.message.role for row in self.rows)
        self._folded: OrderedDict[ProjectionOptions, list[FoldedText]] = OrderedDict()

    def folded_rows(self, options: ProjectionOptions) -> list[FoldedText]:
        # roles absent from the sample cannot change the projection, so they
        # do not get their own cache entry
        options = replace(options, unlabeled_roles=options.unlabeled_roles & self.roles)
        folded = self._folded.get(options)
        if folded is not None:
            self._folded.move_to_end(options)
        else:
            folded = [
                FoldedText("\n".join(s.text for s in project_row(row, anchor, options)))
                for row, anchor in zip(self.rows, self.anchors, strict=True)
            ]
            self._folded[options] = folded
            while len(self._folded) > _MAX_FOLDED_VARIANTS:
                self._folded.popitem(last=False)
        return folded


_MAX_FOLDED_VARIANTS = (
    4  # a guess: a viewer holds one option set, a host toggling a few
)
_MAX_CACHED_SAMPLES = 8  # a guess: a few open tabs; completed samples are immutable

# keyed by path only: a log rewritten at the same path serves stale rows until
# eviction; stat-ing remote logs on every keystroke is not worth that rare case
_cache: OrderedDict[tuple[str, str, str, int], _SampleIndex] = OrderedDict()


def _index_key(
    location: str, sample_id: str | int, epoch: int
) -> tuple[str, str, str, int]:
    return location, type(sample_id).__name__, str(sample_id), epoch


async def find_messages(
    location: str, request: FindMessagesRequest
) -> FindMessagesResponse | None:
    """Search one sample's Messages rows; None when the sample is not found."""
    index = await _sample_index(location, request.sample_id, request.epoch)
    if index is None:
        return None
    options = ProjectionOptions(
        frozenset(request.projection.unlabeled_roles),
        request.projection.tool_call_style,
        request.projection.display_mode,
    )
    return _page(index, index.folded_rows(options), request)


class _RowMatches(NamedTuple):
    occurrences: int
    texts: list[str]


def _page(
    index: _SampleIndex, rows: list[FoldedText], request: FindMessagesRequest
) -> FindMessagesResponse:
    anchors = index.anchors
    query = compile_query(request.text)
    matching = [
        (i, matches)
        for i, matches in ((i, _row_matches(r, query)) for i, r in enumerate(rows))
        if matches.occurrences
    ]
    indices = [i for i, _ in matching]
    cursor = index.anchor_index.get(request.cursor.anchor) if request.cursor else None
    if request.direction == "forward":
        first = bisect_right(indices, cursor) if cursor is not None else 0
        page = matching[first : first + request.limit]
    else:
        last = bisect_left(indices, cursor) if cursor is not None else len(matching)
        page = matching[max(0, last - request.limit) : last][::-1]
    return FindMessagesResponse(
        rows=[
            FindMessagesRow(
                anchor=anchors[i], index=i, count=m.occurrences, texts=m.texts
            )
            for i, m in page
        ],
        total=FindMessagesTotal(
            rows=len(matching),
            occurrences=sum(m.occurrences for _, m in matching),
            relation="eq",
        ),
        complete=index.complete,
    )


def _row_matches(row: FoldedText, query: re.Pattern[str] | None) -> _RowMatches:
    texts: dict[str, None] = {}
    count = 0
    for start, end in row.find_all(query):
        texts.setdefault(row.text[start:end])
        count += 1
    return _RowMatches(count, list(texts))


async def _sample_index(
    location: str, sample_id: str | int, epoch: int
) -> _SampleIndex | None:
    key = _index_key(location, sample_id, epoch)
    cached = _cache.get(key)
    if cached is not None:
        _cache.move_to_end(key)
        return cached
    messages = await _logged_messages(location, sample_id, epoch)
    if messages is not None:
        index = _SampleIndex(messages, complete=True)
        _cache[key] = index
        while len(_cache) > _MAX_CACHED_SAMPLES:
            _cache.popitem(last=False)
        return index
    # a running sample is still changing, so it is never cached
    messages = _running_messages(location, sample_id, epoch)
    if messages is not None:
        return _SampleIndex(messages, complete=False)
    # the sample may have been sealed into the log between the two probes
    messages = await _logged_messages(location, sample_id, epoch)
    return _SampleIndex(messages, complete=True) if messages is not None else None


async def _logged_messages(
    location: str, sample_id: str | int, epoch: int
) -> list[ChatMessage] | None:
    try:
        sample = await read_eval_log_sample_async(
            location,
            sample_id,
            epoch,
            resolve_attachments="core",
            exclude_fields={"events", "store"},
        )
    except IndexError:
        return await _chunked_messages(location, sample_id, epoch)
    return sample.messages


async def _chunked_messages(
    location: str, sample_id: str | int, epoch: int
) -> list[ChatMessage] | None:
    """The final conversation of a sample in the chunked per-sample shape.

    `read_eval_log_sample_async` only reads monolith samples; the chunked shape
    keeps the conversation as `message_refs` into a `messages/` sequence with
    text extracted to an `attachments/` sequence (`attachment://<index>`).
    """
    async with AsyncFilesystem() as fs:
        reader = AsyncZipReader(fs, location)
        names = {entry.filename for entry in (await reader.entries()).entries}
        if classify_sample_shape(names, sample_id, epoch) != "chunked":
            return None
        shell = json.loads(
            await reader.read_member_fully(shell_entry_name(sample_id, epoch))
        )
        pool = await _read_sequence(reader, names, sample_id, epoch, MESSAGES_SEQUENCE)
        attachments = await _read_sequence(
            reader, names, sample_id, epoch, ATTACHMENTS_SEQUENCE
        )
    messages = validate_chat_messages(
        _expand_refs(shell["message_refs"], pool), context={"deserializing": True}
    )

    def content_fn(text: str) -> str:
        if text.startswith(ATTACHMENT_PROTOCOL):
            index = text.removeprefix(ATTACHMENT_PROTOCOL)
            if index.isdigit() and int(index) < len(attachments):
                return str(attachments[int(index)])
        return text

    context = WalkContext(message_cache={}, only_core=True)
    return walk_chat_messages(messages, content_fn, context)


async def _read_sequence(
    reader: AsyncZipReader,
    names: set[str],
    sample_id: str | int,
    epoch: int,
    sequence: str,
) -> list[Any]:
    """A whole chunked sequence; chunks are named by their first item's index."""
    prefix = f"{sample_prefix(sample_id, epoch)}/{sequence}/"
    starts = sorted(
        int(stem)
        for name in names
        if name.startswith(prefix)
        and (stem := name[len(prefix) :].removesuffix(".json")).isdigit()
    )
    items: list[Any] = []
    for start in starts:
        entry = chunk_entry_name(sample_id, epoch, sequence, start)
        items.extend(json.loads(await reader.read_member_fully(entry)))
    return items


def _running_messages(
    location: str, sample_id: str | int, epoch: int
) -> list[ChatMessage] | None:
    """Messages of a sample still in the recorder's buffer, as the live Messages tab shows them.

    Sync on the event loop like `/pending-sample-data`: the buffer can be
    filestore-backed (fsspec) and must not be wrapped in `to_thread`.
    """
    try:
        buffer = sample_buffer(location)
    except FileNotFoundError:
        return None
    data = buffer.get_sample_data(id=str(sample_id), epoch=epoch)
    if data is None:
        return None
    message_pool = validate_chat_messages(
        [json.loads(entry.data) for entry in data.message_pool],
        context={"deserializing": True},
    )
    call_pool = [json.loads(entry.data) for entry in data.call_pool]
    events = materialize_pooled_events(
        [entry.event for entry in data.events], message_pool, call_pool
    )
    attachments = {entry.hash: entry.content for entry in data.attachments}
    return messages_from_events(resolve_events_attachments(events, attachments))
