"""The source-text projection of a Messages tab row.

A row becomes the texts the viewer renders for it, in render order, taken from
the log as written (markdown source, not rendered text). The server only has to
say which rows match and roughly how often; the client's DOM is authoritative
inside a row, so no attempt is made to reproduce the viewer's formatting.
"""

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from inspect_ai._util.citation import Citation, UrlCitation
from inspect_ai._util.content import (
    Content,
    ContentData,
    ContentDocument,
    ContentReasoning,
    ContentText,
    ContentToolUse,
)
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool._tool_call import ToolCall

from ._markdown import strip_markdown_for_count
from ._rows import MessageRow

ToolCallStyle = Literal["complete", "compact", "omit"]
DisplayMode = Literal["rendered", "raw"]
"""The viewer's toggle: `raw` shows markdown source, so nothing is stripped."""
SegmentField = Literal[
    "role", "content", "citation", "reasoning", "tool_call", "tool_output", "tool_error"
]


@dataclass(frozen=True)
class Segment:
    anchor: str
    field: SegmentField
    text: str


@dataclass(frozen=True)
class ProjectionOptions:
    unlabeled_roles: frozenset[str] = frozenset()
    tool_call_style: ToolCallStyle = "complete"
    display_mode: DisplayMode = "rendered"


def projection(
    rows: Sequence[MessageRow],
    anchors: Sequence[str],
    options: ProjectionOptions = ProjectionOptions(),
    include_chrome: bool = True,
) -> list[Segment]:
    return [
        segment
        for row, anchor in zip(rows, anchors, strict=True)
        for segment in project_row(row, anchor, options, include_chrome)
    ]


def project_row(
    row: MessageRow,
    anchor: str,
    options: ProjectionOptions = ProjectionOptions(),
    include_chrome: bool = True,
) -> list[Segment]:
    """Project one row; chrome is text the viewer adds (role heading, "Reasoning" title)."""
    builder = _Builder(anchor, include_chrome, options.display_mode)
    message = row.message
    if message.role not in options.unlabeled_roles:
        builder.add("role", message.role, chrome=True)
    if isinstance(message.content, str):
        builder.add_markdown("content", message.content)
    else:
        for item in message.content:
            _content(builder, item)
    if isinstance(message, ChatMessageAssistant) and options.tool_call_style != "omit":
        for index, call in enumerate(message.tool_calls or []):
            _tool_call(builder, row, index, call, options.tool_call_style)
    return builder.segments


class _Builder:
    def __init__(
        self, anchor: str, include_chrome: bool, display_mode: DisplayMode = "rendered"
    ) -> None:
        self.anchor = anchor
        self.include_chrome = include_chrome
        self.display_mode = display_mode
        self.segments: list[Segment] = []

    def add(self, field: SegmentField, text: str, chrome: bool = False) -> None:
        if text and (self.include_chrome or not chrome):
            self.segments.append(Segment(self.anchor, field, text))

    def add_markdown(self, field: SegmentField, text: str) -> None:
        if self.display_mode == "rendered":
            text = strip_markdown_for_count(text)
        self.add(field, text)


def _content(builder: _Builder, item: Content) -> None:
    if isinstance(item, ContentText):
        builder.add_markdown("content", item.text)
        for citation in item.citations or []:
            builder.add("citation", _citation_label(citation))
    elif isinstance(item, ContentReasoning):
        builder.add("reasoning", _reasoning_title(item), chrome=True)
        builder.add_markdown("reasoning", _reasoning_text(item))
    elif isinstance(item, ContentData):
        builder.add("content", _text(item.data))
    elif isinstance(item, ContentDocument):
        builder.add("content", item.filename)
    elif isinstance(item, ContentToolUse):
        builder.add("tool_call", f"tool: {item.name}", chrome=True)
        builder.add("tool_call", item.context or "")
        builder.add("tool_call", item.name)
        builder.add("tool_call", item.arguments)
        if item.error:
            builder.add("tool_error", item.error)
        else:
            builder.add("tool_output", item.result)


def _citation_label(citation: Citation) -> str:
    """What the viewer prints for a citation: title, else quoted text, else URL; entities decoded."""
    if citation.title:
        label = citation.title
    elif isinstance(citation.cited_text, str):
        label = citation.cited_text
    else:
        label = citation.url if isinstance(citation, UrlCitation) else ""
    return html.unescape(label)


def _reasoning_text(item: ContentReasoning) -> str:
    """Redacted reasoning is ciphertext; the viewer shows its summary instead."""
    if item.redacted:
        return item.summary or ""
    return item.reasoning or item.summary or ""


def _reasoning_title(item: ContentReasoning) -> str:
    shows_summary = bool(item.summary) and (item.redacted or not item.reasoning)
    return "Reasoning (Summary)" if shows_summary else "Reasoning"


def _tool_call(
    builder: _Builder,
    row: MessageRow,
    index: int,
    call: ToolCall,
    style: ToolCallStyle,
) -> None:
    builder.add("tool_call", f"tool: {call.function}", chrome=True)
    if call.view is not None and call.view.title:
        builder.add("tool_call", call.view.title, chrome=True)
    builder.add("tool_call", call.function)
    for value in call.arguments.values():
        builder.add("tool_call", _text(value))
    if style == "compact":
        return
    tool_message = _tool_message(row, index, call)
    if tool_message is None:
        return
    if tool_message.error:
        builder.add("tool_error", tool_message.error.message)
    else:
        _tool_output(builder, tool_message)


def _tool_message(
    row: MessageRow, index: int, call: ToolCall
) -> ChatMessageTool | None:
    if call.id:
        return next((m for m in row.tool_messages if m.tool_call_id == call.id), None)
    return row.tool_messages[index] if index < len(row.tool_messages) else None


def _tool_output(builder: _Builder, message: ChatMessageTool) -> None:
    content = message.content
    for part in [content] if isinstance(content, str) else content:
        if isinstance(part, str):
            builder.add("tool_output", part)
        elif isinstance(part, ContentText):
            builder.add("tool_output", part.text)
        elif isinstance(part, ContentReasoning):
            builder.add("tool_output", _reasoning_text(part))
        elif isinstance(part, ContentData):
            builder.add("tool_output", _text(part.data))


def _text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
