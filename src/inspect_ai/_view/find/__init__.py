"""Find on the Messages tab: row projection, browser-fold matching, endpoint models."""

from inspect_ai._util.textsearch import FoldedText, TextMatch, find_matches

from ._events import project_event
from ._markdown import strip_markdown_for_count
from ._messages import (
    FindMessagesCursor,
    FindMessagesProjection,
    FindMessagesRequest,
    FindMessagesResponse,
    FindMessagesRow,
    FindMessagesTotal,
    find_messages,
)
from ._projection import (
    ProjectionOptions,
    Segment,
    SegmentField,
    ToolCallStyle,
    project_row,
    projection,
)
from ._rows import (
    SYSTEM_ROW_ID,
    MessageRow,
    message_rows,
    messages_from_events,
    row_anchors,
)

__all__ = [
    "SYSTEM_ROW_ID",
    "FindMessagesCursor",
    "FindMessagesProjection",
    "FindMessagesRequest",
    "FindMessagesResponse",
    "FindMessagesRow",
    "FindMessagesTotal",
    "FoldedText",
    "MessageRow",
    "ProjectionOptions",
    "Segment",
    "SegmentField",
    "TextMatch",
    "ToolCallStyle",
    "find_matches",
    "find_messages",
    "message_rows",
    "messages_from_events",
    "project_event",
    "project_row",
    "projection",
    "row_anchors",
    "strip_markdown_for_count",
]
