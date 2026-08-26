"""The data-text projection of a transcript event, for grep-style search."""

from inspect_ai._util.content import ContentDocument, ContentText
from inspect_ai.event._approval import ApprovalEvent
from inspect_ai.event._error import ErrorEvent
from inspect_ai.event._event import Event
from inspect_ai.event._info import InfoEvent
from inspect_ai.event._logger import LoggerEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.tool._tool import ToolResult

from ._projection import Segment, _Builder, _text


def project_event(event: Event, include_chrome: bool = True) -> list[Segment]:
    """Project one event; chrome is its title label, data the text it carries.

    Unsupported event kinds project to nothing.
    """
    builder = _Builder(event.uuid or "", include_chrome)
    if isinstance(event, ModelEvent):
        builder.add("role", event.model, chrome=True)
        builder.add("content", event.output.completion)
    elif isinstance(event, ToolEvent):
        builder.add("tool_call", event.function, chrome=True)
        for value in event.arguments.values():
            builder.add("tool_call", _text(value))
        if event.error is not None:
            builder.add("tool_error", event.error.message)
        else:
            _result(builder, event.result)
    elif isinstance(event, ErrorEvent):
        builder.add("tool_error", "Error", chrome=True)
        builder.add("tool_error", event.error.message)
    elif isinstance(event, InfoEvent):
        builder.add("content", event.source or "Info", chrome=True)
        if event.data is not None:
            builder.add("content", _text(event.data))
    elif isinstance(event, LoggerEvent):
        builder.add("content", event.message.level, chrome=True)
        builder.add("content", event.message.message)
    elif isinstance(event, ApprovalEvent):
        builder.add("tool_call", event.decision, chrome=True)
        builder.add("content", event.message)
        builder.add("tool_call", event.call.function)
        for value in event.call.arguments.values():
            builder.add("tool_call", _text(value))
        builder.add("content", event.explanation or "")
    return builder.segments


def _result(builder: _Builder, result: ToolResult) -> None:
    for part in result if isinstance(result, list) else [result]:
        if isinstance(part, ContentText):
            builder.add("tool_output", part.text)
        elif isinstance(part, ContentDocument):
            builder.add("tool_output", part.filename)
        elif isinstance(part, (str, int, float)):
            builder.add("tool_output", _text(part))
