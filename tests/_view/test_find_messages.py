"""Tests for the view server's find-messages endpoint and its projection."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import inspect_ai.log
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai._view import fastapi_server
from inspect_ai._view.find import (
    SYSTEM_ROW_ID,
    ProjectionOptions,
    find_matches,
    message_rows,
    messages_from_events,
    project_event,
    project_row,
    projection,
    row_anchors,
    strip_markdown_for_count,
)
from inspect_ai.event import Event
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.model._model_output import ChatCompletionChoice
from inspect_ai.tool import ToolCall, ToolCallContent, ToolCallError

QA_LOG = Path(__file__).parent / "test_find_messages" / "find-qa.eval"


def texts(messages: list[ChatMessage], **options: Any) -> list[str]:
    rows = message_rows(messages)
    opts = ProjectionOptions(**options)
    return [
        "\n".join(s.text for s in project_row(row, anchor, opts))
        for row, anchor in zip(rows, row_anchors(rows), strict=True)
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Fold and matching
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "query,text,expected",
    [
        (
            "istanbul",
            "İstanbul istanbul ISTANBUL ıstanbul",
            ["İstanbul", "istanbul", "ISTANBUL"],
        ),
        ("İstanbul", "istanbul ISTANBUL", ["istanbul", "ISTANBUL"]),
        ("ıstanbul", "İstanbul istanbul", []),
        ("strasse", "Straße straße STRASSE", ["Straße", "straße", "STRASSE"]),
        ("straße", "Straße STRASSE", ["Straße", "STRASSE"]),
        ("cafe", "café café", ["café", "café"]),
        ("café", "cafe café", ["cafe", "café"]),
        ("file", "ﬁle file", ["ﬁle", "file"]),
        ("ﬁle", "FILE", ["FILE"]),
        ("ss", "ßß", ["ß", "ß"]),
        ("sa", "ßa", []),
        ("as", "aß", []),
        ("fi", "ﬁ ﬁle", ["ﬁ", "ﬁ"]),
        ("il", "ﬁle", []),
        ("aa", "aaaaa", ["aa", "aa"]),
        ("", "anything", []),
    ],
)
def test_find_matches_fold(query: str, text: str, expected: list[str]) -> None:
    assert [m.text for m in find_matches(text, query)] == expected


@pytest.mark.parametrize(
    "query,text,spans",
    [
        ("ss", "ßß", [(0, 1), (1, 2)]),  # one source char expands to two folded
        ("aa", "aaaaa", [(0, 2), (2, 4)]),
        ("cafe", "café café", [(0, 4), (5, 10)]),  # NFC, then NFD
        ("e", "é", [(0, 2)]),  # the trailing combining mark belongs to the match
        ("fi", "xﬁle", [(1, 2)]),
        ("sa", "ßa xsa", [(4, 6)]),  # a boundary-straddling hit, then a valid one
    ],
)
def test_find_matches_offsets(
    query: str, text: str, spans: list[tuple[int, int]]
) -> None:
    assert [(m.start, m.end) for m in find_matches(text, query)] == spans


@pytest.mark.parametrize(
    "query,text,options,expected",
    [
        ("cafe", "Café café", {"fold": "case"}, []),
        ("café", "Café café", {"fold": "case"}, ["Café", "café"]),
        ("strasse", "Straße", {"fold": "case"}, ["Straße"]),
        ("café", "Café café", {"fold": "none"}, ["café"]),
        ("a.c", "a.c abc", {"fold": "none"}, ["a.c"]),
        ("a.c", "a.c abc", {"mode": "regex", "fold": "none"}, ["a.c", "abc"]),
        ("A\\d", "a1 A2", {"mode": "regex", "fold": "case"}, ["a1", "A2"]),
        ("A\\d", "a1 A2", {"mode": "regex", "fold": "none"}, ["A2"]),
        ("strasse\\b", "Straße Straßen", {"mode": "regex"}, ["Straße"]),
        ("cat", "cat catalog concat", {"word_boundary": True}, ["cat"]),
        ("ca.", "cat catalog", {"mode": "regex", "word_boundary": True}, ["cat"]),
        ("x*", "xx yy", {"mode": "regex"}, ["xx"]),
        ("x*|y", "y", {"mode": "regex"}, ["y"]),
        ("x*|y", "yxy", {"mode": "regex"}, ["y", "x", "y"]),
        (
            "cat|dog",
            "catalog hotdog dog",
            {"mode": "regex", "word_boundary": True},
            ["dog"],
        ),
    ],
)
def test_find_matches_modes(
    query: str, text: str, options: dict[str, Any], expected: list[str]
) -> None:
    assert [m.text for m in find_matches(text, query, **options)] == expected


def test_find_matches_invalid_regex_raises() -> None:
    import re

    with pytest.raises(re.error):
        find_matches("x", "(", mode="regex")


# ═══════════════════════════════════════════════════════════════════════════
# Rows and anchors
# ═══════════════════════════════════════════════════════════════════════════


def test_rows_fold_tool_messages_and_merge_system() -> None:
    messages: list[ChatMessage] = [
        ChatMessageTool(id="orphan", content="dropped"),
        ChatMessageSystem(id="s1", content="be terse"),
        ChatMessageUser(id="u1", content="hi"),
        ChatMessageSystem(id="s2", content=[ContentText(text="and kind")]),
        ChatMessageAssistant(
            id="a1",
            content="",
            tool_calls=[ToolCall(id="c1", function="bash", arguments={"cmd": "ls"})],
        ),
        ChatMessageTool(id="t1", tool_call_id="c1", function="bash", content="a b"),
        ChatMessageTool(id="t2", tool_call_id="c2", function="bash", content="c"),
    ]
    rows = message_rows(messages)
    assert [row.message.id for row in rows] == [SYSTEM_ROW_ID, "u1", "a1"]
    assert [[t.id for t in row.tool_messages] for row in rows] == [[], [], ["t1", "t2"]]
    system = rows[0].message
    assert isinstance(system.content, list)
    assert [c.text for c in system.content if isinstance(c, ContentText)] == [
        "be terse",
        "and kind",
    ]


def test_row_anchors_first_occurrence_keeps_id() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(id="dup", content="a"),
        ChatMessageAssistant(content="no id").model_copy(update={"id": None}),
        ChatMessageUser(id="dup", content="b"),
        ChatMessageUser(id="", content="empty id"),
    ]
    rows = message_rows(messages)
    assert row_anchors(rows) == ["dup", "msg-1", "dup#2", ""]


def test_row_anchors_minted_never_collide_with_literal_ids() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(id="dup", content="a"),
        ChatMessageUser(id="dup#2", content="b"),
        ChatMessageUser(id="dup", content="c"),
        ChatMessageUser(id="", content="d"),
        ChatMessageUser(id="", content="e"),
        ChatMessageUser(id="#4", content="f"),
    ]
    anchors = row_anchors(message_rows(messages))
    # only prior rows count: the literal "#4" arriving later takes the suffix
    assert anchors == ["dup", "dup#2", "dup#2#2", "", "#4", "#4#5"]
    assert len(set(anchors)) == len(anchors)


def test_row_anchors_stable_under_append() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(id="a", content="1"),
        ChatMessageUser(id="a", content="2"),
    ]
    before = row_anchors(message_rows(messages))
    messages.append(ChatMessageUser(id="a#1", content="3"))
    after = row_anchors(message_rows(messages))
    assert before == ["a", "a#1"]
    assert after == ["a", "a#1", "a#1#2"]


def test_messages_from_events_merges_inputs_and_outputs() -> None:
    u1 = ChatMessageUser(id="u1", content="first")
    a1 = ChatMessageAssistant(id="a1", content="reply")
    u2 = ChatMessageUser(id="u2", content="second")
    a2 = ChatMessageAssistant(id="a2", content="reply two")
    summary = ChatMessageUser(id="sum", content="compaction summary")
    events = [
        model_event([u1], a1),
        model_event([u1, a1, u2], a2, error="boom"),
        model_event([u1, a1, u2], a2),
        model_event(
            [
                summary,
                u2,
                a2,
                ChatMessageUser(content="x").model_copy(update={"id": None}),
            ],
            None,
        ),
    ]
    # an unseen message leading an input lands after the list end, not in front
    assert [m.id for m in messages_from_events(events)] == [
        "u1",
        "a1",
        "u2",
        "a2",
        "sum",
    ]


def test_messages_from_events_inserts_unseen_between_known() -> None:
    u1 = ChatMessageUser(id="u1", content="first")
    a1 = ChatMessageAssistant(id="a1", content="reply")
    mid = ChatMessageUser(id="mid", content="injected")
    events = [model_event([u1], a1), model_event([u1, mid, a1], None)]
    assert [m.id for m in messages_from_events(events)] == ["u1", "mid", "a1"]


def model_event(
    input: list[ChatMessage],
    output: ChatMessageAssistant | None,
    error: str | None = None,
) -> ModelEvent:
    return ModelEvent(
        model="mockllm/model",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(
            model="mockllm/model",
            choices=[ChatCompletionChoice(message=output)] if output else [],
        ),
        error=error,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Projection
# ═══════════════════════════════════════════════════════════════════════════


def test_projection_role_heading_and_content() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(id="u", content="**bold** kumquat"),
        ChatMessageAssistant(
            id="a", content=[ContentText(text="one"), ContentText(text="two")]
        ),
        ChatMessageUser(id="t", content=""),
    ]
    assert texts(messages) == ["user\nbold kumquat", "assistant\none\ntwo", "user"]
    assert texts(messages, unlabeled_roles=frozenset({"user"})) == [
        "bold kumquat",
        "assistant\none\ntwo",
        "",
    ]


def test_projection_display_mode_raw_keeps_markdown_source() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(id="u", content="see [docs](https://example.test)"),
        ChatMessageAssistant(
            id="a", content=[ContentReasoning(reasoning="**deep** `x`")]
        ),
    ]
    assert texts(messages) == ["user\nsee docs", "assistant\nReasoning\ndeep x"]
    assert texts(messages, display_mode="raw") == [
        "user\nsee [docs](https://example.test)",
        "assistant\nReasoning\n**deep** `x`",
    ]


def test_projection_content_data_document_and_citation_fallbacks() -> None:
    from inspect_ai._util.citation import DocumentCitation, UrlCitation
    from inspect_ai._util.content import ContentData, ContentDocument

    message = ChatMessageAssistant(
        id="a",
        content=[
            ContentText(
                text="cited",
                citations=[
                    UrlCitation(url="https://t.example", title="Titled"),
                    UrlCitation(url="https://q.example", cited_text="quoted words"),
                    UrlCitation(url="https://u.example", cited_text=(0, 3)),
                    DocumentCitation(cited_text=(0, 3)),
                ],
            ),
            ContentData(data={"k": "ü"}),
            ContentDocument(document="base64", filename="report.pdf"),
        ],
    )
    row = message_rows([message])[0]
    assert [s.text for s in project_row(row, "a", include_chrome=False)] == [
        "cited",
        "Titled",
        "quoted words",
        "https://u.example",
        '{"k": "ü"}',
        "report.pdf",
    ]


def test_projection_standalone_tool_row() -> None:
    messages: list[ChatMessage] = [
        ChatMessageUser(id="u", content="go"),
        ChatMessageAssistant(id="a", content="calls", tool_calls=None),
        ChatMessageTool(id="t", function="bash", content="  out  "),
    ]
    rows = message_rows(messages)
    # a folded tool message without a matching tool call renders nothing
    assert [s.text for s in project_row(rows[1], "a")] == ["assistant", "calls"]
    standalone = message_rows([ChatMessageTool(id="t", function="bash", content="x")])
    assert standalone == []


def test_projection_tool_calls_plain_text_and_styles() -> None:
    call = ToolCall(
        id="c1",
        function="bash",
        arguments={"cmd": "echo hi", "timeout": 5, "opts": {"x": "ü"}},
        view=ToolCallContent(title="Run {{cmd}}", format="markdown", content="x"),
    )
    messages: list[ChatMessage] = [
        ChatMessageAssistant(id="a", content="visible", tool_calls=[call]),
        ChatMessageTool(id="t", tool_call_id="c1", function="bash", content="hi\n"),
    ]
    row = message_rows(messages)[0]
    complete = [
        "assistant",
        "visible",
        "tool: bash",
        "Run {{cmd}}",
        "bash",
        "echo hi",
        "5",
        '{"x": "ü"}',
    ]
    assert [s.text for s in project_row(row, "a")] == complete + ["hi\n"]
    compact = ProjectionOptions(tool_call_style="compact")
    assert [s.text for s in project_row(row, "a", compact)] == complete
    assert [s.text for s in project_row(row, "a", include_chrome=False)] == [
        "visible",
        "bash",
        "echo hi",
        "5",
        '{"x": "ü"}',
        "hi\n",
    ]
    omit = ProjectionOptions(tool_call_style="omit")
    assert [s.text for s in project_row(row, "a", omit)] == ["assistant", "visible"]


def test_projection_pairs_tool_messages_by_call_id_out_of_order() -> None:
    calls = [
        ToolCall(id="c1", function="f", arguments={"n": "first"}),
        ToolCall(id="c2", function="f", arguments={"n": "second"}),
    ]
    messages: list[ChatMessage] = [
        ChatMessageAssistant(id="a", content="", tool_calls=calls),
        ChatMessageTool(id="t2", tool_call_id="c2", function="f", content="two"),
        ChatMessageTool(id="t1", tool_call_id="c1", function="f", content="one"),
    ]
    row = message_rows(messages)[0]
    assert [s.text for s in project_row(row, "a", include_chrome=False)] == [
        "f",
        "first",
        "one",
        "f",
        "second",
        "two",
    ]


def test_projection_server_tool_use_context_and_entities() -> None:
    from inspect_ai._util.citation import UrlCitation
    from inspect_ai._util.content import ContentToolUse

    message = ChatMessageAssistant(
        id="a",
        content=[
            ContentToolUse(
                tool_type="mcp_call",
                id="s1",
                name="search",
                context="docs",
                arguments="q",
                result="r",
            ),
            ContentText(
                text="x", citations=[UrlCitation(url="https://e", title="A &amp; B")]
            ),
        ],
    )
    row = message_rows([message])[0]
    assert [s.text for s in project_row(row, "a", include_chrome=False)] == [
        "docs",
        "search",
        "q",
        "r",
        "x",
        "A & B",
    ]


def test_projection_tool_error_replaces_output() -> None:
    call = ToolCall(id="c1", function="python", arguments={"code": "1/0"})
    messages: list[ChatMessage] = [
        ChatMessageAssistant(id="a", content="", tool_calls=[call]),
        ChatMessageTool(
            id="t",
            tool_call_id="c1",
            function="python",
            content="ignored",
            error=ToolCallError(type="timeout", message="took too long"),
        ),
    ]
    row = message_rows(messages)[0]
    assert [s.text for s in project_row(row, "a")] == [
        "assistant",
        "tool: python",
        "python",
        "1/0",
        "took too long",
    ]


def test_projection_reasoning_title_is_chrome() -> None:
    message = ChatMessageAssistant(
        id="a",
        content=[
            ContentReasoning(reasoning="**thinking**"),
            ContentReasoning(reasoning="", summary="short"),
        ],
    )
    row = message_rows([message])[0]
    assert [s.text for s in project_row(row, "a")] == [
        "assistant",
        "Reasoning",
        "thinking",
        "Reasoning (Summary)",
        "short",
    ]
    assert [s.text for s in project_row(row, "a", include_chrome=False)] == [
        "thinking",
        "short",
    ]


def test_projection_redacted_reasoning_shows_summary_not_ciphertext() -> None:
    call = ToolCall(id="c1", function="think", arguments={})
    redacted = ContentReasoning(
        reasoning="gAAAAciphertext", summary="thought about cats", redacted=True
    )
    messages: list[ChatMessage] = [
        ChatMessageAssistant(id="a", content=[redacted], tool_calls=[call]),
        ChatMessageTool(
            id="t", tool_call_id="c1", function="think", content=[redacted]
        ),
    ]
    row = message_rows(messages)[0]
    assert [s.text for s in project_row(row, "a", include_chrome=False)] == [
        "thought about cats",
        "think",
        "thought about cats",
    ]
    unsummarized = ContentReasoning(reasoning="gAAAA", redacted=True)
    row = message_rows([ChatMessageAssistant(id="a", content=[unsummarized])])[0]
    assert [s.text for s in project_row(row, "a", include_chrome=False)] == []


def test_projection_segments_carry_anchor_and_field() -> None:
    rows = message_rows([ChatMessageUser(id="u", content="hello")])
    assert [
        (s.anchor, s.field, s.text) for s in projection(rows, row_anchors(rows))
    ] == [
        ("u", "role", "user"),
        ("u", "content", "hello"),
    ]


# scout's `event_as_str(event)` for the same events (inspect_scout
# _grep_scanner/_event.py), copied as literals; the data projection is that
# text minus scout's labels (`MODEL:`, `TOOL (fn):`, `Arguments:`, `Result:`,
# `Error:`, `Message:`, `Tool:`, `Args:`, `Explanation:`, `k: `/`k=` arg keys).
SCOUT_EVENT_STRINGS = [
    "MODEL:\nthe completion\n",
    "TOOL (bash):\nArguments:\n  cmd: ls\n  timeout: 5\nResult: a\nb\n",
    "TOOL (python):\nArguments:\n  code: 1/0\nResult: \nError: boom\n",
    "ERROR:\nsample failed\n",
    'INFO (scorer):\n{"note": "ok"}\n',
    "LOG (warning):\nslow tool\n",
    "APPROVAL (reject):\nMessage: do it?\nTool: rm\nArgs: path=/\nExplanation: no\n",
]


def scout_events() -> list[Event]:
    from inspect_ai._util.error import EvalError
    from inspect_ai.event import (
        ApprovalEvent,
        ErrorEvent,
        InfoEvent,
        LoggerEvent,
        LoggingMessage,
        ToolEvent,
    )

    return [
        model_event([], ChatMessageAssistant(id="a", content="the completion")),
        ToolEvent(
            id="c1",
            function="bash",
            arguments={"cmd": "ls", "timeout": 5},
            result="a\nb",
        ),
        ToolEvent(
            id="c2",
            function="python",
            arguments={"code": "1/0"},
            result="",
            error=ToolCallError(type="unknown", message="boom"),
        ),
        ErrorEvent(
            error=EvalError(message="sample failed", traceback="", traceback_ansi="")
        ),
        InfoEvent(source="scorer", data={"note": "ok"}),
        LoggerEvent(
            message=LoggingMessage(level="warning", message="slow tool", created=0.0)
        ),
        ApprovalEvent(
            message="do it?",
            call=ToolCall(id="c3", function="rm", arguments={"path": "/"}),
            approver="human",
            decision="reject",
            explanation="no",
        ),
    ]


def test_project_event_data_matches_scout_event_as_str() -> None:
    data = [
        "\n".join(s.text for s in project_event(event, include_chrome=False))
        for event in scout_events()
    ]
    assert data == [
        "the completion",
        "ls\n5\na\nb",
        "1/0\nboom",
        "sample failed",
        '{"note": "ok"}',
        "slow tool",
        "do it?\nrm\n/\nno",
    ]
    for segments, scout in zip(data, SCOUT_EVENT_STRINGS, strict=True):
        for text in segments.split("\n"):
            assert text in scout


def test_project_event_chrome_and_unsupported() -> None:
    events = scout_events()
    assert [s.text for s in project_event(events[1])] == ["bash", "ls", "5", "a\nb"]
    assert [s.text for s in project_event(events[5])] == ["warning", "slow tool"]
    from inspect_ai.event import StepEvent

    assert project_event(StepEvent(action="begin", name="x")) == []


def test_project_event_tool_result_document_filename() -> None:
    from inspect_ai._util.content import ContentDocument
    from inspect_ai.event import ToolEvent

    event = ToolEvent(
        id="c1",
        function="read",
        arguments={},
        result=[
            ContentText(text="body"),
            ContentDocument(document="", filename="f.pdf"),
        ],
    )
    assert [s.text for s in project_event(event, include_chrome=False)] == [
        "body",
        "f.pdf",
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Markdown stripping for counts
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "name,source,expected",
    [
        (
            "link",
            "see [istanbul docs](https://istanbul.example) now",
            "see istanbul docs now",
        ),
        ("strong", "**bold** and __also__", "bold and also"),
        (
            "emphasis",
            "*slanted* and _italic_ but snake_case_name stays",
            "slanted and italic but snake_case_name stays",
        ),
        ("strikethrough", "~~gone~~ text", "gone text"),
        ("unpaired_markers", "2 * 3 and a_b and **", "2 * 3 and a_b and **"),
        ("heading", "## Overview\n### Sub", "Overview\nSub"),
        ("blockquote", "> quoted\n> more", "quoted\nmore"),
        (
            "list_markers",
            "- one\n* two\n+ three\n1. four\n  - nested",
            "one\ntwo\nthree\nfour\nnested",
        ),
        ("fence_lines", "```python\nx = 1\n```\n~~~\ny\n~~~", "x = 1\n\ny\n"),
        ("task_list", "- [ ] todo\n- [x] done\n1. [X] third", "todo\ndone\nthird"),
        ("indented_code", "    x = 1\n\tkept", "    x = 1\n\tkept"),
        (
            "indented_code_block_kept",
            "para\n\n    - literal\n    **x**\n- item",
            "para\n\n    - literal\n    **x**\nitem",
        ),
        (
            "intraword_double_underscore",
            "foo__bar__baz and __strong__",
            "foo__bar__baz and strong",
        ),
        ("code_span_run_lengths", "`a``b` and ``c`d``", "a``b and c`d"),
        (
            "code_span_content_kept_without_backticks",
            "use `**not bold**` and ``[no link](x)`` here",
            "use **not bold** and [no link](x) here",
        ),
        (
            "nul_bytes_are_text",
            "\x000\x00 and `c` \x001\x00",
            "\x000\x00 and c \x001\x00",
        ),
        ("private_use_is_text", "\ue0000\ue000 `c`", "\ue0000\ue000 c"),
        (
            "fenced_code_kept",
            "```\n# not a heading\n- not a bullet\n| a | b |\n```\n- bullet",
            "# not a heading\n- not a bullet\n| a | b |\n\nbullet",
        ),
    ],
)
def test_strip_markdown_rule(name: str, source: str, expected: str) -> None:
    assert strip_markdown_for_count(source) == expected


def test_strip_markdown_istanbul_link_case() -> None:
    sample = inspect_ai.log.read_eval_log_sample(str(QA_LOG), "find-qa", 1)
    answer = next(m.text for m in sample.messages if "## Overview" in m.text)
    stripped = strip_markdown_for_count(answer)
    # the link URL no longer counts
    assert len(find_matches(answer, "istanbul")) == 7
    assert len(find_matches(stripped, "istanbul")) == 6
    assert "https://example.com/istanbul" not in stripped


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint
# ═══════════════════════════════════════════════════════════════════════════

PROJECTION = {"unlabeled_roles": [], "tool_call_style": "complete"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(fastapi_server.view_server_app(default_dir=str(tmp_path)))


def find(
    client: TestClient,
    text: str,
    log: Path = QA_LOG,
    sample_id: str = "find-qa",
    **body: Any,
) -> dict[str, Any]:
    request = {
        "sample_id": sample_id,
        "epoch": 1,
        "text": text,
        "direction": "forward",
        "cursor": None,
        "limit": 10,
        "projection": PROJECTION,
        **body,
    }
    response = client.post(f"/find-messages/{log}", json=request)
    assert response.status_code == 200, response.text
    return response.json()


# find-qa.eval: one mockllm sample "find-qa" whose scripted messages carry the
# fold cases (İstanbul/ISTANBUL, Straße/STRASSE, café NFC+NFD, 検索), a markdown
# answer with a link whose URL repeats "istanbul", a bash call whose 40 output
# lines each hold LONGTOKEN, and "kumquat" spread over user, assistant and tool
# text. Counts are hand-derived over the source text with markdown syntax
# stripped (so the link URL does not count); tool arguments and every tool
# output line count.
@pytest.mark.parametrize(
    "text,rows,occurrences",
    [
        ("kumquat", 5, 8),
        ("istanbul", 2, 7),
        ("İstanbul", 2, 7),
        ("ISTANBUL", 2, 7),
        ("ıstanbul", 0, 0),
        ("strasse", 2, 4),
        ("straße", 2, 4),
        ("café", 1, 2),
        ("検索", 3, 6),
        ("LONGTOKEN", 3, 43),
        ("assistant", 20, 20),
        ("", 0, 0),
    ],
)
def test_endpoint_totals(
    client: TestClient, text: str, rows: int, occurrences: int
) -> None:
    result = find(client, text, limit=1000)
    assert result["total"] == {
        "rows": rows,
        "occurrences": occurrences,
        "relation": "eq",
    }
    assert result["complete"] is True
    assert len(result["rows"]) == rows
    assert sum(row["count"] for row in result["rows"]) == occurrences


def test_endpoint_row_texts_are_distinct_source_substrings(client: TestClient) -> None:
    rows = find(client, "strasse")["rows"]
    assert [(r["count"], r["texts"]) for r in rows] == [
        (3, ["Straße", "straße", "STRASSE"]),
        (1, ["straße"]),
    ]
    assert rows[0]["anchor"] != rows[1]["anchor"]
    # row positions (tool messages fold into their calling row)
    assert [r["index"] for r in rows] == [2, 12]
    long_row = find(client, "LONGTOKEN")["rows"][1]
    assert (long_row["index"], long_row["count"], long_row["texts"]) == (
        7,
        41,
        ["LONGTOKEN"],
    )


def test_endpoint_paging_forward_and_backward(client: TestClient) -> None:
    everything = find(client, "kumquat", limit=100)["rows"]
    assert len(everything) == 5

    forward: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = find(client, "kumquat", limit=2, cursor=cursor)["rows"]
        if not page:
            break
        forward.extend(page)
        cursor = {"anchor": page[-1]["anchor"]}
    assert forward == everything

    backward: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = find(client, "kumquat", limit=2, cursor=cursor, direction="backward")[
            "rows"
        ]
        if not page:
            break
        backward.extend(page)
        cursor = {"anchor": page[-1]["anchor"]}
    assert backward == everything[::-1]


def test_endpoint_page_total_is_this_page_until_the_scan_ends(
    client: TestClient, tmp_path: Path
) -> None:
    log = tmp_path / "hits.eval"
    write_sample_log(
        log,
        [ChatMessageUser(id=f"u{i}", content="hit") for i in range(5)],
    )
    first = find(client, "hit", log=log, sample_id="s", limit=2)
    assert first["total"] == {"rows": 2, "occurrences": 2, "relation": "gte"}
    assert first["complete"] is True
    mid = find(
        client,
        "hit",
        log=log,
        sample_id="s",
        limit=2,
        cursor={"anchor": first["rows"][-1]["anchor"]},
    )
    assert mid["total"] == {"rows": 2, "occurrences": 2, "relation": "gte"}
    last = find(
        client,
        "hit",
        log=log,
        sample_id="s",
        limit=2,
        cursor={"anchor": mid["rows"][-1]["anchor"]},
    )
    assert last["total"] == {"rows": 1, "occurrences": 1, "relation": "eq"}


def test_endpoint_scan_budget_stops_the_page_after_the_first_match(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspect_ai._view.find._messages import _SCAN_BUDGET_S

    ticks = {"n": 0}

    def now() -> float:
        ticks["n"] += 1
        return ticks["n"] * _SCAN_BUDGET_S

    monkeypatch.setattr("inspect_ai._view.find._messages.time.perf_counter", now)
    log = tmp_path / "slow.eval"
    write_sample_log(
        log,
        [ChatMessageUser(id=f"u{i}", content="hit") for i in range(5)],
    )
    page = find(client, "hit", log=log, sample_id="s", limit=10)
    assert page["total"]["rows"] == 1
    assert page["total"]["relation"] == "gte"
    monkeypatch.undo()
    rest = find(
        client,
        "hit",
        log=log,
        sample_id="s",
        limit=10,
        cursor={"anchor": page["rows"][-1]["anchor"]},
    )
    assert rest["total"]["rows"] == 4
    assert rest["total"]["relation"] == "eq"


def test_endpoint_scan_budget_continues_after_a_slow_first_match(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from inspect_ai._util.textsearch import FoldedText
    from inspect_ai._view.find._messages import _SampleIndex

    clock = {"t": 0.0}
    monkeypatch.setattr(
        "inspect_ai._view.find._messages.time",
        SimpleNamespace(perf_counter=lambda: clock["t"]),
    )
    original = _SampleIndex.folded_row

    def fold(self: _SampleIndex, i: int, options: ProjectionOptions) -> FoldedText:
        if clock["t"] == 0.0:
            clock["t"] = 1.0
        return original(self, i, options)

    monkeypatch.setattr(_SampleIndex, "folded_row", fold)
    log = tmp_path / "slow.eval"
    write_sample_log(
        log,
        [ChatMessageUser(id=f"u{i}", content="hit") for i in range(5)],
    )
    page = find(client, "hit", log=log, sample_id="s", limit=10)
    # first fold "took" 1s; the 50ms budget starts after that hit, so the
    # rest of this small sample still fits on the first page
    assert page["total"]["rows"] == 5
    assert page["total"]["relation"] == "eq"


def test_endpoint_finds_compact_tool_chrome(client: TestClient, tmp_path: Path) -> None:
    log = tmp_path / "tool.eval"
    write_sample_log(
        log,
        [
            ChatMessageAssistant(
                id="a",
                content="",
                tool_calls=[
                    ToolCall(id="c1", function="bash", arguments={"cmd": "ls"})
                ],
            ),
        ],
    )
    compact = {**PROJECTION, "tool_call_style": "compact"}
    page = find(client, "tool: bash", log=log, sample_id="s", projection=compact)
    assert page["total"] == {"rows": 1, "occurrences": 1, "relation": "eq"}


def test_endpoint_miss_yields_the_event_loop(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    yields = {"n": 0}
    original = asyncio.sleep

    async def sleep(delay: float, result: object = None) -> object:
        yields["n"] += 1
        return await original(delay, result)

    monkeypatch.setattr("inspect_ai._view.find._messages.asyncio.sleep", sleep)
    log = tmp_path / "miss.eval"
    write_sample_log(
        log,
        [ChatMessageUser(id=f"u{i}", content="aaa") for i in range(40)],
    )
    page = find(client, "zzz", log=log, sample_id="s", limit=10)
    assert page["total"] == {"rows": 0, "occurrences": 0, "relation": "eq"}
    assert yields["n"] >= 1


def test_endpoint_unknown_cursor_restarts_at_near_edge(client: TestClient) -> None:
    everything = find(client, "kumquat", limit=100)["rows"]
    stale = {"anchor": "gone"}
    assert find(client, "kumquat", limit=2, cursor=stale)["rows"] == everything[:2]
    assert (
        find(client, "kumquat", limit=2, cursor=stale, direction="backward")["rows"]
        == everything[-1:-3:-1]
    )


def test_segments_do_not_glue_across_boundaries() -> None:
    from inspect_ai._view.find._messages import _SampleIndex

    index = _SampleIndex([ChatMessageAssistant(id="a", content="ant hill")], True)
    joined = index.folded_rows(ProjectionOptions())[0].text
    assert joined == "assistant\nant hill"
    assert len(find_matches(joined, "assistantant")) == 0


def test_endpoint_texts_uncapped_and_limit_over_1000_rejected(
    client: TestClient, tmp_path: Path
) -> None:
    variants = " ".join(f"{'X' * i}{'x' * (40 - i)}" for i in range(40))
    log = tmp_path / "variants.eval"
    write_sample_log(log, [ChatMessageUser(id="u", content=variants)])
    row = find(client, "x" * 40, log=log, sample_id="s")["rows"][0]
    assert row["count"] == 40 and len(row["texts"]) == 40
    assert row["texts"][0] == "x" * 40 and row["texts"][39] == "X" * 39 + "x"
    body = dict(
        sample_id="s",
        epoch=1,
        text="x",
        direction="forward",
        limit=1001,
        projection=PROJECTION,
    )
    assert client.post(f"/find-messages/{log}", json=body).status_code == 422


def test_endpoint_chunked_sample_is_searched_whole(
    client: TestClient, tmp_path: Path
) -> None:
    from inspect_ai.log._recorders.chunked import convert_eval_logs_to_chunked

    # the chunked per-sample shape (samples/{id}_epoch_{epoch}/messages/0.json);
    # the long turn is extracted to the attachments sequence by the converter
    source = tmp_path / "source.eval"
    write_sample_log(
        source,
        [
            ChatMessageUser(id="u", content="kumquat question"),
            ChatMessageAssistant(id="a", content="filler " * 2000 + "kumquat answer"),
            ChatMessageUser(id="u2", content="kumquat again, in the second chunk"),
        ],
    )
    convert_eval_logs_to_chunked(str(source), str(tmp_path / "chunked"), chunk_size=2)
    log = tmp_path / "chunked" / "source.eval"
    result = find(client, "kumquat", log=log, sample_id="s")
    assert result["total"] == {"rows": 3, "occurrences": 3, "relation": "eq"}
    assert result["complete"] is True
    assert [(r["index"], r["anchor"]) for r in result["rows"]] == [
        (0, "u"),
        (1, "a"),
        (2, "u2"),
    ]


def write_sample_log(path: Path, messages: list[ChatMessage]) -> None:
    from inspect_ai.log import EvalSample

    log = inspect_ai.log.EvalLog(
        status="success",
        eval=inspect_ai.log.EvalSpec(
            created="2025-01-01T00:00:00Z",
            task="task",
            task_id="task_id",
            dataset=inspect_ai.log.EvalDataset(),
            model="model",
            config=inspect_ai.log.EvalConfig(),
        ),
        samples=[EvalSample(id="s", epoch=1, input="q", target="", messages=messages)],
    )
    inspect_ai.log.write_eval_log(log, str(path), "eval")


def test_endpoint_projection_options(client: TestClient) -> None:
    hidden = {**PROJECTION, "unlabeled_roles": ["assistant"]}
    assert find(client, "assistant", projection=hidden)["total"]["rows"] == 0
    # the 40 output lines are hidden in compact mode; the user turn, the
    # compact call line and the final answer remain
    compact = {**PROJECTION, "tool_call_style": "compact"}
    assert find(client, "LONGTOKEN", projection=compact)["total"] == {
        "rows": 3,
        "occurrences": 3,
        "relation": "eq",
    }


def test_endpoint_projection_defaults_and_raw_display_mode(
    client: TestClient,
) -> None:
    # a host echoes only what it changes; omitted projection = viewer defaults
    body = {
        "sample_id": "find-qa",
        "epoch": 1,
        "text": "istanbul",
        "direction": "forward",
        "limit": 10,
    }
    response = client.post(f"/find-messages/{QA_LOG}", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["total"]["occurrences"] == 7
    # raw mode shows markdown source, so the link URL counts too
    raw = find(client, "istanbul", projection={"display_mode": "raw"})
    assert raw["total"]["occurrences"] == 8


def test_endpoint_unknown_sample_is_404(client: TestClient) -> None:
    body: dict[str, Any] = dict(
        sample_id="nope",
        epoch=1,
        text="x",
        direction="forward",
        limit=1,
        projection=PROJECTION,
    )
    assert client.post(f"/find-messages/{QA_LOG}", json=body).status_code == 404
    body["sample_id"] = "find-qa"
    assert (
        client.post(
            f"/find-messages/{QA_LOG.parent / 'missing.eval'}", json=body
        ).status_code
        == 404
    )


def test_endpoint_running_sample_from_buffer(
    client: TestClient, tmp_path: Path
) -> None:
    from inspect_ai.log._condense import ATTACHMENT_PROTOCOL
    from inspect_ai.log._recorders.buffer.filestore import (
        Manifest,
        SampleBufferFilestore,
        SampleManifest,
        Segment,
        SegmentFile,
    )
    from inspect_ai.log._recorders.buffer.types import (
        AttachmentData,
        EventData,
        MessagePoolData,
        SampleData,
    )

    log_path = str(tmp_path / "running.eval")
    inspect_ai.log.write_eval_log(
        inspect_ai.log.EvalLog(
            status="started",
            eval=inspect_ai.log.EvalSpec(
                created="2025-01-01T00:00:00Z",
                task="task",
                task_id="task_id",
                dataset=inspect_ai.log.EvalDataset(),
                model="model",
                config=inspect_ai.log.EvalConfig(),
            ),
        ),
        log_path,
        "eval",
    )
    # the user turn is pooled (the event carries only input_refs) and its text
    # is an attachment, as the recorder writes long content
    u1 = ChatMessageUser(id="u1", content=f"{ATTACHMENT_PROTOCOL}h1")
    a1 = ChatMessageAssistant(id="a1", content="kumquat answer")
    first = model_event([], a1).model_copy(update={"input_refs": [(0, 1)]})
    events = [first, model_event([u1, a1], None)]
    pool = [
        MessagePoolData(
            id=1, sample_id="live", epoch=1, msg_id="u1", data=u1.model_dump_json()
        )
    ]
    attachments = [
        AttachmentData(
            id=1, sample_id="live", epoch=1, hash="h1", content="kumquat question"
        )
    ]
    buffer = SampleBufferFilestore(log_path, create=True)
    buffer.write_segment(
        0,
        [
            SegmentFile(
                id="live",
                epoch=1,
                data=SampleData(
                    events=[
                        EventData(
                            id=i + 1,
                            event_id=f"e{i}",
                            sample_id="live",
                            epoch=1,
                            event=event.model_dump(mode="json", exclude_none=True),
                        )
                        for i, event in enumerate(events)
                    ],
                    attachments=attachments,
                    message_pool=pool,
                ),
            )
        ],
    )
    buffer.write_manifest(
        Manifest(
            samples=[
                SampleManifest(
                    summary=inspect_ai.log.EvalSampleSummary(
                        id="live", epoch=1, input="q", target=""
                    ),
                    segments=[0],
                )
            ],
            segments=[Segment(id=0, last_event_id=2, last_attachment_id=1)],
        )
    )

    result = find(client, "kumquat", log=Path(log_path), sample_id="live")
    assert result["total"] == {"rows": 2, "occurrences": 2, "relation": "gte"}
    assert result["complete"] is False
    assert [(r["index"], r["anchor"], r["texts"]) for r in result["rows"]] == [
        (0, "u1", ["kumquat"]),
        (1, "a1", ["kumquat"]),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════════════════════════


def test_sample_index_key_distinguishes_id_types_and_canonicalizes_roles() -> None:
    from inspect_ai._view.find._messages import _index_key, _SampleIndex

    assert _index_key("log", 1, 1) != _index_key("log", "1", 1)
    index = _SampleIndex([ChatMessageUser(id="u", content="hi")], complete=True)
    index.folded_rows(ProjectionOptions(unlabeled_roles=frozenset({"user", "tool"})))
    index.folded_rows(ProjectionOptions(unlabeled_roles=frozenset({"user", "system"})))
    assert len(index._folded) == 1


async def test_sample_index_cache_isolates_location_epoch_and_evicts_lru(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_ai._view.find import _messages

    reads: list[tuple[str, str | int, int]] = []

    async def logged(
        location: str, sample_id: str | int, epoch: int
    ) -> list[ChatMessage] | None:
        reads.append((location, sample_id, epoch))
        return [ChatMessageUser(id="u", content=location)]

    monkeypatch.setattr(_messages, "_logged_messages", logged)
    _messages._cache.clear()
    a = await _messages._sample_index("a.eval", "s", 1)
    assert await _messages._sample_index("a.eval", "s", 1) is a
    await _messages._sample_index("b.eval", "s", 1)
    await _messages._sample_index("a.eval", "s", 2)
    assert reads == [("a.eval", "s", 1), ("b.eval", "s", 1), ("a.eval", "s", 2)]
    # touching a keeps it recent; filling the cache evicts the oldest (b) first
    await _messages._sample_index("a.eval", "s", 1)
    for i in range(_messages._MAX_CACHED_SAMPLES - 2):
        await _messages._sample_index(f"fill{i}.eval", "s", 1)
    reads.clear()
    assert await _messages._sample_index("a.eval", "s", 1) is a
    await _messages._sample_index("b.eval", "s", 1)
    assert reads == [("b.eval", "s", 1)]


def test_sample_index_folded_variants_are_bounded() -> None:
    from inspect_ai._view.find._messages import _MAX_FOLDED_VARIANTS, _SampleIndex

    index = _SampleIndex([ChatMessageUser(id="u", content="hi")], complete=True)
    variants = [
        ProjectionOptions(frozenset(roles), style, mode)
        for roles in (set[str](), {"user"})
        for style in ("complete", "compact", "omit")
        for mode in ("rendered", "raw")
    ]
    for options in variants:
        index.folded_rows(options)
    assert len(index._folded) == _MAX_FOLDED_VARIANTS
    assert list(index._folded) == variants[-_MAX_FOLDED_VARIANTS:]


async def test_sample_index_reprobes_log_after_buffer_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspect_ai._view.find import _messages

    probes: list[str] = []

    async def logged(
        location: str, sample_id: str | int, epoch: int
    ) -> list[ChatMessage] | None:
        probes.append("log")
        return [ChatMessageUser(id="u", content="sealed")] if len(probes) > 2 else None

    def running(
        location: str, sample_id: str | int, epoch: int
    ) -> list[ChatMessage] | None:
        probes.append("buffer")
        return None

    monkeypatch.setattr(_messages, "_logged_messages", logged)
    monkeypatch.setattr(_messages, "_running_messages", running)
    _messages._cache.clear()
    index = await _messages._sample_index("race.eval", "s", 1)
    assert probes == ["log", "buffer", "log"]
    assert index is not None and index.complete
