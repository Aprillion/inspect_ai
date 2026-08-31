# Find on the Messages tab (server side)

Cmd+F on the Messages tab is answered by the view server; client side: ts-mono `design/pluggable-find.md`.

## Principles

- The server decides which rows (or events) match and roughly how often; the DOM decides where.
  `count` is an estimate the client corrects, `texts` are substrings of the row's projection under the
  requested display mode (the DOM contains them; under `rendered` the log source may not, e.g. `foo bar`
  from `**foo** bar`) which it highlights literally.
- Source text, not rendering: rows are projected from the log as written; the viewer's formatting is
  never reproduced.
- Chrome is text the viewer adds that users read (role heading, `Reasoning`, tool label). It is
  flag-controlled (`include_chrome`) so grep-style search uses data only.
- The fold is the browser standard (NFKD, marks dropped, casefold); grep's `ignore_case` is casefold.
- Matches are reported as projection substrings (offsets mapped back through the fold), never as folded text.

## Endpoint

`POST /find-messages/{log}` with
`{sample_id, epoch, text, direction, cursor?: {anchor}, limit, projection?: {unlabeled_roles?, tool_call_style?,
display_mode?}}` returns `{rows: [{anchor, index, count, texts}], total: {rows, occurrences, relation}, complete}`.

- `rows`: matching rows in travel order (backward = nearest first), strictly after/before the cursor
  row; a vanished cursor anchor restarts from the near edge; `limit` (1–1000) caps rows, larger values
  are rejected (422), not clamped. `projection` defaults to the viewer's defaults (`unlabeled_roles=[]`,
  `tool_call_style="complete"`, `display_mode="rendered"`), so a host echoes only what it changed.
- `count`: non-overlapping folded matches in the row's source projection — an estimate of the DOM,
  which the client corrects. `texts`: all distinct projection substrings matched, in first-appearance
  order; the client highlights them literally.
- `total` is **this page** (`rows`/`occurrences` = this response). `relation` is `"eq"` only when
  this request walked off the end of the source in `direction` and the sample is sealed; otherwise
  `"gte"` and the client sums pages (the band is M+ until then). A page stops at `limit` matching
  rows or ~50ms after the first match, so the first hits can paint while the rest of the scan
  continues. `complete` is whether the sample is sealed (a running buffer is never `complete`).
  Empty `text` → empty result (the user is typing). Models: `_view/find/_messages.py`.

## Projection (`inspect_ai._view.find`)

Rows are the viewer's fold (`message_rows`): each non-tool message plus the tool messages following
it; system messages merge into one first row; id-less messages get `msg-{index}`. Anchors are message
ids (`""` for an empty id) unless a prior row holds that anchor, then `id#row` with `#row` repeated while
taken; only prior rows are consulted, so anchors never change on live append. `index` = row position.

`project_row(row, anchor, options, include_chrome=True) -> [Segment{anchor, field, text}]` emits
the row's data text in render order: role heading (`role` / `tool: fn`, unless in `unlabeled_roles`),
message text blocks, `Reasoning` + reasoning text (the summary when redacted or empty), citation title (else quoted text,
else URL), data blocks as JSON, document filenames, server tool name / arguments / result or error; per tool call the function name, each argument (strings as is, else
`json.dumps`), then the tool message's output text or error message. `tool_call_style`: `omit` → no
tool text; `compact` → no tool output; `complete` → all. Segments join with `"\n"`, so a heading never
glues onto the body. Chrome is the text the viewer adds (role heading, `Reasoning`);
`include_chrome=False` gives data-only segments for grep.

Message and reasoning text pass through `strip_markdown_for_count` (skipped under `display_mode`
`raw`, where the viewer shows markdown source): fenced, indented and inline code content kept verbatim, fence
lines and backticks dropped; `__` is literal intraword (CommonMark); `[text](url)` → text; heading, quote, list and task markers and paired `**` `__` `*` `_` `~~`
removed. Only syntax goes, so every substring of the result also occurs in the rendered text; the
point is that a phrase spanning a marker (`some **bold**`) matches "some bold" as in the browser.
Tool args and outputs are counted verbatim.

`project_event(event, include_chrome=True) -> [Segment]` is the same shape for transcript events
(model, tool, error, info, logger, approval; others → nothing): chrome is the event's label (model
name, tool function, `Error`, info source, log level, approval decision), data is the completion,
argument values, result or error message, info data, log message, approval message / function /
args / explanation.

## Matching (`inspect_ai._util.textsearch`)

`find_matches(text, query, *, mode: literal|regex, fold: browser|case|none, word_boundary) ->
[TextMatch{start, end, text}]`; `FoldedText(text, fold)` folds once with an offset map and
`find_all(compile_query(...))` is a regex scan over the folded copy with `finditer` semantics (empty
hits skipped, a non-empty alternative at the same spot still counts); `word_boundary` wraps the whole
pattern (`\b(?:…)\b`). Find uses literal + browser;
grep uses literal or regex + case (+ word boundary).

`browser` fold per code point: NFKD → drop category M* (`\p{M}`) → `str.casefold()`; parity with
Chrome's find-in-page on everything probed: İ/i, ß/ss, é/e (NFC and NFD), ﬁ/fi; dotless ı distinct.
`case` is `str.casefold()` alone; `none` is exact. Matches start and end on source code-point
boundaries (`sa` ≠ `ßa`) and trailing combining marks belong to the match, so `texts` are what a
browser highlights. A literal query is folded and matched exactly; a regex is not folded (that would
rewrite `\S`), runs with `re.IGNORECASE` under a fold (whose Unicode rules also let `i` match dotless `ı` — a
regex-mode caveat), and must spell folded forms itself (`strasse`, not `straße`). Folding changes length (ß → ss), hence the offset map. 3 ms warm for 5k
hits in a 375k-char sample; the known scaling limit (no cancellation-aware scan).

## Sources and caching

Log first (`read_eval_log_sample_async`, attachments `core`; a sample in the chunked per-sample shape,
which that reader does not handle, is reassembled from its `message_refs`, `messages/` and
`attachments/` sequences); else the recorder's sample buffer via
`get_sample_data` (the live tab's poll, sync on the loop like `/pending-sample-data`) with the
client's `messagesFromEvents` reconstruction; else the log once more (it may have sealed between
probes). Completed samples are immutable, so their folded rows are cached (last 8, keyed by log path and id
type and value, per projection options restricted to roles present); buffer samples never are. The
key carries no mtime/size: a log rewritten at the same path serves stale rows until eviction, which is
accepted over stat-ing a possibly remote file on every keystroke.

## Why not

- No render parity with the viewer (tool views, arg summaries, citation numbers, notices): the server only says which rows match, `count` is an estimate the client corrects, and `texts` are projection substrings the DOM contains anyway; porting the renderer would be code that rots for no user-visible gain.
- No occurrences or offsets on the wire: positions exist only in the DOM; the server ships variants, the client finds them literally.
- Fold in Python, not JS: `RegExp` `iu` is simple case folding only (no ß/ss, İ/i, accents), and hawk's server implements the same contract.
- Inspect_ai `gte` is a **page** that has not walked off a sealed sample yet
  (limit or the ~50ms budget), not a refusal to count. Exact M is the client's
  sum once a page reports `eq`.
- No running-delta correction of M from the **DOM**: in-row N uses DOM counts;
  M stays the summed server pages.
- Anchor-only cursor: anchors are unique by construction (duplicate rule), so a row is a position.
- No `texts` cap: a folded term has few distinct spellings, and a cap would hide real variants.

## Scout adoption

Scout's `grep_scanner` would import `inspect_ai._util.textsearch.find_matches` (mode `regex` when
`regex=True`, fold `case` when `ignore_case`, `word_boundary` passed through; `re.error` maps to
`PatternError`) and `inspect_ai._view.find.project_event(event, include_chrome=False)` in place of
`_event.py::event_as_str`. The parity test to carry over: for the event fixture in
`tests/_view/test_find_messages.py::test_project_event_data_matches_scout_event_as_str`, every data
segment is a substring of scout's current `event_as_str` — the only delta is scout's labels
(`TOOL (fn):`, `Arguments:`, `k: `, …), which are grep formatting, not searchable data. That switch
is a separate scout PR.
