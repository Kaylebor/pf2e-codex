# Private PDF vision evaluation

Status: exploratory; none of the newly evaluated OCR/VLM roles is part of the
corpus pipeline. PP-DocLayoutV3 remains the existing structural model.

This note records content-free results from manual tests against privately owned,
born-digital Paizo PDFs. It deliberately contains no page images, extracted rule
text, watermark values, local paths, or review-database data.

## Non-negotiable boundary

The PDF native word layer and its opaque anchors remain authoritative. Vision and
OCR models may provide structural evidence: region type, reading order, table
shape, or a choice among bounded repair candidates. Their generated text must not
replace native words, create anchors, or silently repair corpus content.

The current production path remains PP-DocLayoutV3 for boxes and order, followed
by deterministic binding to native anchors. The experiments below investigate a
possible secondary review lane only.

## Test environment and safety finding

The local tests used llama.cpp build 10666 (`353f662`) on a Radeon RX 7900 XTX
with 24 GiB VRAM. llama.cpp also exposed the Ryzen 7 7800X3D integrated device,
so local evaluation must select the discrete GPU explicitly.

A 27B Q5 model consumed approximately 23.6 of 24 GiB and caused AMDGPU command
submission failures followed by loss of desktop graphics contexts. Q5 is unsafe
for this machine's interactive session and must not be retried. The tested 27B
Q4 profile used approximately 18.9 GB after loading and 20.1 GB during inference.

Every subsequent local round must obey these gates:

- Run exactly one model server and one request at a time. Do not overlap an OCR
  model, a general VLM, Paddle/Transformers, or another llama.cpp instance.
- Stop or verify inactive any user model service before starting a manual server.
- Use `--split-mode none --main-gpu 0`; never allow automatic splitting onto the
  integrated device.
- Keep at least 5 GiB VRAM free during inference. Abort the request if it crosses
  that floor.
- Use the Q4 quantization for the 27B Qwen test. Do not load its Q5 quantization.
- Delete rendered private crops after recording content-free measurements.
- Do not run a batch unattended until one representative request has stayed
  within the VRAM envelope.

The safe Qwen launch profile was:

```text
--ctx-size 8192
--parallel 1
--cache-type-k q8_0
--cache-type-v q8_0
--fit on
--fit-target 8192
--fit-ctx 8192
--split-mode none
--main-gpu 0
--batch-size 2048
--ubatch-size 512
--image-min-tokens 1024
--image-max-tokens 4096
--mtmd-batch-max-tokens 1024
--cache-ram 0
--no-cache-prompt
--flash-attn auto
--no-kv-unified
--no-warmup
```

Do not combine that with an explicit `--n-gpu-layers 99`: an explicit layer
count prevents the fit mechanism from adjusting an otherwise unset count. The
machine-wide llama.cpp preset has not been changed by this evaluation.

## Findings

### Full-page general-VLM tasks

Qwen3.8-27B Q4 reliably identified the tested page's broad two-column layout,
table dimensions, column-major reading order, and a continuation from the
bottom of the left column to the top of the right column.

It was not reliable as a geometry generator:

- A compact non-thinking response found the broad structure but collapsed the
  page to three regions and omitted the table region from its region list.
- A detailed per-heading request exhausted a 2,048-token response budget and
  returned truncated JSON.
- A focused reasoning request completed in 943 tokens, but supplied image-pixel
  coordinates despite a normalized 0-1000 contract. Grammar clipping then made
  several boxes invalid.

Therefore Qwen may choose among supervisor-generated candidate IDs, but it must
not invent coordinates, transcribe rule text, or enumerate unconstrained page
regions. Context length and response length are separate controls: the safe
8,192-token server context does not justify a 256-token response cap. Focused
next-round tasks allow up to 4,096 response tokens and reject truncation.

Qwen's thinking settings are model-specific. Its model card documents
`reasoning_effort` levels `xhigh`, `medium`, and `low`, plus different sampling
guidance for thinking and non-thinking modes. The locally tested per-request
`reasoning_budget` field was ignored, while the llama.cpp startup option worked;
that budget is a backend intervention rather than a Qwen-native level.

For the next comparison, use Qwen's documented profiles rather than a shared
"LLM default": thinking uses temperature 1.0, `top_p=0.95`, `top_k=20`,
`min_p=0`, presence penalty 0, and repetition penalty 1.0; non-thinking uses
temperature 0.7, `top_p=0.8`, `top_k=20`, `min_p=0`, presence penalty 1.5,
and repetition penalty 1.0. Record how the local llama.cpp chat template
actually enables or disables thinking. For the thinking trials, request Qwen's
documented `xhigh` level. For non-thinking trials, use Qwen's documented
`chat_template_kwargs.enable_thinking=false`. Although
`reasoning_effort=none` disabled thinking in the tested llama.cpp build, `none`
is not a Qwen-documented reasoning level and is not the portable test contract.

### OCR-specialist tasks

Whole-page generic layout questions were not valid tests for PaddleOCR-VL or
GLM-OCR. Both projects use a layout detector to crop regions, then ask their
recognizer a small task-specific question. Their recognizers are not general
page-layout reasoners.

On one isolated 580x185 table crop, compared only with the PDF's native text:

| Model | Native prompt | Structural output | Normalized text agreement | Runtime |
| --- | --- | --- | ---: | ---: |
| PaddleOCR-VL 1.6 | `Table Recognition` | OTSL, 5x3 table recovered | 0.946 | about 0.39 s |
| GLM-OCR | `Table Recognition:` | HTML, 5 rows / 20 cells | 0.781 | about 0.53 s |

This is a single-crop observation, not a benchmark. It establishes that a
specialist crop lane is promising and that Paddle is the first candidate for
table reconstruction; it does not authorize generated text as corpus text.

Model adapters must remain separate:

| Model | Intended next-round role | Required behavior |
| --- | --- | --- |
| PP-DocLayoutV3 | Primary deterministic region proposals | Boxes/order only; bind native anchors |
| PaddleOCR-VL 1.6 | First table/crop structure witness | Native task label; OTSL/Markdown; temperature 0 |
| GLM-OCR | Independent table/crop witness | Exact task labels; HTML/text; deterministic sampling; no reasoning parser |
| Qwen3.8-27B Q4 | Local vision-capable generalist parallel to Luna | Judge bounded candidates and later semantic evidence; never act as OCR |
| Luna | Hosted generalist baseline | Receive the same image/native/candidate packet as Qwen |
| Terra | Difficult semantic or extraction escalation | Review disagreement or genuinely mixed mechanics |

The longer-term comparison is therefore Qwen versus Luna, not Qwen versus the
OCR specialists. After layout evidence is repaired, both generalists should
also be evaluated on the actual workflow decisions: retained/excluded/mixed
classification and bounded Foundry-coverage judgments. Paddle or GLM output is
input evidence to that comparison and never the semantic answer.

For GLM-OCR, the documented deterministic profile is temperature 0,
`top_p=0.00001`, `top_k=1`, repetition penalty 1.1, and an 8,192-token output
ceiling. In llama.cpp, disable reasoning/chat-output parsing for this recognizer;
current integrations have placed ordinary output in a reasoning field or failed
PEG parsing. Paddle's official llama.cpp example also uses temperature 0.

## Round 2: manual protocol

Round 2 answers whether specialist crops and bounded generalist judgment improve
the existing V5 parser enough to justify integration. Qwen is not an OCR model:
it is evaluated as a smaller local alternative to Luna. Where a visual judgment
is needed, Qwen and Luna should receive the same image, authoritative native
text, deterministic candidates, and output contract. Terra remains escalation.
This round does not mutate the private review database or build a public base.

### Progress

The first `table-simple` case is complete. It used a new crop rather than the
single table from the exploratory round.

| Run | Crop | Result | Native agreement | Completion tokens | Time | Total VRAM used after request |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL 1.6 | detector box plus 10-point context | heading as one merged row, then 10x2 table | 1.000 over expanded crop | 226 | 0.919 s | 4,190 MiB |
| GLM-OCR | detector box plus 10-point context | 10x2 table; excluded the nearby heading | 1.000 over contained table words | 351 | 1.702 s | 5,236 MiB |
| PaddleOCR-VL 1.6 | exact detector box | 10x2 table | 1.000 over 72 contained native words | 218 | 0.835 s | within the first Paddle envelope |

The expanded-crop Paddle output was textually perfect but structurally wrong:
it treated a nearby heading as a merged table row. Using the exact deterministic
detector box removed that error. Text agreement by itself would therefore have
selected the wrong structure. Exact detector boxes are now the default input to
specialist recognizers; context margins must be a separately named candidate,
never an implicit crop modification.

GLM also established a local adapter requirement. With flash attention disabled,
this llama.cpp build rejects a quantized V cache before accepting requests.
GLM must use its F16 K/V cache profile; that setting is not shared with Paddle or
Qwen. The failed load processed no image and left no model server running.

No Qwen selection was needed because exact cropping left only one valid table
structure. The review database was not opened for writes, and all model servers
were stopped before advancing.

The second `table-dense` case is also complete. It contained 190 native words
(191 normalized tokens) across long prose-heavy cells.

| Model | Structure | Native agreement | Completion tokens | Time | Total VRAM used after request |
| --- | --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL 1.6 | 7x2, no merged cells | 1.000; 191/191 tokens | 407 | 1.534 s | 4,254 MiB |
| GLM-OCR | 7x2, no merged cells | 0.998; 187/191 tokens | 526 | 2.370 s | 5,291 MiB |

Both models independently agreed on the structure, so Qwen was again skipped.
Paddle is the stronger first recognizer on the two table cases so far; GLM is a
useful independent structural witness but its generated text is less exact.
This does not affect corpus text because reconstruction continues to use native
anchors only.

The first dense-image request exceeded the shell argument-size limit before it
reached the model. Subsequent manual requests read base64 from a private
temporary file instead of placing it in an argument. The empty-payload HTTP 500
is a transport-construction failure and is not counted as a model result.

The first generalist `heading-artifact` case is complete. V5 had promoted a
small repeated trait label into the section heading. The bounded candidates
were: A, keep the small label as the heading; B, use the larger title and treat
the small label as a trait/category badge; or `none`.

| Generalist | Mode | Repetitions | Decision | Completion tokens | Inference/wall time |
| --- | --- | ---: | --- | ---: | ---: |
| Qwen3.8-27B Q4 | documented xhigh thinking | 2 | B, high confidence | 570 / 620 | 40.833 / 41.454 s |
| Qwen3.8-27B Q4 | documented non-thinking | 2 | B, high confidence | 35 / 27 | 4.444 / 3.930 s |
| gpt-5.6-luna | schema-constrained Codex CLI | 1 | B, high confidence | 27 | 5.270 s wall |

All five judgments agreed with the visual hierarchy and the known parser
defect. The Qwen server's highest observed total VRAM use was 18,853 MiB, above
the required free-reserve floor. Its cold load took approximately 79 seconds;
that cost is process-scoped rather than per judgment.

On this easy bounded case, Qwen thinking added no decision quality over
non-thinking while increasing output and latency substantially. The provisional
route is therefore local non-thinking Qwen for ordinary visual hierarchy,
followed by thinking Qwen or Luna only for low confidence, instability, or a
harder candidate set. This is not yet justified for semantic licensing or
Foundry-coverage decisions; those require separate Qwen-versus-Luna tests.

The Luna comparison used Codex CLI 0.149.1 in an ephemeral read-only invocation
with the identical image, candidates, and output schema. It used 18,126 input
tokens and produced 27 output tokens. No worker tools, source transcription, or
review-database writes were permitted.

Two read-only `foundry-coverage` cases then tested the actual corpus decision.
The first candidate had complete lexical, trigram, and numeric coverage. The
second was a plausible lexical overlap with token coverage 0.702, trigram
coverage 0.683, and numeric coverage 0.4.

| Case | Qwen non-thinking | Luna | Terra escalation |
| --- | --- | --- | --- |
| likely complete | `covered`, high; 31 output tokens | `covered`, high; 56 output tokens | not needed |
| plausible partial | `additional-mechanics`, high; 27 output tokens | `covered`, high; 25 output tokens | `additional-mechanics`, high; 99 output tokens |

Terra independently agreed with Qwen on the disputed partial candidate. Luna
would therefore have created a false duplicate exclusion on this example. One
case cannot rank the models generally, but it proves that Luna is a comparator,
not ground truth, and that confidence does not make single-model exclusion safe.

The workflow must fail open when Qwen and Luna disagree: the section stays in
ordinary work as `ADD` or is deferred; it is never terminally rejected as a
Foundry duplicate. The 0.4 numeric coverage also suggests a cheap deterministic
guard: incomplete numeric/dice coverage should prevent terminal duplicate
rejection before a model is called. That guard remains provisional until a
stratified sample shows whether equivalent mechanics can produce legitimately
different normalized numeric signatures.

These semantic comparisons were ephemeral and schema-constrained. They read the
private section and clean Foundry row without images, tools, or database writes;
no source text or model output is retained after the content-free result is
recorded here.

### Stratified coverage qualification

A subsequent read-only qualification sampled 20 active Remaster sections: five
each from Player Core, GM Core, Monster Core, and Player Core 2. The deterministic
strata contained two exact matches, four high-completeness signatures, four
numeric gaps, four middling lexical matches, and six low or mismatched candidates.
Qwen received four batches totaling about 39 KiB after current-snapshot candidate
deduplication. The peak local server allocation remained within the established
safe Q4 envelope.

| Model | Reviewed | Covered | Additional mechanics | Uncertain |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B Q4, non-thinking | 20 | 13 | 7 | 0 |
| Luna | 20 | 10 | 9 | 1 |
| Terra escalation | 16 | 11 | 4 | 1 |

Qwen and Luna disagreed on five of 20 sections. Terra reviewed every Qwen
`covered` proposal, every disagreement, and a deterministic sample of Qwen
`additional-mechanics` results. Terra rejected two of Qwen's 13 proposed
coverage exclusions. Manual adjudication found both rejections substantive:
one Foundry row omitted a mechanically significant rank even though the naive
numeric coverage score was 1.0, and one private section contained text from an
adjacent rule.

A second manual check examined the three cases where Qwen and Terra agreed on
coverage but Luna did not. It found a dangling incomplete section, a heading/body
mismatch, and a heading-plus-page-number stub. The first two are unsafe inputs to
coverage review; the last is non-rule material whose correct resolution is an
intentional structural exclusion, not a Foundry duplicate proof. Agreement
between Qwen and Terra is therefore insufficient until deterministic structural
gates run first.

The temporary evaluator initially joined retained candidates from both the
active and an older Foundry snapshot, duplicating identical candidate IDs in its
packets. Filtering to the active snapshot produced the same 20 private sections
and the same unique candidate IDs, reducing 50 candidate rows to 25. No semantic
call needs repetition, but this confirms that every evaluation path must reuse
the production current-snapshot serializer rather than issuing an ad hoc join.

The qualification does not justify draining the semantic queues yet. The next
implementation should be deterministic and model-cheap:

- Reject stale snapshots and collapse candidates by Foundry ID before packing.
- Route dangling endings, heading/body incoherence, adjacent-rule contamination,
  and page-number stubs through structural repair or intentional quarantine.
- Replace set-like numeric coverage with an occurrence-aware signature that
  retains the mechanic attached to each number, including condition ranks.
- Let Qwen retain obvious additional mechanics cheaply. A `covered` judgment may
  advance only after all deterministic gates pass and Terra independently agrees.
- Treat Luna as a useful comparator during qualification, not as a production
  authority or a mandatory third vote.

Additional model sampling before those gates would mostly measure reactions to
known-bad inputs. Repair and re-serialize first, then repeat a smaller stratified
qualification against the exact production evidence path.

### Gate-first prompt and evidence ablation

A follow-up 16-case ablation separated prompt effects from evidence effects. It
used four initially known structural defects, four exact Foundry identities, one
known missing-mechanic case, three numeric-risk cases, and four ambiguous cases.
After inspecting native blocks and bounded neighbors, five of the risk/ambiguous
records were also demonstrably malformed. The adjudicated set therefore contained
nine structural defects, four exact matches, one real missing condition rank, and
two valid covered sections whose unmatched numbers were publication metadata or
a page citation rather than mechanics.

Every model received the same revised gate-first schema. It first had to classify
the input as `valid`, `needs-layout`, or `insufficient-context`; coverage was
permitted only for valid inputs. The prompt explicitly stated that a fragment is
not covered merely because Foundry contains its complete version and that repeated
numbers must be compared by mechanical role. Baseline packets contained the
isolated native section and current-snapshot Foundry candidates. Enriched packets
also contained bounded neighbors, native block boundaries, layout flags, numeric
contexts, and a redacted structural crop for each of the four initially known
layout cases.

| Model | Evidence | Needs layout | Covered | Additional | Adjudicated errors |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B Q4, non-thinking | baseline | 9 | 6 | 1 | 0 |
| Qwen3.8-27B Q4, non-thinking | enriched | 7 | 7 | 2 | 3 |
| Luna | baseline | 9 | 7 | 0 | 1 |
| Luna | enriched | 8 | 6 | 2 | 1, fail-open |
| Terra | baseline | 9 | 6 | 1 | 0 |
| Terra | enriched | 9 | 6 | 1 | 0 |

All three baseline runs caught all nine structural defects, including five that
were not in the initial structural stratum, and accepted all four exact matches.
The revised prompt and output space therefore explain much of the earlier failure:
the previous coverage-only schema forced models to answer the wrong question.

Terra was stable across evidence conditions and correctly handled the known
condition-rank difference. Luna missed that difference in the baseline but fixed
it with enrichment; one malformed input then moved to a valid fail-open addition.
Qwen's baseline matched the adjudicated set, but enrichment made it mark two
malformed inputs as valid additions and changed the known missing-mechanic case
to an unsafe covered result. More evidence was not monotonically better.

The enriched lane also cost roughly three times the input of the baseline in
this deliberately unoptimized manual setup: Qwen used about 7.7k versus 23.7k
prompt tokens, Luna 39.4k versus 125.8k input tokens, and Terra 43.4k versus
136.2k input tokens. Per-image isolation repeats prompt scaffolding, so these
are not production cost estimates, but the additional evidence produced no
Terra decision change. A single Sol escalation over the three enriched model
disagreements returned the adjudicated result for all three.

The resulting workflow should keep repair and coverage separate:

- Use deterministic checks plus a compact gate-first packet for initial routing.
- Exact normalized identities remain deterministic and need no model.
- Qwen may cheaply route obvious malformed or additional sections, but it must
  not authorize coverage suppression.
- Qwen routes structurally valid non-exact candidates with a compact baseline
  packet. `additional-mechanics` remains included without another coverage call;
  `covered` and `uncertain` advance to independent Sol confirmation.
- Only Qwen and Sol agreement may suppress a non-exact PDF occurrence as covered
  by Foundry. Sol disagreement fails open to ordinary retained work. Terra stays
  reserved for mechanics-only extraction from genuinely mixed retained sections
  and later rework, not routine coverage confirmation.
- Neighbors, images, and specialist OCR belong to the layout-repair lane. After
  repair, rebuild the section and return to the compact semantic packet instead
  of carrying all repair evidence into coverage review.

This ablation is small and intentionally difficult. It supports the routing and
prompt design above; it does not establish a general model error rate.

### Cases

Select one private example for each case below. Store the product/page mapping
only in an ignored local note.

1. Ordinary two-column prose with an unambiguous order.
2. A bottom-left to top-right column continuation.
3. A cross-page continuation.
4. A simple table with stable rows and columns.
5. A dense or merged-cell table.
6. A sidebar adjacent to main-column prose.
7. A heading whose attachment is ambiguous in native extraction.
8. A region containing action glyphs or compact labels.
9. A dense Monster Core stat block currently using native-layout fallback.
10. A mixed page with at least three region types.

The tracked case names are intentionally abstract. Images, page numbers, native
text, anchor IDs, and local paths stay in `.local-corpus/` and are removed when
the comparison is complete.

### Per-case sequence

1. Save the current V5 result and PP-DocLayout regions as the baseline.
2. Derive all candidate regions, reading orders, or attachments
   deterministically from native geometry. Give them opaque local IDs.
3. For a table or compact region, run Paddle on only the crop. Record structure
   metrics, then stop the server and return VRAM to baseline.
4. Run GLM on the same crop only when the case is a table, Paddle disagrees with
   native geometry, or an independent witness is needed. Stop it afterward.
5. When two or more deterministic candidates remain, run Qwen as the local
   generalist and compare it with Luna on the same bounded packet. Ask each to
   select one candidate ID or `none`; do not send a coordinate-generation task.
6. Reconstruct text exclusively from native anchors under the selected
   structure. Verify every anchor occurs exactly once.
7. Compare the candidate with V5. Do not activate it or mutate review state.

Paddle and GLM are deterministic single runs unless the output is malformed.
Malformed output is a failure, not a reason to sample repeatedly. For Qwen,
run two thinking and two non-thinking selections with each mode's documented
sampling profile. Compare those results with one Luna judgment using identical
evidence. Treat instability or Qwen/Luna disagreement as unresolved; do not
majority-vote it into a repair.

### Content-free result record

Record one local result object per case with these fields. Hashes may identify
transient inputs locally, but no hash or metric file is promoted automatically.

```json
{
  "case": "table-simple",
  "baseline_flags": ["unresolved-table"],
  "candidate_count": 2,
  "paddle": {
    "schema_valid": true,
    "rows": 5,
    "columns": 3,
    "native_text_agreement": 0.946,
    "latency_ms": 390,
    "peak_vram_mib": null
  },
  "glm": null,
  "qwen": null,
  "anchor_inventory_equal": true,
  "anchor_ownership_exactly_once": true,
  "improves_v5": true,
  "maintainer_note": "content-free summary only"
}
```

Use actual measured values; the example numbers merely show the shape. Do not
store OCR text, prompts containing private text, boxes derived from private
pages, or paths in a tracked report.

### Adoption gate

The secondary lane remains rejected unless all of these are true:

- Every proposed repair preserves the complete native-anchor inventory exactly
  once and changes no native word.
- The method improves at least one currently unresolved V5 layout class without
  regressing an already-correct case.
- Repeated Qwen selections are stable, or the task is routed to a maintainer.
- Table dimensions and cell associations agree with native geometry and manual
  inspection; text similarity alone is insufficient.
- Peak VRAM remains above the 5 GiB free-reserve floor for every local model.
- Private artifacts are transient or ignored, and no review database is changed.

If the gate passes, the next implementation should be the smallest adapter for
the demonstrated defect class. It should not introduce a generic OCR pipeline.

## Primary references

Documentation was checked on 2026-08-29. Local llama.cpp behavior refers to
build 10666 and must be rechecked after an upgrade.

- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md)
- [Qwen generation configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/generation_config.json)
- [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [PaddleOCR-VL pipeline](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md)
- [PaddleOCR-VL 1.6 model](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)
- [GLM-OCR repository](https://github.com/zai-org/GLM-OCR)
- [GLM-OCR pipeline configuration](https://github.com/zai-org/GLM-OCR/blob/main/glmocr/config.yaml)
- [llama.cpp GLM-OCR integration discussion](https://github.com/ggml-org/llama.cpp/discussions/19721)
- [Paddle PEG parser issue](https://github.com/ggml-org/llama.cpp/issues/25339)
- [GLM-OCR reasoning-output issue](https://github.com/ggml-org/llama.cpp/issues/20248)
