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
| Qwen3.8-27B Q4 | Select bounded order/attachment candidates | Return candidate IDs, never coordinates or transcription |
| Luna/Terra | Later semantic/license review | Consume repaired native text only; outside this layout round |

For GLM-OCR, the documented deterministic profile is temperature 0,
`top_p=0.00001`, `top_k=1`, repetition penalty 1.1, and an 8,192-token output
ceiling. In llama.cpp, disable reasoning/chat-output parsing for this recognizer;
current integrations have placed ordinary output in a reasoning field or failed
PEG parsing. Paddle's official llama.cpp example also uses temperature 0.

## Round 2: manual protocol

Round 2 answers whether specialist crops and bounded candidate selection improve
the existing V5 parser enough to justify integration. It does not review
licensing, mutate the private review database, or build a public base.

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
5. Run Qwen only when two or more deterministic candidates remain. Ask it to
   select one candidate ID or `none`; do not send a coordinate-generation task.
6. Reconstruct text exclusively from native anchors under the selected
   structure. Verify every anchor occurs exactly once.
7. Compare the candidate with V5. Do not activate it or mutate review state.

Paddle and GLM are deterministic single runs unless the output is malformed.
Malformed output is a failure, not a reason to sample repeatedly. For Qwen,
run two thinking and two non-thinking selections with each mode's documented
sampling profile. Treat disagreement as unresolved; do not majority-vote it
into a repair.

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
