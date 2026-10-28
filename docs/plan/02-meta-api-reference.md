# 02: Meta Model API reference for FrameLab

Everything here comes from the official docs (`dev.meta.ai/docs/*`) and the official
cookbooks (`github.com/meta-models/meta-model-cookbook`, `meta-oss-cookbook`) as of
28 Oct 2026. Items marked **VERIFY** are not fully documented and must be checked in
milestone M1 with a spike script. Record the result in `11-risks-and-open-questions.md`.

## 1. Basics

- Base URL: `https://api.meta.ai/v1` (all hosted models).
- Auth: `Authorization: Bearer $MODEL_API_KEY`. Key format `LLM|{numeric_id}|{secret}`.
- The OpenAI Python SDK works when pointed at the base URL. It does **not** read
  `MODEL_API_KEY` automatically; pass it explicitly.
- Context window: 1,048,576 tokens for Muse Spark / Muse Wonder.
- APIs: Chat Completions (OpenAI-compatible), Responses API (agentic, server state,
  reasoning replay), Messages API (Anthropic-compatible), Files API, Images, ASR.

```python
# framelab/clients/meta.py (core)
import os
from openai import OpenAI, AsyncOpenAI

BASE_URL = "https://api.meta.ai/v1"


def sync_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])


def async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])
```

## 2. Model IDs and roles

| Constant (config.py) | Model ID | Role in FrameLab |
|---|---|---|
| `MODEL_SPARK` | `muse-spark-1.3` | Planner, Analyst, Designer, report prose |
| `MODEL_WONDER` | `muse-wonder-1.0` | Auditor (fallback to `MODEL_SPARK` if `FRAMELAB_USE_WONDER=0`) |
| `MODEL_SAM` | `sam-3.1` | Video tracking and image segmentation |
| `MODEL_VOICE` | `muse-voice-transcribe-1.0` | Speech → text |
| `MODEL_IMAGE` | `muse-image-1.0` | (S-2) apparatus diagram |
| `MODEL_SPARK_OPEN_BASETEN` | `meta-models/muse-spark-open-1026` | (S-3) Designer on open weights (hackathon-specific slug, not the docs' `meta-models/muse-spark-open`) |

**Do not use Contributor mode** (`*-contributor`) for real user videos. Contributor content
may be used to train Meta models, and videos may show people. Contributor mode may be used
for synthetic text-only dev calls if cost matters.

## 3. Prices (for `costs.py`)

| Model | Price |
|---|---|
| Muse Spark 1.3 (Standard) | input $1.25 / 1M, cached input $0.15 / 1M, output $4.25 / 1M |
| Muse Wonder 1.0 | input $4.00 / 1M, cached input $0.20 / 1M, output $20.00 / 1M |
| SAM 3.1 | images $2.50 / 1,000; video $0.20 / 1,000 frames (min 50 frames per video request) |
| Muse Voice Transcribe | $0.18 per audio hour (billed per second) |
| Muse Image | $0.01 per generated image |
| Web search grounding | $2.50 / 1,000 queries (FrameLab does not use it) |

Rate limits: Spark/Wonder 3,000 RPM / 4M TPM; SAM 3,000 RPM; Image 150 RPM; Voice 256
concurrent streams and 16,000 streams/hour.

`costs.py` computes an estimated USD cost per call from `usage` (tokens) or units
(frames, images, seconds) and the orchestrator sums it into `state.json.cost_usd`.

## 4. Files API (upload once, reuse the `file_id`)

```python
async def upload_file(path: Path, mime: str = "video/mp4") -> str:
    async with async_client() as api:
        uploaded = await api.files.create(
            file=(path.name, path.read_bytes(), mime),
            purpose="user_data",
        )
    assert uploaded.id.startswith("file-")
    return uploaded.id
```

- Upload `ingest/video_with_audio.mp4` for the Spark Planner and the trimmed silent
  `ingest/analysis.mp4` for SAM (fewer billed frames). SAM also accepts MP4 with audio (V4 ✓).
- Video must be `video/mp4`. No size/duration limit was hit up to 1,200 frames (V3 ✓).
- Retention (V10 ✓): `files.retrieve` shows `expires_at: null` (files never expire on their
  own); `files.delete(file_id)` works → **delete every uploaded file at the end of the run**
  (try/except, log failures).

## 5. SAM 3.1: segmentation and tracking

Docs: `dev.meta.ai/docs/media-segmentation`, `/docs/sam/segmenting`,
`/docs/sam/reading-segmentation`. Cookbook: `07_sam/01_api_basics/python/sam_api.py`.

Key facts:
- Prompt = **one short concrete noun phrase per request** (e.g. `"orange ball"`,
  `"ruler"`). For several concepts, send several requests.
- Image: `input_image` with a data URL. Video: upload via Files API, then `input_video`
  with `file_id`.
- Responses are **streamed**. Decode them with the official parser `meta_sam_parser`
  (`pip install meta_sam_parser`). Never parse the text grammar yourself.
- Video results track each object across frames and keep its `object_id`.
- Box/point prompts are also supported (`build_object_prompt`) for fallback when text
  prompting fails.

### 5.1 Request + parse (adapted from the cookbook)

```python
from contextlib import aclosing
from meta_sam_parser import (
    parse_responses_stream,
    image_segmentation_format,
    video_segmentation_format,
    decode_mask_to_raster,
    SegmentationMaskRecord,
    SegmentationBoxRecord,
)


def _input(concept: str, media: dict) -> list[dict]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": concept.strip()}, media],
        }
    ]


async def _events(stream):
    """Convert SDK events to plain dicts and synthesize output_text.done.
    The API streams response.content_part.done but not response.output_text.done,
    which the parser needs to close a lane. (Copied behaviour from cookbook sam_api.py.)"""
    done = set()
    async for ev in stream:
        d = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
        lane = (d.get("item_id"), d.get("output_index"), d.get("content_index"))
        if d.get("type") == "response.output_text.done":
            if lane not in done:
                done.add(lane)
                yield d
            continue
        yield d
        part = d.get("part")
        if (
            d.get("type") == "response.content_part.done"
            and lane not in done
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            done.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": d.get("item_id"),
                "output_index": d.get("output_index"),
                "content_index": d.get("content_index"),
                "text": part["text"],
            }


async def segment_video(file_id: str, concept: str, raw_log: Path):
    async with async_client() as api:
        stream = await api.responses.create(
            model="sam-3.1",
            stream=True,
            input=_input(concept, {"type": "input_video", "file_id": file_id}),
        )
        async with stream, aclosing(_events(stream)) as events:
            # Tee events to raw_log (JSONL) for golden replay before parsing.
            parsed = parse_responses_stream(_tee(events, raw_log), video_segmentation_format())
            async with parsed:
                async for _snapshot in parsed:  # cumulative snapshots; emit progress
                    pass
                return await parsed.final_result()


async def segment_image(image_path: Path, concept: str, raw_log: Path):
    async with async_client() as api:
        stream = await api.responses.create(
            model="sam-3.1",
            stream=True,
            input=_input(concept, {"type": "input_image", "image_url": data_url(image_path)}),
        )
        async with stream, aclosing(_events(stream)) as events:
            return await parse_responses_stream(
                _tee(events, raw_log), image_segmentation_format()
            ).final_result()
```

Validate every result:

```python
def validate(result) -> None:
    if result.outcome.status != "completed":
        raise SamIncomplete(
            getattr(result.outcome, "reason", "unknown"), getattr(result.outcome, "detail", None)
        )
    errs = [d for d in result.diagnostics if d.severity == "error"]
    if errs:
        raise SamParseError("; ".join(f"line {d.line}: {d.message}" for d in errs))


# A completed result with zero masks is a valid EMPTY match, not an error.
```

### 5.2 Record types (from `meta_sam_parser._types`)

```text
SegmentationBoxRecord:  kind="box",  order, object_id: str, left, top, right, bottom: float,
                        frame: FrameReference|None (frame.frame_index: int), confidence: float|None
SegmentationMaskRecord: kind="mask", order, object_id: str, identity, revision: int,
                        mask: SegmentationMask(encoding, payload, width:int, height:int),
                        bounds: SegmentationMaskBounds(left, top, right, bottom: float),
                        frame: FrameReference|None, confidence: float|None
SegmentationTextRecord: kind="text", order, text
```

- `decode_mask_to_raster(record.mask) -> bytes`: row-major, `width*height` bytes, each 0/1.
  Convert with `np.frombuffer(raw, dtype=np.uint8).reshape(height, width)`.
- The raster covers the **mask bounds**, not the whole frame. Map local pixel (row, col) to
  image coordinates with:
  `x = bounds.left + (col + 0.5) * (bounds.right - bounds.left) / width`,
  `y = bounds.top + (row + 0.5) * (bounds.bottom - bounds.top) / height`.
  In the cookbook example the mask size equals the box size, so the scale factor is
  usually 1. Keep the general formula anyway.
- Video results can contain several revisions of a mask for the same (object, frame).
  **Keep the highest `revision`** per `(object_id, frame_index)`.
- **Verified in M1 (11 §1):**
  - `frame_index` is 0-based and 1:1 with the decoded frames of the uploaded file (V1).
    Billing/usage: `response.completed.response.usage.video_frames_processed` = all frames.
  - Coordinates are in the uploaded video's pixel space (V2, error ≈ 0.3–0.5 px). The raw
    grammar is `<NNf>OBJ<|box;x1;y1;x2;y2;w;h|><|mask;x;y;data=W,H,…|>`; box x2/y2 are
    inclusive, parser `bounds` are half-open. Always use the parser.
  - Mask raster size == bounds size (V11). Masks are ~1 px larger per side than the true edge.
  - 240 fps × 3 s and 60 fps × 20 s work; throughput ≈ 20 frames/s (V3).
  - **Motion onset (V12): in video mode SAM reports an object only from the frame AFTER it
    first moves.** Objects that are static from the start of the clip get NO masks until they
    move (a ball held still for 2 s: first mask at motion start + 1 frame; a clip that starts
    with motion: first mask at frame 1). Once tracked, the object stays tracked while static.
    ⇒ Static things (ruler, pivot, string) are segmented in **image** mode on key frames.
    ⇒ For free fall the first mask frame ≈ release + 1 frame (05 §2.1).
    ⇒ Stretch S-1 (melting ice, never moves) needs image-mode segmentation per key frame or a
      box-prompt video request (§5.3); verify before building S-1.

### 5.3 Box/point prompt fallback

If the text prompt returns no object or the wrong one, re-run with a box prompt on the
first analysis frame. The box comes from the Planner (`spec.target_hint_box`, optional)
or from the largest moving blob (frame differencing).

```python
from meta_sam_parser import build_object_prompt, PromptObject, FramePrompt, PromptBox, PromptPoint

# Signatures (meta_sam_parser 0.1.0):
#   PromptBox(left:int, top:int, right:int, bottom:int)   # right/bottom half-open, whole pixels
#   PromptPoint(x:int, y:int, label="positive"|"negative")
#   FramePrompt(frame:int=0, box=None, points=())          # frame 0 = the only frame of an image
#   PromptObject(id:int|str, prompts:Sequence[FramePrompt])
text = build_object_prompt(
    size=(W, H),
    objects=[
        PromptObject(
            id=1,
            prompts=[
                FramePrompt(frame=k0, box=PromptBox(l, t, r, b), points=[PromptPoint(cx, cy)])
            ],
        ),
    ],
)
# Send `text` as the input_text instead of the noun phrase (same request shape).
# W, H = pixel size of the uploaded (normalized) video; all coords must be ints inside it.
```

## 6. Muse Spark 1.3

### 6.1 Chat Completions + tool calling (Analyst, Designer)

Proven pattern from `01_api_fundamentals/03_tool_calling.ipynb`:

```python
resp = client.chat.completions.create(
    model="muse-spark-1.3",
    messages=messages,
    tools=TOOLS,  # [{"type":"function","function":{name, description, parameters}}]
    reasoning_effort="high",  # minimal | low | medium | high  (+ "max" on 1.3)
    prompt_cache_key=f"framelab-analyst-v{PROMPT_VERSION}",
)
msg = resp.choices[0].message
if msg.tool_calls:
    messages.append(msg)  # MUST replay the assistant message holding tool_calls first
    for tc in msg.tool_calls:
        result = dispatch(tc.function.name, json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
```

- Sending a tool result without first replaying the assistant `tool_calls` message gives
  HTTP 400.
- Parallel tool calls are supported. Execute them in order and return all results.
- Reasoning text, if any, is in `msg.model_extra.get("reasoning_content")` (log it to
  events as `reasoning_summary`, truncated to 2,000 chars).
- Prompt caching: put the long stable system prompt first. Use `role: "developer"` or
  `"system"` for it, and pass a stable `prompt_cache_key`. Read `usage.prompt_tokens_details.cached_tokens`.

### 6.2 Structured output

Proven on Chat Completions (`04_structured_output.ipynb`):

```python
completion = client.beta.chat.completions.parse(
    model="muse-spark-1.3", messages=messages, response_format=MyPydanticModel
)
obj = completion.choices[0].message.parsed
# or: response_format={"type":"json_schema","json_schema":{"name":..., "schema":..., }}
```

Always validate with Pydantic. On failure, retry once with the validation error appended.

### 6.3 Video input (Planner)

From `dev.meta.ai/docs/video-understanding`: upload the MP4 via Files API, then reference it
in a **Responses API** request:

```python
resp = await api.responses.create(
    model="muse-spark-1.3",
    reasoning={"effort": "medium"},
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": PLANNER_USER_PROMPT},  # includes transcript JSON
                {
                    "type": "input_file",
                    "file_id": file_id,
                },  # or {"type":"input_video","file_id":...}
            ],
        }
    ],
    instructions=PLANNER_SYSTEM_PROMPT,
    text={
        "format": {
            "type": "json_schema",
            "name": "experiment_spec",
            "schema": strict_schema(ExperimentSpec),
            "strict": True,
        }
    },
)
spec = ExperimentSpec.model_validate_json(resp.output_text)
```

- **Verified in M1 (V5, V6, V9):** `input_file`, `input_video` (Responses, `file_id`) and
  Chat `video_url` (data URL) all work. Use `input_file`.
- **Spark does NOT hear the embedded audio** (it answered "silent" for a clip with clear
  speech). The Voice transcript is therefore REQUIRED input for the Planner.
- Spark samples frames sparsely (it misread first/last frame counters), so its
  `analysis_window_s` is coarse. Pad it (04 §2.4) and let code refine it (05 §2.1).
- Cost is negligible: ~1.5k input tokens for a 5 s clip (~$0.007 per call).
- Structured output on the Responses API works with `strict: false`; `strict: true` requires a
  strict-compatible schema (every object `additionalProperties: false`, every property listed
  in `required`, optional values as nullable types). Implement `framelab.clients.meta.
  strict_schema(model) -> dict` (recursive; see `scripts/spikes/spike_followup.py`) and use
  `strict: true` everywhere. Always validate with Pydantic models configured
  `extra="forbid"`.
- Fallback if video input fails: send 8 key frames as `input_image` data URLs plus the
  transcript (Chat Completions supports images).

### 6.4 Responses API notes
- Multi-turn: `previous_response_id=resp.id` (server-side state, `store=True` default).
- Reasoning replay for stateless use: `include=["reasoning.encrypted_content"]`, `store=False`.
- `resp.output_text` gives the final text; `resp.output` gives typed items.

## 7. Muse Wonder 1.0 (Auditor)

- Model ID `muse-wonder-1.0`, Standard mode only on the Model API, reasoning levels
  `minimal … max`, structured output and tool calling supported.
- **M1 finding (V7, V7b): Wonder is not reliable for our structured tasks yet.** Non-strict:
  extra keys and wrong content. Strict (`strict: true` + strict schema): keys obeyed, but on
  the same short extraction prompt Chat returned `target_concept: "Measurement"` and Responses
  returned `experiment: "melting"` with an essay as target. Spark answered perfectly in the
  same test. Wonder spends ~1.4–1.7k reasoning tokens per call (~$0.03).
- **Decision:** the Auditor defaults to `muse-spark-1.3` (`FRAMELAB_USE_WONDER=0`) with its own
  system prompt and NO shared conversation history (independence comes from the separate
  context, not the model). In M7 re-evaluate Wonder once on a realistic auditor input (full
  draft + ledger + gates, strict `Verdict` schema). Enable it only if it returns correct
  verdicts on the PASS and the trap fixture. If enabled: Chat Completions, strict schema,
  `reasoning_effort="high"`, Pydantic `extra="forbid"`, one retry, then Spark fallback.
  Log the auditor model in every verdict.

## 8. Muse Voice Transcribe

From `06_muse_voice/01_voice_api_fundamentals/utils.py`:

```python
import json, httpx

TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"


def transcribe(path: Path, keywords: list[str], language_bias: list[str] | None) -> dict:
    request = {"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    if keywords:
        request["keywords"] = keywords
    if language_bias:
        request["languageBias"] = language_bias
    with path.open("rb") as audio:
        r = httpx.post(
            TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"},
            files={
                "request": (None, json.dumps(request), "application/json"),
                "audio": (path.name, audio),
            },
            timeout=300,
        )
    r.raise_for_status()
    return (
        r.json()
    )  # {"sessionId","transcript","audioDurationMs","turns":[{"transcript","startMs",...}]}
```

- Audio: WAV, mono, 16-bit PCM, 16 kHz or 24 kHz. Max 10 min / 32 MB.
  `ffmpeg -i in.mp4 -vn -ac 1 -ar 16000 -sample_fmt s16 audio.wav`
- Modes: `PUSH_TO_TALK` (default, whole clip is one turn source), `ENDPOINTING` (turn
  boundaries), `DIARIZATION` (speaker labels). FrameLab uses **`ENDPOINTING`** so that
  parameters and hypothesis appear as separate turns with `startMs`. Unknown modes are
  silently ignored by the server, so only use these three names.
- `keywords`: `["bandul", "pendulum", "gravitasi", "gravity", "sentimeter", "centimeter",
  "meter", "detik", "second", "periode", "period", "penggaris", "ruler", "bola", "ball",
  "hipotesis", "hypothesis", "diameter"]`.
- `languageBias`: **omit** (auto-detect + code-switching). M1 (V8): `["indonesian"]` and
  `["id"]` are both accepted, but with `["indonesian"]` numbers came back as words ("zero
  point four five") instead of digits ("0.45"), so it makes extraction harder.
- `ENDPOINTING` returns `turns[]` with `startMs/endMs`; `PUSH_TO_TALK` returns none (V8).
- Re-test with a real human voice recording in M5 (the M1 test used Windows TTS).
- Indonesian is one of the 25 supported languages. Timestamps are turn-level only. There is
  no TTS.
- Errors: 400 = too long/malformed, 413 = too large.

## 9. Muse Image (stretch S-2)

From `05_muse_image/01_image_api_fundamentals`:

```python
resp = client.responses.create(
    model="muse-image-1.0",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Redraw this experiment as a clean textbook line "
                    "diagram on white background: a ball falling next to a vertical ruler. "
                    "No text, no labels, no numbers.",
                },
                {"type": "input_image", "image_url": data_url(keyframe_png)},
            ],
        }
    ],
)
b64 = next(i.result for i in resp.output if i.type == "image_generation_call")
```

- $0.01/image. Built-in web/image search and shell are on by default. Disable them for
  determinism if parameters are supported (`enable_web_search`, `enable_image_search`,
  `enable_shell`: **VERIFY** placement).
- Never let the model draw labels or numbers. Overlay labels with Pillow afterwards.
- Content-safety blocks return HTTP 400. Treat as "diagram unavailable" (non-fatal).

## 10. Muse Spark Open via Baseten (stretch S-3)

From `meta-oss-cookbook/muse-spark/hosted/baseten.md` + hackathon guide:

```python
client = OpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url="https://inference.baseten.co/v1")
resp = client.chat.completions.create(
    model="meta-models/muse-spark-open-1026",  # hackathon slug (guide), NOT meta-models/muse-spark-open
    messages=[...],
)
```

Used only for the Designer when `FRAMELAB_DESIGNER_BACKEND=baseten`. README must state it
clearly (bonus eligibility). License note: products using Muse Spark Open must display
"Built with Muse Spark Open". Add it to the UI footer when S-3 is enabled.

## 11. Error handling and retries (all clients)

Pattern from `01_api_fundamentals/09_error_handling.ipynb`:
- Retry on 429, 500, 502, 503, 504 and network errors: exponential backoff with full
  jitter, base 1 s, max 30 s, max 5 attempts.
- Do not retry other 4xx. Raise `MetaClientError(status, body_excerpt, request_id)`.
- Timeouts: Spark/Wonder 180 s, SAM video 600 s (stream), Voice 300 s, Files upload 300 s.
- Every call emits `model_call` (start) and `model_result` (end, with usage/cost/latency)
  events. Never log request bodies containing base64 media. Log sizes instead.
