INSTRUCTION_PROMPT = """
Task:
You will be given a multi-person scene prompt and asked to structure and/or expand it according to the requirements specified below.

The original prompt may vary widely in its level of specificity. It can range from a highly detailed description—explicitly specifying the setting, environment, individual actions, clothing, posture, spatial arrangement, and interactions—to a coarse, high-level prompt that provides only a general description of the scene.

Accordingly, the role of the system adapts to the prompt’s granularity:
- Highly detailed prompts: The system primarily performs parsing and structuring, converting the given description into the required format without omitting, altering, or contradicting any details.
- Coarse or high-level prompts: The system additionally performs controlled expansion, instantiating plausible person-specific roles, actions, and interaction details to produce multiple faithful realizations of the original intent.
- Prompts of intermediate specificity: The system naturally combines both behaviors, structuring the provided details while expanding unspecified aspects in a plausible manner.

Faithfulness to the original prompt is mandatory in all cases.
All structured or expanded outputs must describe scenes that satisfy and align with the original prompt:
- If the original prompt is general, each expanded version should represent a plausible realization of the described scene. Diversity across realizations is encouraged.
- If the original prompt is highly detailed, no required detail may be missing, modified, or contradicted. Additional details may be introduced only when they do not conflict with the original description.

For every input prompt, you will produce five structured/expanded versions:
- For general prompts, the five outputs should be as diverse as possible while remaining faithful.
- For highly detailed prompts, the five outputs may be similar, differing only in aspects that can be safely diversified.
- If no meaningful expansion is possible without risking inconsistency, it is acceptable to focus primarily on accurate structural parsing.

The overarching goal is to faithfully realize the original prompt, either by exploring diverse plausible instantiations (when under-specified) or by expressing the same scene through consistent structured representations (when fully specified).

Step 1: Scene-Level Realization
For each realization, first determine a coherent scene setup, including:
- A concise description of the setting or environment
- The overall situation implied by the prompt
- The participants involved and their roles in the scene
Environment descriptions should be brief and non-distracting.
You may vary the setting across realizations to increase diversity, as long as the setting remains compatible with the original prompt. Do not introduce environments or contexts that violate the prompt’s intent.

Step 2: Person-Level Descriptions
For each person in the scene, provide an individual description that focuses on that person while maintaining continuity with others.
Each person description should include the following elements (or closely related information):
1. Look / Appearance
- Describe the person’s appearance concisely (e.g., clothing, hairstyle, general look).
- Appearance should be appropriate for the scene.
- Clothing colors and styles may vary across realizations for diversity.
- If the original prompt specifies appearance details, they must be preserved exactly. Do not modify or contradict them.

2. Actions
- Describe meaningful actions the person is performing.
- Actions should clearly relate to the scene and to other people when applicable.

3. Pose and Posture
- Specify body orientation and posture (e.g., facing forward, turned sideways, leaning, standing upright).
Vary framing naturally:
Some people may appear full-body; Others may appear half-body or upper-body; Different people may be seen from slightly different viewpoints or camera distances to create visual depth and asymmetry.

4. Interactions
Clearly express how people relate to one another: Gestures, Physical or object-based interactions
It is encouraged to reference other people briefly (e.g., “looking toward the speaker,” “standing beside her”) to maintain scene coherence.
Each description should still focus primarily on the individual being described.

5. Spatial Relations
Describe each person’s position relative to others (e.g., “to her left,” “behind him,” “closer to the foreground”).
Write descriptions in a consistent visual order (e.g., left-to-right or front-to-back), aligned with the bounding box or layout order.

Diversity and Faithfulness Constraints
- All five realizations must faithfully satisfy the original prompt.
- Diversity is encouraged, but never at the cost of violating the prompt.

Specifically:
For general or under-specified prompts:
- The five realizations should explore diverse yet plausible instantiations of the scene.
- Diversity may come from environment choice, clothing, spatial arrangement, viewpoint, or interaction dynamics.

For highly specific prompts:
- The five realizations may be similar.
- Seek diversity primarily through camera viewpoint, framing, spatial composition, or subtle mood differences.
- Do not invent new actions, roles, or attributes that are not implied by the prompt.
If meaningful expansion is difficult without risking inconsistency, prioritize accurate structural parsing over forced diversity.

Believability and Interaction Fit
Across all realizations:
- Scenes must remain physically and socially plausible.
- Poses, body orientation, and interactions should feel natural for the setting.
- Interactions should make sense given the environment and the relationship implied by the prompt.

The ultimate objective is to realize the original prompt as clearly and faithfully as possible, either by:
- Exploring multiple plausible interpretations when the prompt is under-specified, or
- Expressing the same fully specified scene through consistent structured representations when details are fixed.

Incremental Person-by-Person Prompt Construction
A key requirement of the task is that each scene must be represented through an incremental, person-by-person prompt construction process.
After deciding on a complete scene realization (including setting, participants, and interactions), you must decompose the scene into a sequence of prompts, where each prompt introduces one additional person.

Assume the scene contains N people. You must generate N prompt stages, where:

- Stage 1 describes a scene in which only the first person is known to exist.
- Stage 2 describes a scene in which the first and second persons are known to exist.
…
- Stage N describes the full scene containing all N people.

Each stage must be written as a complete, natural, standalone prompt, but it may only describe the people that have been introduced up to that stage.
Future people must not be mentioned, implied, or referenced before their corresponding stage.

Choosing the Person Introduction Order

Before writing the prompts, you must determine a natural and compositionally stable order in which people are introduced.
The ordering should satisfy the following principles:
- Introduce foreground or interaction-central people earlier than background figures.
- Prefer introducing people who are less dependent on others first.
- Avoid introducing a person whose actions, gaze, or purpose would be unclear without knowing another person already exists.
The goal is that each person can be plausibly drawn or imagined at the time they are introduced, even without knowledge of future people.

For example:

In an interaction such as someone giving flowers to another person, it is often more natural to:
- First introduce the seated or receiving person, described as facing forward or toward an unspecified direction.
- Then introduce the person bending forward to hand over the flowers.
This avoids describing early-stage actions that depend too strongly on an unseen person.

In general, introduction order should progress from:
- main/foreground → surroundings
- more visually and spatially independent → more interaction-dependent
- closer / foreground → farther / background

Stage-Specific Prompt Requirements

Each stage must:
- Describe only the people introduced up to that stage.
- Remain self-consistent and visually plausible if interpreted in isolation.
- Use natural language that does not assume missing participants.

As new people are added:
- Previously described people may be slightly refined or contextualized, but:
    -- Their core attributes, actions, and positions must remain consistent.
    -- You must not contradict earlier descriptions.
- New interactions may be introduced only when both participants are present in the current stage.

Each stage prompt should feel like:
“This is the full scene as far as I currently know.”

Although each stage is partial, all stages together should:
- Form a monotonic refinement of the same underlying scene.
- Converge naturally toward the final, complete realization.
- Never require retroactive reinterpretation of earlier stages.
The final stage (Stage N) should fully realize the scene described by the original prompt, while earlier stages represent valid partial observations of that same scene.


Prompt Segmentation and Person-Level Annotation

In addition to producing a natural-language prompt for each stage, you must
also provide a segment-level annotation for that prompt.

For each stage, you will output:
1. A complete, natural prompt describing the scene at that stage.
2. A segmentation of that prompt into ordered text segments, each annotated
   with a tag indicating which entity it refers to.

Stage Prompt Requirements
-------------------------
- The stage prompt should be written as a single, fluent, self-contained
  description of the scene.
- There is no requirement that the prompt explicitly describe people one by one
  or follow any rigid template.
- The wording should feel natural, as if written for an image generation model
  or captioning task.
- The segmentation process happens AFTER the prompt is written.

Segment Annotation Format
-------------------------
Each stage prompt must be segmented into an ordered list of text fragments.

Each segment consists of:
- text: a contiguous substring of the prompt
- tag:
    tag = 0 : scene-level or shared description not specific to any individual
    tag = 1 : description specific to the first person
    tag = 2 : description specific to the second person
    ...
    tag = k : description specific to the k-th person introduced in the scene

Exact Reconstruction Constraint
-------------------------------
Concatenating all segment texts in order must reproduce the original stage
prompt EXACTLY.

Formally:
    "".join(segment["text"] for segment in segments) == caption

This includes:
- All words
- All punctuation
- All spaces

Quotation marks, commas, periods, and spaces between sentences must belong to
one of the adjacent segments. No characters may be dropped, duplicated, or
reordered.

Segmentation Guidelines
-----------------------
- Segments should follow the natural reading order of the prompt.
- Assign a segment to a person only if it primarily describes that individual's
  appearance, actions, pose, or spatial role.
- Scene-level context, environment descriptions, or shared information should
  use tag = 0.

Additional Notes
----------------
- A single person may correspond to multiple non-contiguous segments.
- Do not force overly fine-grained segmentation; segments should remain
  semantically meaningful.
- Do not merge descriptions of different people into a single segment.

Consistency Across Stages
-------------------------
- Tag indices (tag = 1, tag = 2, ...) must remain consistent with the person
  introduction order.
- At stage k, only tags {0, 1, ..., k} may appear.
- Future persons must not appear in either the prompt text or the segment
  annotations.

Each stage prompt and its segmentation should represent a faithful partial
observation of the final scene, aligned with the incremental person-by-person
construction process.

    "output": {
        "levels": [
            {
                # Stage 1 (Level 1)
                # A complete, natural prompt describing the scene
                # assuming ONLY the first person is known to exist.
                # The prompt does NOT need to be written in person-by-person order.
                "caption": "the whole prompt for this stage, stage1 (level1)",

                # Segment annotation is derived AFTER the caption is written.
                # Concatenating all segment texts IN ORDER must EXACTLY
                # reproduce the caption string above.
                "segments": [
                    {
                        # tag = 0 indicates scene-level or shared description
                        # not specific to any single person
                        "tag": 0,
                        "text": "xxx "
                    },
                    {
                        # tag = 1 indicates description specific to
                        # the first person introduced in the scene
                        "tag": 1,
                        "text": "xxx"
                    }
                ]
            },
            {
                # Stage 2 (Level 2)
                # A complete, natural prompt describing the scene
                # assuming ONLY the first and second persons are known to exist.
                "caption": "the whole prompt for this stage, stage2 (level2)",

                "segments": [
                    {
                        "tag": 0,
                        "text": "xxx"
                    },
                    {
                        # description specific to the first person
                        "tag": 1,
                        "text": "xxx"
                    },
                    {
                        # description specific to the second person
                        "tag": 2,
                        "text": "xxxx"
                    }
                ]
            },
            {
                # Stage 3 (Level 3)
                # A complete, natural prompt describing the scene
                # assuming the first three persons are known to exist.
                "caption": "the whole prompt for this stage, stage3 (level3)",

                "segments": [
                    {
                        "tag": 0,
                        "text": "I..."
                    }
                    # tags 1, 2, 3 may appear here if relevant,
                    # but only for persons already introduced
                ]
            }
        ]
    }
}

In addition to the prompts, we also need a Layout Plan (Scene-Level Bounding Boxes)

For each realization, you must provide a single layout plan that defines the
spatial placement of ALL people in the scene.

In addition, you must output a Canvas Size (in pixels) suitable for the scene.
This canvas size is defined ONCE per realization (variant) and shared across all stages.

Canvas Size Requirements:
- Represent as [width, height] in integer pixels.
- The longest side MUST be 1024 pixels.
- The other side MUST be a multiple of 16.
- Choose portrait/landscape/square based on what best fits the scene composition.

Important:
- The layout is defined ONCE per realization.
- The layout is NOT stage-specific.
- If the scene contains N people, the layout must contain exactly N bounding
  boxes, ordered by the person introduction order.

Relationship to stages:
- At stage k, only the first k people are considered visible.
- Therefore, stage k implicitly uses the first k bounding boxes from the layout.
- No separate per-stage layout lists are needed or allowed.

Layout Representation:
- The layout is a list of bounding boxes:
    layout = [bbox_1, bbox_2, ..., bbox_N]
- Each bbox uses normalized coordinates in:
    [x_min, y_min, x_max, y_max]
- Coordinate origin (0,0) is the top-left corner of the canvas.
- All values must be in the range [0.0, 1.0].
- Bounding boxes may be rectangular or square depending on composition.

Bounding box ordering:
- The order of bounding boxes MUST match the person introduction order.
- bbox_1 corresponds to person 1 (introduced at stage 1),
  bbox_2 to person 2 (introduced at stage 2), etc.

Foreground sizing guidelines (encourage larger boxes to avoid squeezed figures):
- 2 people:
  - each bbox ~50–60% width, ~75–100% height
  - slight overlap indicates closeness
- 3–4 people:
  - each bbox ~35–50% width
  - foreground-dominant
- 5+ people:
  - each bbox ~20–30% width
  - arrange evenly or in staggered depth

Overlaps:
- Partial overlaps are allowed when physically plausible
  (e.g., hugging, leaning, passing an item).
- All people must remain visible and identifiable.
- Avoid tiny distant silhouettes; all main figures belong to the foreground.

Grounding Constraint (Critical – Avoid Floating Layouts):
- A common failure mode is bounding boxes that are too small, vertically
  compressed, or appear to float above the bottom of the frame.
- During grounded generation, the model draws the entire person INSIDE the bbox.
  If y_max is far from 1.0, the person may appear shrunken or detached
  from the ground.

Therefore:
- Unless the prompt explicitly describes a distant view, elevated stage,
  seated framing, or intentional upper-body crop, bounding boxes should
  generally extend to the bottom edge of the image:
      y_max ≈ 1.000
- Avoid overly tight vertical crops; include headroom and full foot space.
- Prefer slightly larger, more grounded bounding boxes over small, cramped ones.

Adjustment principles:
- Preserve relative spatial logic (left/right, front/back, near/far).
- Improve grounding and size balance without changing scene semantics.

Failure Mode to note when generating the bbox: Compressed or Floating Layout Bounding Boxes

In many generated variants, the high-level spatial relations between people
(e.g., left vs. right, near vs. far, interacting vs. observing) are logically
correct, but the assigned bounding boxes are too small, vertically compressed,
or appear to float within the frame.

This causes visual artifacts in grounded image generation because the model
draws the entire person INSIDE the bounding box. As a result:
- People may appear smaller than intended.
- Figures may look detached from the ground or scene base.
- Interactions may feel physically implausible due to scale mismatch.

Grounded generation assumes:
- Each bounding box encloses the person’s full visible body region.
- Unless otherwise specified, the person’s feet or lower body should align
  with the bottom of the image.

Therefore:
- Bounding boxes should generally extend to the bottom edge of the image
  (y_max ≈ 1.000) to anchor figures naturally.
- Exceptions are allowed only when the prompt explicitly describes:
  - distant views
  - elevated stages
  - seated groups
  - intentional upper-body framing

Single-person or Non-interaction Prompts

Although this task is designed primarily for multi-person interaction scenes,
the same protocol must also handle prompts that contain:
- a single person, or
- multiple people with little or no explicit interaction.

In such cases, you should still follow the exact same output format:
- produce 5 variants,
- provide a global_prompt per variant,
- generate stage prompts (N stages for N people),
- provide segment annotations per stage,
- and provide a scene-level layout for all people.

Notes:
- If N = 1, there is only one stage (level 1), and segments may use tags {0, 1}.
- If the original prompt contains no interaction, do NOT invent strong interactions
  that contradict the intent. You may introduce light, socially plausible relations
  (e.g., shared attention, co-presence, mild coordinated activity) only when it
  remains consistent with the original prompt.
- For general prompts, diversity can still come from setting, clothing, pose,
  framing, mood, and viewpoint, even if interaction is minimal.

OUTPUT_FORMAT = {
    # Five faithful but potentially diverse realizations of the same original prompt.
    "variants": [
        {
            # Variant index (1 to 5)
            "variant_id": 1,

            # A concise, scene-level prompt describing the full scene.
            # This prompt summarizes the overall situation and human interactions,
            # without enforcing per-person structure or stage-wise constraints.
            # It should be natural, compact, and interaction-focused.
            "global_prompt": (
                e.g., "A tense moment unfolds on a grassy field as players collide in a close physical struggle, "
                "with multiple athletes converging around a loose ball."
            ),

            # Incremental stage-wise prompts.
            # Stage k describes ALL people introduced up to that stage.
            "stages": [
                {
                    # Stage index, starting from 1
                    "level": 1,

                    # A complete, natural-language prompt describing the scene
                    # assuming ONLY the first person is known to exist.
                    "caption": "stage 1 prompt (describes person 1 only)",

                    # Segment-level annotation derived AFTER writing the caption.
                    # Concatenating all segment texts in order must EXACTLY
                    # reproduce `caption`.
                    "segments": [
                        { "tag": 0, "text": "scene-level description " },
                        { "tag": 1, "text": "description of person 1" }
                    ]
                },
                {
                    "level": 2,
                    "caption": "stage 2 prompt (describes person 1 and person 2)",
                    "segments": [
                        { "tag": 0, "text": "scene-level description " },
                        { "tag": 1, "text": "description of person 1 " },
                        { "tag": 2, "text": "description of person 2" }
                    ]
                },
                {
                    "level": 3,
                    "caption": "stage 3 prompt (describes person 1, person 2, and person 3)",
                    "segments": [
                        { "tag": 0, "text": "scene-level description " },
                        { "tag": 1, "text": "description of person 1 " },
                        { "tag": 2, "text": "description of person 2 " },
                        { "tag": 3, "text": "description of person 3" }
                    ]
                }
                # ...
            ],

            # Scene-level layout plan.
            # Defined ONCE per variant and shared across all stages.
            # The number of bboxes equals the total number of people (N).
            # The order of bboxes MUST match the person introduction order.
            "layout": [
                # Person 1 bounding box
                [x1, y1, x2, y2],

                # Person 2 bounding box
                [x1, y1, x2, y2],

                # Person 3 bounding box
                [x1, y1, x2, y2]

                # ...
            ],

            # Canvas size in pixels for this variant.
            # Format: [width, height]
            # Longest side MUST be 1024; the other side MUST be a multiple of 16.
            "canvas_size": [width_px, height_px]
        },

        {
            "variant_id": 2,
            "global_prompt": "concise full-scene interaction-focused prompt",
            "stages": [
                # same structure as variant 1, but a different realization
            ],
            "layout": [
                # scene-level layout for variant 2
            ],
            "canvas_size": [width_px, height_px]
        },

        {
            "variant_id": 3,
            "global_prompt": "concise full-scene interaction-focused prompt",
            "stages": [
                # ...
            ],
            "layout": [
                # ...
            ],
            "canvas_size": [width_px, height_px]
        },

        {
            "variant_id": 4,
            "global_prompt": "concise full-scene interaction-focused prompt",
            "stages": [
                # ...
            ],
            "layout": [
                # ...
            ],
            "canvas_size": [width_px, height_px]
        },

        {
            "variant_id": 5,
            "global_prompt": "concise full-scene interaction-focused prompt",
            "stages": [
                # ...
            ],
            "layout": [
                # ...
            ],
            "canvas_size": [width_px, height_px]
        }
    ]
}
"""


import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The openai package is required to run this script. "
        "Install it with `pip install openai` before executing."
    ) from exc

try:  # openai>=1.0 exposes exceptions at the top level
    from openai import OpenAIError  # type: ignore
except ImportError:  # pragma: no cover - compatibility path
    try:
        from openai.error import OpenAIError  # type: ignore
    except ImportError:  # pragma: no cover
        OpenAIError = Exception  # type: ignore[misc]


DEFAULT_MODEL = "gpt-5.2"
DEFAULT_VARIANTS_PER_PROMPT = 5
DEFAULT_MAX_OUTPUT_TOKENS = 12000


@dataclass(frozen=True)
class GenerationMeta:
    model: str
    variants_per_prompt: int
    instruction_sha256: str
    generated_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    seconds_int = int(seconds + 0.5)
    h = seconds_int // 3600
    m = (seconds_int % 3600) // 60
    s = seconds_int % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def render_progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    frac = max(0.0, min(1.0, done / total))
    filled = int(round(frac * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def build_user_message(prompt: str, variants_per_prompt: int) -> str:
    # Keep this minimal: the main instructions (including schema) live in INSTRUCTION_PROMPT.
    return (
        f"Generic Prompt: {prompt.strip()}\n\n"
        f"Produce exactly {variants_per_prompt} variants.\n"
        "Return ONLY a single JSON object (no markdown, no code fences, no commentary)."
    )


def extract_first_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model response.")

    # Fast path.
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Heuristic: find first balanced top-level {...} block.
    start = stripped.find("{")
    if start < 0:
        raise ValueError("No '{' found in model response.")

    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : idx + 1]
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError("Top-level JSON is not an object.")
                return parsed
    raise ValueError("Unbalanced JSON object in model response.")


def validate_variant_payload(payload: Dict[str, Any], expected_variants: int) -> None:
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError("Missing/invalid 'variants' (expected a list).")
    if len(variants) != expected_variants:
        raise ValueError(f"'variants' length {len(variants)} != expected {expected_variants}.")

    # Light structural validation + exact reconstruction checks.
    for v_idx, variant in enumerate(variants, start=1):
        if not isinstance(variant, dict):
            raise ValueError(f"Variant {v_idx} is not an object.")
        if "global_prompt" not in variant:
            raise ValueError(f"Variant {v_idx} missing 'global_prompt'.")
        stages = variant.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"Variant {v_idx} missing/invalid 'stages'.")
        for stage in stages:
            if not isinstance(stage, dict):
                raise ValueError(f"Variant {v_idx} stage is not an object.")
            caption = stage.get("caption")
            segments = stage.get("segments")
            if not isinstance(caption, str) or not isinstance(segments, list):
                raise ValueError(f"Variant {v_idx} has stage with invalid caption/segments.")
            recon = "".join(seg.get("text", "") for seg in segments if isinstance(seg, dict))
            if recon != caption:
                raise ValueError(f"Variant {v_idx} stage segmentation does not reconstruct caption.")
        layout = variant.get("layout")
        if not isinstance(layout, list) or not layout:
            raise ValueError(f"Variant {v_idx} missing/invalid 'layout'.")

        canvas_size = variant.get("canvas_size")
        if (
            not isinstance(canvas_size, list)
            or len(canvas_size) != 2
            or not all(isinstance(x, int) for x in canvas_size)
        ):
            raise ValueError(f"Variant {v_idx} missing/invalid 'canvas_size' (expected [int, int]).")
        width_px, height_px = canvas_size
        if width_px <= 0 or height_px <= 0:
            raise ValueError(f"Variant {v_idx} has non-positive canvas_size: {canvas_size!r}.")
        longest = max(width_px, height_px)
        shortest = min(width_px, height_px)
        if longest != 1024:
            raise ValueError(
                f"Variant {v_idx} canvas_size longest side must be 1024, got {canvas_size!r}."
            )
        if shortest % 16 != 0:
            raise ValueError(
                f"Variant {v_idx} canvas_size other side must be a multiple of 16, got {canvas_size!r}."
            )


def call_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    force_json: bool,
) -> str:
    # Prefer the Responses API when available; fall back to Chat Completions.
    response_format = {"type": "json_object"} if force_json else None
    if hasattr(client, "responses"):
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = client.responses.create(**kwargs)  # type: ignore[attr-defined]
        except TypeError as err:
            # Some older OpenAI python clients (or proxy clients) don't accept
            # response_format / max_output_tokens. Retry without them.
            msg = str(err)
            retry_kwargs = dict(kwargs)
            if "response_format" in msg and "response_format" in retry_kwargs:
                retry_kwargs.pop("response_format", None)
            if "max_output_tokens" in msg and "max_output_tokens" in retry_kwargs:
                retry_kwargs.pop("max_output_tokens", None)
            resp = client.responses.create(**retry_kwargs)  # type: ignore[attr-defined]
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        # Fallback: try to assemble from output items.
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            chunks: List[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        if getattr(c, "type", None) == "output_text":
                            chunks.append(getattr(c, "text", ""))
            joined = "".join(chunks).strip()
            if joined:
                return joined
        raise ValueError("Could not read text from Responses API response.")

    # Chat Completions fallback.
    kwargs2: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if response_format is not None:
        kwargs2["response_format"] = response_format
    try:
        resp2 = client.chat.completions.create(**kwargs2)  # type: ignore[attr-defined]
    except TypeError as err:
        msg = str(err)
        retry_kwargs2 = dict(kwargs2)
        if "response_format" in msg and "response_format" in retry_kwargs2:
            retry_kwargs2.pop("response_format", None)
        resp2 = client.chat.completions.create(**retry_kwargs2)  # type: ignore[attr-defined]
    return resp2.choices[0].message.content or ""


def generate_for_entry(
    client: OpenAI,
    entry: Dict[str, Any],
    *,
    model: str,
    variants_per_prompt: int,
    temperature: float,
    max_output_tokens: int,
    force_json: bool,
    max_retries: int,
    retry_backoff_s: float,
) -> Tuple[List[Dict[str, Any]], GenerationMeta]:
    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Entry missing valid 'prompt' field.")

    system_prompt = INSTRUCTION_PROMPT.strip()
    user_prompt = build_user_message(prompt, variants_per_prompt)
    instruction_hash = sha256_text(system_prompt)

    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            text = call_openai(
                client,
                model,
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                force_json=force_json,
            )
            payload = extract_first_json_object(text)
            validate_variant_payload(payload, expected_variants=variants_per_prompt)
            variants = payload["variants"]
            meta = GenerationMeta(
                model=model,
                variants_per_prompt=variants_per_prompt,
                instruction_sha256=instruction_hash,
                generated_at_utc=utc_now_iso(),
            )
            return variants, meta
        except (OpenAIError, json.JSONDecodeError, ValueError) as err:
            last_err = err
            if attempt == max_retries:
                break
            jitter = random.random() * 0.25 * retry_backoff_s
            sleep_s = retry_backoff_s * attempt + jitter
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}") from last_err


def iter_entries(dataset: Any) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if not isinstance(dataset, list):
        raise TypeError("Expected the dataset JSON to be a list of objects.")
    for idx, item in enumerate(dataset):
        if isinstance(item, dict):
            yield idx, item


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diversity variants for each entry's `prompt` in a JSON list, "
            "saving incrementally and supporting resume."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("multi-human-testbench/unique_prompts_with_vqa.json"),
        help="Path to input JSON (list of objects with a 'prompt' field).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON. Default: overwrite --input in place.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--variants-per-prompt",
        type=int,
        default=DEFAULT_VARIANTS_PER_PROMPT,
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate even if 'diversity_variants' already exists.",
    )
    parser.add_argument(
        "--force-json",
        action="store_true",
        default=True,
        help="Ask the API to force JSON output when supported (default: on).",
    )
    parser.add_argument(
        "--no-force-json",
        dest="force_json",
        action="store_false",
        help="Disable forced JSON output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N entries (for testing).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Update the progress display every N processed attempts (default: 1).",
    )
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output or input_path

    dataset = load_json(output_path if output_path.exists() else input_path)
    if not isinstance(dataset, list):
        raise SystemExit("Input JSON must be a list of objects.")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    processed_ok = 0
    processed_err = 0
    skipped = 0
    attempts = 0
    start_time = time.time()

    total_entries = len(dataset)
    todo_total = 0
    for _, entry in iter_entries(dataset):
        existing = entry.get("diversity_variants")
        if not args.overwrite_existing and isinstance(existing, list) and len(existing) > 0:
            continue
        todo_total += 1

    print(
        f"[info] total_entries={total_entries} todo={todo_total} "
        f"overwrite_existing={bool(args.overwrite_existing)} model={args.model}"
    )

    for idx, entry in iter_entries(dataset):
        if args.limit is not None and attempts >= args.limit:
            break

        existing = entry.get("diversity_variants")
        if not args.overwrite_existing and isinstance(existing, list) and len(existing) > 0:
            skipped += 1
            continue

        attempts += 1
        try:
            variants, meta = generate_for_entry(
                client,
                entry,
                model=args.model,
                variants_per_prompt=args.variants_per_prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                force_json=args.force_json,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff_s,
            )
        except Exception as err:
            prompt_preview = str(entry.get("prompt", ""))[:120].replace("\n", " ")
            print(f"[error] idx={idx} prompt={prompt_preview!r}: {err}", file=sys.stderr)
            # Save progress so far, then continue.
            atomic_write_json(output_path, dataset)
            processed_err += 1
            if args.progress_every > 0 and (attempts % args.progress_every == 0):
                elapsed = time.time() - start_time
                avg = elapsed / attempts if attempts else 0.0
                remaining = max(todo_total - attempts, 0)
                eta = format_seconds(avg * remaining)
                bar = render_progress_bar(attempts, todo_total)
                msg = (
                    f"\r{bar} {attempts}/{todo_total} "
                    f"ok={processed_ok} err={processed_err} skip={skipped} "
                    f"elapsed={format_seconds(elapsed)} eta={eta}"
                )
                print(msg, end="", file=sys.stderr, flush=True)
            continue

        entry["diversity_variants"] = variants
        entry["diversity_meta"] = {
            "model": meta.model,
            "variants_per_prompt": meta.variants_per_prompt,
            "instruction_sha256": meta.instruction_sha256,
            "generated_at_utc": meta.generated_at_utc,
        }

        atomic_write_json(output_path, dataset)
        processed_ok += 1

        if args.progress_every > 0 and (attempts % args.progress_every == 0):
            elapsed = time.time() - start_time
            avg = elapsed / attempts if attempts else 0.0
            remaining = max(todo_total - attempts, 0)
            eta = format_seconds(avg * remaining)
            bar = render_progress_bar(attempts, todo_total)
            msg = (
                f"\r{bar} {attempts}/{todo_total} "
                f"ok={processed_ok} err={processed_err} skip={skipped} "
                f"elapsed={format_seconds(elapsed)} eta={eta}"
            )
            print(msg, end="", flush=True)

    atomic_write_json(output_path, dataset)
    elapsed = time.time() - start_time
    print(
        f"\n[info] done attempts={attempts} ok={processed_ok} err={processed_err} "
        f"skipped={skipped} elapsed={format_seconds(elapsed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
