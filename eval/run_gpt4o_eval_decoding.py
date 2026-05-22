#!/usr/bin/env python3
"""
Decode VQAScore-style questions with GPT-4o and accumulate yes/no answers.

This helper mirrors the structure of run_vqascore_eval_customized.py but
directly queries a GPT-4o style model to obtain discrete answers for each
question instead of numeric VQAScore probabilities.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - platform without fcntl
    fcntl = None  # type: ignore[assignment]

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - informative runtime guidance
    raise SystemExit(
        "The openai package is required for GPT-4o decoding. "
        "Install it with `pip install openai` before executing."
    ) from exc

try:  # openai>=1.x exposes exceptions from the top level
    from openai import OpenAIError  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - backwards compatibility for 0.x
    try:
        from openai.error import OpenAIError  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - last-resort fallback
        OpenAIError = Exception  # type: ignore[misc,assignment]

from evaluation_tools import run_vqascore_eval as base_eval

REPO_ROOT = base_eval.REPO_ROOT
API_MODEL_MAX_LONG_EDGE = base_eval.API_MODEL_MAX_LONG_EDGE

DEFAULT_ANNOTATIONS = Path("<YOUR_ANNOTATIONS_JSON>")
DEFAULT_SELECTED_BUNDLE = Path("<YOUR_SELECTED_BUNDLE_JSON>")
DEFAULT_OUTPUT = Path("<YOUR_OUTPUT_JSON>")

DEFAULT_QUESTION_TEMPLATE = "In this figure, {} Please answer yes or no."
DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous visual inspector. Examine the provided image and answer "
    "each evaluation question. Start your reply with either 'Yes' or 'No' (case "
    "insensitive), optionally followed by a short justification. Do not respond "
    "with other words before the yes/no indicator."
)

HUMAN_QUALITY_DEFAULT_QUESTIONS = [
    {
        "question": "Look at all people in the image. Does each person have a normal number of arms and legs?",
        "expected": "yes",
    },
    {
        "question": "Do any people in the image appear to have extra or missing limbs?",
        "expected": "no",
    },
    {
        "question": "Does anyone in the image have overlapping or merged body parts that look unnatural?",
        "expected": "no",
    },
    {
        "question": "Are there any duplicated arms, legs, or hands that do not align with a person's body?",
        "expected": "no",
    },
]

YES_TOKENS = {
    "yes",
    "y",
    "yeah",
    "yep",
    "affirmative",
    "correct",
    "true",
    "certainly",
}
NO_TOKENS = {
    "no",
    "n",
    "nope",
    "nah",
    "negative",
    "false",
}
TOKEN_PATTERN = re.compile(r"[a-z]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Waldo & Wenda images by decoding GPT-4o yes/no answers."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Annotations JSON containing eval_questions/general_prompt_question (default: %(default)s)",
    )
    parser.add_argument(
        "--selected-bundle",
        type=Path,
        default=DEFAULT_SELECTED_BUNDLE,
        help="Stage 2 selection bundle JSON with final image paths (default: %(default)s)",
    )
    parser.add_argument(
        "--backbone",
        default="gpt-4o",
        help="OpenAI model name to query for answers (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON file to accumulate decoded answers (default: %(default)s)",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Specific sample indices to evaluate; overrides default selection",
    )
    parser.add_argument(
        "--prompts-from",
        type=Path,
        help="Load sample indices from JSON (list of ints or strings)",
    )
    parser.add_argument(
        "--image-sources",
        nargs="*",
        help="Filter runs by name (exact match or substring). Use 'all' or '*' for every run.",
    )
    parser.add_argument(
        "--question-template",
        default=DEFAULT_QUESTION_TEMPLATE,
        help="Template used to format each question. Must contain '{}'. Default: %(default)s",
    )
    parser.add_argument(
        "--answer-template",
        default=None,
        help="Unused placeholder for parity with VQAScore runner (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--disable-eval-questions",
        action="store_true",
        help="Skip annotation eval_questions.",
    )
    parser.add_argument(
        "--disable-general-question",
        action="store_true",
        help="Skip annotation general_prompt_question.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for GPT-4o (falls back to OPENAI_API_KEY and related env vars).",
    )
    parser.add_argument(
        "--model-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra option forwarded to the OpenAI Responses API (repeatable).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for GPT answers (omit to use the model default).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for a single API call (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay (seconds) between retries (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override the default system instruction passed to GPT.",
    )
    parser.add_argument(
        "--image-detail",
        default="high",
        help="Detail setting forwarded for each input_image content item (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the runs that would be scored and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute answers even when cached values exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging noise.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar even when tqdm is available.",
    )
    parser.add_argument(
        "--print-scores",
        action="store_true",
        help="Print each final/question answer as it is written.",
    )
    parser.add_argument(
        "--print-questions",
        action="store_true",
        help="Include question text when printing answers (implies --print-scores).",
    )
    parser.add_argument(
        "--enable-human-quality",
        action="store_true",
        help="Also ask four fixed human-quality questions about limb realism for each image.",
    )
    parser.add_argument(
        "--human-quality-questions",
        type=Path,
        help="Optional JSON file that overrides the default human-quality questions "
        "(list of objects with 'question' and 'expected' yes/no fields).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately when any question fails instead of skipping the sample.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_json(path: Path, retries: int = 5, delay: float = 0.2) -> Any:
    """Load JSON while tolerating concurrent writers via shared locks/retries."""
    last_error: Optional[json.JSONDecodeError] = None
    for attempt in range(1, retries + 1):
        try:
            with path.open("r", encoding="utf-8") as handle:
                locked = False
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                        locked = True
                    except OSError:
                        locked = False
                raw = handle.read()
                if locked and fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
        except FileNotFoundError:
            raise
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(delay * attempt)
    if last_error:
        raise last_error
    return {}


def parse_sample_indices(args: argparse.Namespace, available: Sequence[str]) -> List[str]:
    if args.prompts:
        return [str(idx) for idx in args.prompts if str(idx) in available]
    if args.prompts_from:
        data = load_json(args.prompts_from)
        if isinstance(data, dict):
            raise ValueError(f"Expected a list in {args.prompts_from}, got dict")
        return [str(idx) for idx in data if str(idx) in available]
    return sorted(available, key=lambda v: int(v) if v.isdigit() else v)


def source_is_selected(run_label: str, filters: Optional[Sequence[str]]) -> bool:
    if not filters:
        return True
    run_lower = run_label.lower()
    for token in filters:
        token_lower = str(token).lower()
        if token_lower in {"all", "*"}:
            return True
        if token_lower == run_lower:
            return True
        if token_lower and token_lower in run_lower:
            return True
    return False


def ensure_nested(container: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    current = container
    for key in keys:
        current = current.setdefault(key, {})
    return current


def lookup_existing(
    results: Dict[str, Any],
    backbone: str,
    run_label: str,
    sample_idx: str,
    final_key: str,
) -> Optional[Dict[str, Any]]:
    current = results.get("scores", {})
    for key in (backbone, run_label, sample_idx):
        current = current.get(key)
        if current is None:
            return None
    return current.get(final_key)


def store_result(
    results: Dict[str, Any],
    backbone: str,
    run_label: str,
    sample_idx: str,
    final_key: str,
    payload: Dict[str, Any],
) -> None:
    branch = ensure_nested(results.setdefault("scores", {}), [backbone, run_label, sample_idx])
    branch[final_key] = payload


def final_identifier(seed_id: Optional[str], image_path: str) -> str:
    digest = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:8]
    parts = []
    if seed_id not in (None, ""):
        parts.append(f"seed:{seed_id}")
    parts.append(f"img:{digest}")
    return "|".join(parts)


def resolve_final_image(seed_entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    candidates: List[str] = []
    final_image = seed_entry.get("final_image")
    if final_image:
        candidates.append(str(final_image))
    source_paths = seed_entry.get("source_paths") or {}
    maybe_final = source_paths.get("final")
    if maybe_final:
        candidates.append(str(maybe_final))

    for candidate in candidates:
        resolved = Path(candidate)
        if not resolved.is_absolute():
            resolved = (REPO_ROOT / candidate).resolve()
        if resolved.exists():
            try:
                relative = str(resolved.relative_to(REPO_ROOT))
            except ValueError:
                relative = None
            return str(resolved), relative
    return None, None


def build_question_entries(
    annotation: Dict[str, Any],
    include_eval: bool,
    include_general: bool,
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []

    if include_eval:
        eval_questions = annotation.get("eval_questions") or []
        if isinstance(eval_questions, list):
            for idx, raw_question in enumerate(eval_questions):
                if raw_question is None:
                    continue
                original = str(raw_question).strip()
                if not original:
                    continue
                sanitized, _ = base_eval.sanitize_caption_text(original)
                sanitized = sanitized.strip()
                if not sanitized:
                    continue
                escaped = sanitized.replace("{", "{{").replace("}", "}}")
                questions.append(
                    {
                        "key": f"eval:{idx}",
                        "kind": "eval",
                        "index": idx,
                        "question": original,
                        "question_sanitized": sanitized,
                        "question_for_template": escaped,
                    }
                )

    if include_general:
        general = annotation.get("general_prompt_question")
        if general:
            original = str(general).strip()
            if original:
                sanitized, _ = base_eval.sanitize_caption_text(original)
                sanitized = sanitized.strip()
                if sanitized:
                    escaped = sanitized.replace("{", "{{").replace("}", "}}")
                    questions.append(
                        {
                            "key": "general",
                            "kind": "general",
                            "index": 0,
                            "question": original,
                            "question_sanitized": sanitized,
                            "question_for_template": escaped,
                        }
                    )

    return questions


def normalize_expected_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("y"):
        return "yes"
    return "no"


def load_human_quality_question_set(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        source = HUMAN_QUALITY_DEFAULT_QUESTIONS
    else:
        raw = load_json(path)
        if not isinstance(raw, list):
            raise ValueError("Human-quality question file must contain a list.")
        source = raw

    question_set: List[Dict[str, str]] = []
    for entry in source:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question", "")).strip()
        if not question:
            continue
        expected = normalize_expected_answer(entry.get("expected", "yes"))
        question_set.append({"question": question, "expected": expected})

    if not question_set:
        raise ValueError("Human-quality question list is empty.")
    return question_set


def evaluate_human_quality(
    *,
    question_set: Sequence[Dict[str, str]],
    client: Any,
    model: str,
    system_prompt: str,
    image_path: str,
    image_b64: str,
    image_detail: str,
    temperature: Optional[float],
    max_retries: int,
    retry_delay: float,
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    matches_all = True
    for idx, entry in enumerate(question_set):
        question_text = entry["question"]
        expected = entry["expected"]
        prompt_text = f"{question_text} Please answer yes or no."
        response_text = call_model_for_answer(
            client,
            model=model,
            system_prompt=system_prompt,
            question_text=prompt_text,
            image_path=image_path,
            image_b64=image_b64,
            image_detail=image_detail,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            extra_kwargs=extra_kwargs,
        )
        answer_lower = response_text.strip().lower()
        parsed = parse_yes_no_answer(answer_lower)
        match = parsed == expected
        if parsed == "unknown":
            match = False
        if not match:
            matches_all = False
        details.append(
            {
                "index": idx,
                "question": question_text,
                "expected": expected,
                "raw_answer": response_text.strip(),
                "parsed_answer": parsed,
                "match": match,
            }
        )

    score = 1.0 if matches_all else 0.0
    return {"score": score, "questions": details}


def encode_image_base64(image_path: str, cache: Dict[str, str]) -> str:
    cached = cache.get(image_path)
    if cached:
        return cached
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    cache[image_path] = encoded
    return encoded


def build_image_content(image_path: str, encoded: str, detail: str) -> Dict[str, Any]:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        suffix = Path(image_path).suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".gif":
            mime = "image/gif"
        elif suffix == ".bmp":
            mime = "image/bmp"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            mime = "image/png"
    data_url = f"data:{mime};base64,{encoded}"
    payload: Dict[str, Any] = {"type": "input_image", "image_url": data_url}
    if detail:
        payload["detail"] = detail
    return payload


def extract_text_from_response(response: Any) -> Optional[str]:
    if hasattr(response, "output_text"):
        output_text = getattr(response, "output_text")
        if isinstance(output_text, list):
            joined = "\n".join(str(part) for part in output_text).strip()
            if joined:
                return joined
        elif isinstance(output_text, str):
            stripped = output_text.strip()
            if stripped:
                return stripped

    payload: Optional[Dict[str, Any]] = None
    if hasattr(response, "model_dump"):
        try:
            payload = response.model_dump()
        except Exception:
            payload = None
    elif isinstance(response, dict):
        payload = response

    if payload:
        outputs = payload.get("output") or payload.get("outputs") or []
        collected: List[str] = []
        for item in outputs:
            contents = item.get("content", [])
            for content in contents:
                content_type = content.get("type")
                if content_type in {"output_text", "text"}:
                    text = content.get("text")
                    if text:
                        collected.append(str(text))
        if collected:
            return "\n".join(collected).strip()

        content = payload.get("content")
        if isinstance(content, list):
            collected = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    text = part.get("text")
                    if text:
                        collected.append(str(text))
            if collected:
                return "\n".join(collected).strip()

    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    collected = []
                    for part in content:
                        if part.get("type") == "text":
                            text = part.get("text")
                            if text:
                                collected.append(str(text))
                    if collected:
                        return "\n".join(collected).strip()
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text.strip()
    return None


def call_model_for_answer(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    question_text: str,
    image_path: str,
    image_b64: str,
    image_detail: str,
    temperature: Optional[float],
    max_retries: int,
    retry_delay: float,
    extra_kwargs: Dict[str, Any],
) -> str:
    user_prompt = (
        f"{question_text}\n\nRespond with 'Yes' or 'No' first, followed by an optional, "
        "very short justification."
    )
    for attempt in range(1, max_retries + 1):
        try:
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            build_image_content(image_path, image_b64, image_detail),
                            {"type": "input_text", "text": user_prompt},
                        ],
                    },
                ],
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            if extra_kwargs:
                create_kwargs.update(extra_kwargs)
            response = client.responses.create(**create_kwargs)
        except OpenAIError as error:
            if attempt == max_retries:
                raise
            time.sleep(retry_delay * attempt)
            continue
        text = extract_text_from_response(response)
        if text is None or not text.strip():
            raise RuntimeError("Model response did not contain textual content.")
        return text.strip()
    raise RuntimeError("Exceeded maximum retries without receiving a valid response.")


def parse_yes_no_answer(answer_text: str) -> str:
    tokens = TOKEN_PATTERN.findall(answer_text.lower())
    if tokens:
        first = tokens[0]
        if first in YES_TOKENS:
            return "yes"
        if first in NO_TOKENS:
            return "no"
    for token in tokens:
        if token in YES_TOKENS:
            return "yes"
        if token in NO_TOKENS:
            return "no"
    return "unknown"


def summarize(results: Dict[str, Any], backbone: str) -> Dict[str, Tuple[float, int]]:
    summary: Dict[str, Tuple[float, int]] = {}
    scores_section = results.get("scores", {}).get(backbone, {})
    for run_label, per_run in scores_section.items():
        values: List[float] = []
        for per_sample in per_run.values():
            for record in per_sample.values():
                avg = record.get("average_score")
                if isinstance(avg, (int, float)):
                    values.append(float(avg))
        if values:
            summary[run_label] = (sum(values) / len(values), len(values))
    return summary


def compute_overall_stats(results: Dict[str, Any], backbone: str) -> Optional[Dict[str, float]]:
    yes_total = 0.0
    question_total = 0
    human_total = 0.0
    human_count = 0
    scores_section = results.get("scores", {}).get(backbone, {})
    for per_run in scores_section.values():
        for per_sample in per_run.values():
            for record in per_sample.values():
                yes_count = record.get("yes_count")
                question_count = record.get("question_count")
                if isinstance(question_count, int) and question_count > 0:
                    if isinstance(yes_count, (int, float)):
                        yes_total += float(yes_count)
                        question_total += question_count
                        continue
                questions = record.get("questions", [])
                if isinstance(questions, list):
                    question_total += len(questions)
                    yes_total += sum(
                        1.0
                        for entry in questions
                        if str(entry.get("parsed_answer", "")).lower() == "yes"
                        or (entry.get("score") == 1.0)
                    )
                human_score = record.get("human_quality_score")
                if isinstance(human_score, (int, float)):
                    human_total += float(human_score)
                    human_count += 1
    if question_total:
        summary = {
            "yes_count": yes_total,
            "question_count": float(question_total),
            "average": yes_total / question_total,
        }
        if human_count:
            summary["human_quality_total"] = human_total
            summary["human_quality_count"] = float(human_count)
            summary["human_quality_average"] = human_total / human_count
        return summary
    return None


def main() -> None:
    args = parse_args()
    if args.print_questions:
        args.print_scores = True

    if "{}" not in args.question_template:
        raise ValueError("Question template must contain '{}' placeholder.")
    if args.answer_template and not args.quiet:
        print(
            "Note: --answer-template is ignored in GPT decoding mode; answers are direct model outputs.",
            file=sys.stderr,
        )

    annotations = load_json(args.annotations)
    bundle = load_json(args.selected_bundle)
    samples = bundle.get("samples") or {}
    if not isinstance(samples, dict):
        raise ValueError("selected_bundle samples must be a dict keyed by sample index.")

    available_samples = [key for key in samples.keys() if key in annotations]
    if not available_samples:
        raise ValueError("No overlapping samples between annotations and selected bundle.")

    sample_indices = parse_sample_indices(args, available_samples)
    if not sample_indices:
        raise ValueError("No sample indices selected for evaluation.")

    results_path = args.output
    if results_path.exists():
        results_payload = load_json(results_path)
    else:
        results_payload = {"meta": {}}

    model_options = base_eval.parse_model_options(args.model_option)
    api_key = args.api_key or model_options.pop("api_key", None)
    if not api_key:
        for env_name in base_eval.POSSIBLE_API_KEY_ENVS:
            candidate = os.getenv(env_name)
            if candidate:
                api_key = candidate
                break

    temperature: Optional[float] = args.temperature
    if "temperature" in model_options:
        try:
            temperature = float(model_options.pop("temperature"))
        except (TypeError, ValueError):
            raise ValueError("Invalid temperature value provided via --model-option.")

    max_retries = args.max_retries
    if "max_retries" in model_options:
        try:
            max_retries = int(model_options.pop("max_retries"))
        except (TypeError, ValueError):
            raise ValueError("Invalid max_retries value provided via --model-option.")

    retry_delay = args.retry_delay
    if "retry_delay" in model_options:
        try:
            retry_delay = float(model_options.pop("retry_delay"))
        except (TypeError, ValueError):
            raise ValueError("Invalid retry_delay value provided via --model-option.")

    system_prompt = args.system_prompt or model_options.pop("system_prompt", DEFAULT_SYSTEM_PROMPT)

    allowed_response_kwargs = {
        "max_output_tokens",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "metadata",
        "seed",
        "response_format",
    }
    response_kwargs: Dict[str, Any] = {}
    for key in list(model_options.keys()):
        if key in allowed_response_kwargs:
            response_kwargs[key] = model_options.pop(key)

    if model_options and not args.quiet:
        ignored = ", ".join(sorted(model_options.keys()))
        print(
            f"Ignoring unrecognized --model-option keys for Responses API: {ignored}",
            file=sys.stderr,
        )

    results_payload.setdefault("meta", {})
    meta_updates = {
        "annotations_path": str(args.annotations),
        "selected_bundle_path": str(args.selected_bundle),
        "question_template": args.question_template,
        "include_eval_questions": not args.disable_eval_questions,
        "include_general_question": not args.disable_general_question,
        "backbone": args.backbone,
        "temperature": temperature,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
        "system_prompt": system_prompt,
        "image_detail": args.image_detail,
        "last_run": now_utc_iso(),
    }
    if args.image_sources:
        meta_updates["image_source_filters"] = list(args.image_sources)
    meta_updates["print_questions"] = args.print_questions
    human_quality_questions: List[Dict[str, str]] = []
    if args.enable_human_quality:
        human_quality_questions = load_human_quality_question_set(args.human_quality_questions)
        meta_updates["human_quality_enabled"] = True
        meta_updates["human_quality_question_count"] = len(human_quality_questions)
        if args.human_quality_questions:
            meta_updates["human_quality_questions_path"] = str(args.human_quality_questions)
    else:
        meta_updates["human_quality_enabled"] = False
    results_payload["meta"].update(meta_updates)

    pending: List[Dict[str, Any]] = []
    failures: List[str] = []

    for sample_idx in sample_indices:
        annotation = annotations.get(sample_idx)
        if not annotation:
            failures.append(f"Sample {sample_idx}: missing annotation entry.")
            continue
        question_entries = build_question_entries(
            annotation,
            include_eval=not args.disable_eval_questions,
            include_general=not args.disable_general_question,
        )
        if not question_entries:
            failures.append(f"Sample {sample_idx}: no questions available after filtering.")
            continue

        sample_payload = samples.get(sample_idx)
        if not sample_payload:
            failures.append(f"Sample {sample_idx}: missing selection data.")
            continue
        runs = sample_payload.get("runs") or {}
        if not isinstance(runs, dict):
            failures.append(f"Sample {sample_idx}: unexpected runs structure.")
            continue

        for run_label, run_payload in runs.items():
            if not source_is_selected(run_label, args.image_sources):
                continue
            seeds = run_payload.get("seeds") or []
            if not isinstance(seeds, list):
                failures.append(f"Sample {sample_idx}: run '{run_label}' seeds malformed.")
                continue
            for seed_entry in seeds:
                final_image, relative_image = resolve_final_image(seed_entry)
                if not final_image:
                    failures.append(
                        f"Sample {sample_idx}: run '{run_label}' seed {seed_entry.get('seed_id')} missing final image."
                    )
                    continue
                seed_id = seed_entry.get("seed_id")
                final_key = final_identifier(seed_id, final_image)
                existing = lookup_existing(
                    results_payload, args.backbone, run_label, sample_idx, final_key
                )
                if existing and not args.overwrite:
                    existing_keys = [q.get("key") for q in existing.get("questions", [])]
                    expected_keys = [q["key"] for q in question_entries]
                    if existing_keys == expected_keys:
                        continue
                pending.append(
                    {
                        "sample_index": sample_idx,
                        "run_label": run_label,
                        "seed_id": seed_id,
                        "final_key": final_key,
                        "final_image": final_image,
                        "relative_image": relative_image,
                        "questions": question_entries,
                    }
                )

    if not pending and not args.dry_run:
        results_payload = base_eval.locked_json_update(results_path, meta_updates, lambda _: None)
        if not args.quiet:
            print("Nothing to evaluate (all answers cached).")
            if failures:
                print("Issues encountered:")
                for msg in failures:
                    print(f"  - {msg}")
        return

    if args.dry_run:
        print(f"Planned evaluations: {len(pending)} final images")
        for item in pending:
            print(
                f"[{item['run_label']}] sample {item['sample_index']} seed {item.get('seed_id')} "
                f"-> {len(item['questions'])} questions"
            )
            for question in item["questions"]:
                formatted = args.question_template.format(question["question_for_template"])
                print(f"    - {question['key']}: {question['question']} -> {formatted}")
        if failures:
            print("Issues detected during setup:")
            for msg in failures:
                print(f"  - {msg}")
        return

    if not api_key:
        raise RuntimeError(
            "An OpenAI-compatible API key is required. Provide --api-key, "
            "--model-option api_key=..., or set an environment variable such as OPENAI_API_KEY."
        )
    client = OpenAI(api_key=api_key)

    needs_resize = base_eval.backbone_requires_long_edge_resize(args.backbone)
    resize_context = TemporaryDirectory() if needs_resize else nullcontext(None)

    image_resize_cache: Dict[str, str] = {}
    image_b64_cache: Dict[str, str] = {}

    if not args.quiet:
        print(
            f"Decoding {len(pending)} final images with model '{args.backbone}' "
            f"using template '{args.question_template}'"
        )
        if needs_resize:
            print(
                f"Applying longest-edge {API_MODEL_MAX_LONG_EDGE}px resize before encoding images."
            )

    with resize_context as resize_tmp:
        resize_dir = Path(resize_tmp) if resize_tmp else None
        progress_bar = None
        if not args.quiet and not args.no_progress and tqdm is not None and pending:
            progress_bar = tqdm(total=len(pending), desc="Decoding finals", unit="image")

        for item in pending:
            try:
                eval_image = base_eval.prepare_eval_image_path(
                    item["final_image"],
                    needs_resize,
                    resize_dir,
                    image_resize_cache,
                    API_MODEL_MAX_LONG_EDGE,
                )
            except Exception as exc:
                failures.append(
                    f"Sample {item['sample_index']}: failed to prepare image '{item['final_image']}': {exc}"
                )
                if progress_bar is not None:
                    progress_bar.update(1)
                continue

            encoded_image = encode_image_base64(eval_image, image_b64_cache)

            question_payload: List[Dict[str, Any]] = []
            yes_count = 0
            error_occurred = False

            for question in item["questions"]:
                prompt_text = args.question_template.format(question["question_for_template"])
                try:
                    response_text = call_model_for_answer(
                        client,
                        model=args.backbone,
                        system_prompt=system_prompt,
                        question_text=prompt_text,
                        image_path=eval_image,
                        image_b64=encoded_image,
                        image_detail=args.image_detail,
                        temperature=temperature,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        extra_kwargs=response_kwargs,
                    )
                except Exception as exc:
                    failure_msg = (
                        f"Sample {item['sample_index']}: run '{item['run_label']}' seed {item.get('seed_id')} "
                        f"question {question['key']} failed: {exc}"
                    )
                    failures.append(failure_msg)
                    if not args.quiet:
                        print(f"[error] {failure_msg}", file=sys.stderr)
                    if args.fail_fast:
                        raise
                    error_occurred = True
                    break

                answer_lower = response_text.strip().lower()
                parsed = parse_yes_no_answer(answer_lower)
                score_value = 1.0 if parsed == "yes" else 0.0
                yes_count += 1 if parsed == "yes" else 0

                question_payload.append(
                    {
                        "key": question["key"],
                        "kind": question["kind"],
                        "index": question["index"],
                        "question": question["question"],
                        "question_sanitized": question["question_sanitized"],
                        "prompt_text": prompt_text,
                        "raw_answer": response_text.strip(),
                        "answer_lower": answer_lower,
                        "parsed_answer": parsed,
                        "score": score_value,
                    }
                )

                if (args.print_scores or args.print_questions) and not args.quiet:
                    print(
                        f"[{item['run_label']}] sample {item['sample_index']} seed {item.get('seed_id')} "
                        f"{question['key']} -> {parsed.upper()} (score {score_value:.3f})"
                    )
                    if args.print_questions and not args.quiet:
                        print(f"      question: {question['question']}")
                        print(f"      answer: {response_text.strip()}")

            if error_occurred or not question_payload:
                if progress_bar is not None:
                    progress_bar.update(1)
                continue

            human_quality_result: Optional[Dict[str, Any]] = None
            if args.enable_human_quality and human_quality_questions:
                try:
                    human_quality_result = evaluate_human_quality(
                        question_set=human_quality_questions,
                        client=client,
                        model=args.backbone,
                        system_prompt=system_prompt,
                        image_path=eval_image,
                        image_b64=encoded_image,
                        image_detail=args.image_detail,
                        temperature=temperature,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        extra_kwargs=response_kwargs,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    failure_msg = (
                        f"Sample {item['sample_index']}: run '{item['run_label']}' "
                        f"seed {item.get('seed_id')} human-quality check failed: {exc}"
                    )
                    failures.append(failure_msg)
                    if not args.quiet:
                        print(f"[error] {failure_msg}", file=sys.stderr)
                    human_quality_result = {
                        "score": 0.0,
                        "questions": [],
                        "error": str(exc),
                    }

            average_score = yes_count / len(question_payload) if question_payload else 0.0
            now_iso = now_utc_iso()

            payload = {
                "evaluated_at": now_iso,
                "backbone": args.backbone,
                "run_label": item["run_label"],
                "sample_index": item["sample_index"],
                "seed_id": item.get("seed_id"),
                "final_image": item["relative_image"] or item["final_image"],
                "absolute_image": item["final_image"],
                "question_template": args.question_template,
                "questions": question_payload,
                "average_score": average_score,
                "question_count": len(question_payload),
                "yes_count": yes_count,
            }
            if human_quality_result is not None:
                payload["human_quality"] = human_quality_result
                payload["human_quality_score"] = float(human_quality_result.get("score", 0.0))

            meta_updates["last_run"] = now_iso
            results_payload.setdefault("meta", {}).update(meta_updates)
            store_result(
                results_payload,
                args.backbone,
                item["run_label"],
                item["sample_index"],
                item["final_key"],
                payload,
            )
            results_payload = base_eval.locked_json_update(
                results_path,
                meta_updates,
                lambda data, payload=payload, item=item: store_result(
                    data,
                    args.backbone,
                    item["run_label"],
                    item["sample_index"],
                    item["final_key"],
                    payload,
                ),
            )

            if progress_bar is not None:
                progress_bar.update(1)
            elif not args.quiet:
                print(
                    f"Decoded sample {item['sample_index']} seed {item.get('seed_id')} "
                    f"({len(question_payload)} questions, yes-rate {average_score:.4f})"
                )

        if progress_bar is not None:
            progress_bar.close()

    overall_stats = compute_overall_stats(results_payload, args.backbone)
    if overall_stats is not None:
        meta_with_overall = dict(meta_updates)
        meta_with_overall.update(
            {
                "overall_yes_count": overall_stats["yes_count"],
                "overall_question_count": overall_stats["question_count"],
                "overall_average_score": overall_stats["average"],
            }
        )
        if "human_quality_average" in overall_stats:
            meta_with_overall.update(
                {
                    "overall_human_quality_score": overall_stats["human_quality_average"],
                    "overall_human_quality_count": overall_stats["human_quality_count"],
                }
            )
        results_payload.setdefault("meta", {}).update(meta_with_overall)
        results_payload = base_eval.locked_json_update(
            results_path,
            meta_with_overall,
            lambda data, meta=meta_with_overall: data.setdefault("meta", {}).update(meta),
        )

    if not args.quiet:
        print(f"Updated {results_path}")
        if failures:
            print("Issues encountered during decoding:")
            for msg in failures:
                print(f"  - {msg}")

        summary = summarize(results_payload, args.backbone)
        if summary:
            print("Average yes-rate (per run):")
            for run_label, (mean_score, count) in summary.items():
                print(f"  {run_label}: mean={mean_score:.4f} over {count} finals")
        if overall_stats is not None:
            total_yes = overall_stats["yes_count"]
            total_questions = overall_stats["question_count"]
            print(
                f"Overall yes-rate: {total_yes:.0f}/{total_questions:.0f} "
                f"= {overall_stats['average']:.4f}"
            )


if __name__ == "__main__":
    main()


'''
python evaluation_tools/run_gpt4o_eval_decoding.py --backbone gpt-4o --api-key <YOUR_API_KEY> --disable-general-question

'''
