#!/usr/bin/env python3
"""Shared helpers for evaluating the new benchmark alignment runs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DEFAULT = REPO_ROOT / "new_benchmark2" / "eval_subset65_refined_w_questions.json"
RESULTS_ROOT_DEFAULT = REPO_ROOT / "selected_exports_new_benchmark2"
DERIVED_ANNOTATIONS_DIR = REPO_ROOT / "eval_newbenchmark2" / "derived_annotations"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = _strip_trailing_commas(text)
        if fixed != text:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        raise


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(f"{path.suffix}.pid{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def _strip_trailing_commas(raw: str) -> str:
    """Remove trailing commas that make otherwise-valid JSON fail to parse."""
    if "," not in raw:
        return raw
    out_chars: List[str] = []
    in_string = False
    escape = False
    quote_char = '"'
    length = len(raw)
    i = 0
    while i < length:
        ch = raw[i]
        if in_string:
            out_chars.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            i += 1
            continue
        if ch in {'"', "'"}:
            in_string = True
            quote_char = ch
            out_chars.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < length and raw[j] in " \t\r\n":
                j += 1
            if j < length and raw[j] in "]}":
                i += 1
                continue
        out_chars.append(ch)
        i += 1
    return "".join(out_chars)


def _sanitize_component(raw: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "annotations"


def _annotation_cache_dir(annotations_path: Path) -> Path:
    resolved = annotations_path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    safe_stem = _sanitize_component(resolved.stem or "annotations")
    return DERIVED_ANNOTATIONS_DIR / f"{safe_stem}_{digest}"


@dataclass(frozen=True)
class RunSpec:
    """Descriptor for a (model, prompt_type) evaluation bundle."""

    method: str
    prompt_type: str
    bundle_path: Path
    run_labels: Tuple[str, ...]

    @property
    def slug(self) -> str:
        return f"{self.method}__prompt_{self.prompt_type.lower()}"

    def describe(self) -> str:
        labels = ", ".join(self.run_labels)
        return f"{self.method} prompt_{self.prompt_type} (runs: {labels})"


def normalize_prompt_types(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["A", "B", "C"]
    normalized: List[str] = []
    for ch in raw.upper():
        if ch in {"A", "B", "C"} and ch not in normalized:
            normalized.append(ch)
    return normalized or ["A", "B", "C"]


def discover_run_specs(
    results_root: Path,
    *,
    prompt_types: Optional[Iterable[str]] = None,
    model_filter: Optional[Iterable[str]] = None,
) -> List[RunSpec]:
    prompt_set = {p.upper() for p in prompt_types} if prompt_types else {"A", "B", "C"}
    model_set = {m for m in model_filter} if model_filter else None
    specs: List[RunSpec] = []

    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
        if model_set and method_name not in model_set:
            continue
        for prompt_type in sorted(prompt_set):
            prompt_dir = method_dir / f"prompt_{prompt_type}"
            bundle = prompt_dir / "selected_bundle.json"
            if not bundle.exists():
                continue
            data = load_json(bundle)
            samples = data.get("samples") or {}
            run_labels: List[str] = []
            for sample_payload in samples.values():
                runs = sample_payload.get("runs") or {}
                for run_label in runs.keys():
                    if run_label not in run_labels:
                        run_labels.append(run_label)
            if not run_labels:
                continue
            specs.append(
                RunSpec(
                    method=method_name,
                    prompt_type=prompt_type,
                    bundle_path=bundle,
                    run_labels=tuple(run_labels),
                )
            )
    return specs


def build_prompt_view(
    annotations: Dict[str, Dict[str, Any]],
    prompt_type: str,
    dest_dir: Path = DERIVED_ANNOTATIONS_DIR,
) -> Path:
    prompt_type = prompt_type.upper()
    payload: Dict[str, Dict[str, Any]] = {}
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / f"annotations_prompt_{prompt_type}.json"

    def _coerce_questions(raw: Optional[Sequence[Any]]) -> List[str]:
        questions: List[str] = []
        if not raw:
            return questions
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                questions.append(text)
        return questions

    for sample_idx, entry in annotations.items():
        general_question = entry.get("general_prompt_question")
        general_str = str(general_question).strip() if general_question else None
        eval_questions: List[str]

        if prompt_type in {"A", "B"}:
            prompt_text = entry.get(f"prompt_{prompt_type}")
            if not prompt_text:
                continue
            eval_questions = [str(prompt_text)]
        elif prompt_type == "C":
            eval_questions = _coerce_questions(entry.get("prompt_C_eval_questions") or [])
            if not eval_questions:
                continue
        else:
            raise ValueError(f"Unsupported prompt type '{prompt_type}'.")

        payload[sample_idx] = {
            "eval_questions": eval_questions,
            "general_prompt_question": general_str,
        }

    if not payload:
        raise ValueError(f"No samples available for prompt_{prompt_type}.")

    write_json(output_path, payload)
    return output_path


def build_annotation_views(
    annotations_path: Path,
    prompt_types: Sequence[str],
) -> Dict[str, Path]:
    annotations = load_json(annotations_path)
    cache_dir = _annotation_cache_dir(annotations_path)
    derived: Dict[str, Path] = {}
    for prompt_type in prompt_types:
        derived[prompt_type] = build_prompt_view(annotations, prompt_type, cache_dir)
    return derived


def summarize_seed_scores(per_sample: Dict[str, Dict[str, Dict[str, Any]]]) -> Tuple[int, int, float, float]:
    """Return (sample_count, seed_count, mean_seed, mean_sample)."""
    sample_avgs: List[float] = []
    seed_scores: List[float] = []
    sample_count = 0
    seed_count = 0
    for sample_idx, finals in per_sample.items():
        seed_values = [
            float(record.get("average_score", 0.0))
            for record in finals.values()
            if isinstance(record, dict)
        ]
        if not seed_values:
            continue
        seed_scores.extend(seed_values)
        seed_count += len(seed_values)
        sample_count += 1
        sample_avgs.append(sum(seed_values) / len(seed_values))

    mean_seed = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
    mean_sample = sum(sample_avgs) / len(sample_avgs) if sample_avgs else 0.0
    return sample_count, seed_count, mean_seed, mean_sample


def _normalize_prompt_label(raw: str) -> str:
    label = raw.strip().upper()
    if label.startswith("PROMPT_"):
        label = label[len("PROMPT_") :]
    return label


def parse_run_filter_string(spec: str) -> Tuple[str, str]:
    normalized = spec.strip()
    if not normalized:
        raise ValueError("Empty run filter entry.")
    for sep in (":", "|", ","):
        if sep in normalized:
            method, prompt = normalized.split(sep, 1)
            break
    else:
        parts = normalized.split("__", 1)
        if len(parts) == 2:
            method, prompt = parts
        else:
            raise ValueError(
                f"Invalid run filter '{spec}'. Expected format 'method:prompt_type'."
            )
    method = method.strip()
    prompt = _normalize_prompt_label(prompt)
    if not method or prompt not in {"A", "B", "C"}:
        raise ValueError(
            f"Invalid run filter '{spec}'. Prompt type must be one of A/B/C."
        )
    return method, prompt


def collect_run_filters(
    values: Sequence[str],
    filter_file: Optional[Path],
) -> Set[Tuple[str, str]]:
    filters: Set[Tuple[str, str]] = set()

    def add_entry(entry: str) -> None:
        method, prompt = parse_run_filter_string(entry)
        filters.add((method, prompt))

    for raw in values:
        if not raw:
            continue
        normalized = str(raw).replace(",", " ")
        tokens = [token.strip() for token in normalized.split() if token.strip()]
        for token in tokens:
            add_entry(token)

    if filter_file:
        payload = load_json(filter_file)
        if isinstance(payload, dict):
            runs = payload.get("runs") or payload.get("run_filters")
        else:
            runs = payload
        if not isinstance(runs, list):
            raise ValueError("Run filter file must contain a list of strings.")
        for entry in runs:
            add_entry(str(entry))
    return filters


def filter_run_specs_by_set(
    run_specs: Sequence[RunSpec],
    allowed: Set[Tuple[str, str]],
) -> List[RunSpec]:
    if not allowed:
        return list(run_specs)
    return [
        spec
        for spec in run_specs
        if (spec.method, spec.prompt_type.upper()) in allowed
    ]


@lru_cache(maxsize=128)
def _load_bundle_sample_metadata(bundle_path_str: str) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    bundle_path = Path(bundle_path_str)
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"No 'samples' dict found in {bundle_path}")

    def sort_key(value: str) -> Tuple[int, str]:
        return (0, int(value)) if str(value).isdigit() else (1, str(value))

    ordered_indices = tuple(sorted((str(idx) for idx in samples.keys()), key=lambda v: sort_key(v)))
    sample_runs: Dict[str, Tuple[str, ...]] = {}
    for sample_idx, sample_payload in samples.items():
        runs = sample_payload.get("runs") or {}
        if not isinstance(runs, dict):
            raise ValueError(f"Sample {sample_idx} in {bundle_path} has invalid 'runs' block.")
        run_labels = tuple(sorted(str(label) for label in runs.keys()))
        sample_runs[str(sample_idx)] = run_labels
    return ordered_indices, sample_runs


def load_sample_indices_from_bundle(bundle_path: Path) -> List[str]:
    indices, _ = _load_bundle_sample_metadata(str(bundle_path.resolve()))
    return list(indices)


def load_sample_runs_from_bundle(bundle_path: Path) -> Dict[str, Tuple[str, ...]]:
    _, sample_runs = _load_bundle_sample_metadata(str(bundle_path.resolve()))
    return {key: tuple(value) for key, value in sample_runs.items()}


def determine_pending_samples(
    output_path: Path,
    backbone: str,
    run_labels: Sequence[str],
    sample_indices: Sequence[str],
    *,
    force: bool,
    sample_runs: Optional[Dict[str, Sequence[str]]] = None,
) -> List[str]:
    normalized_samples = [str(idx) for idx in sample_indices]
    if force or not output_path.exists():
        return normalized_samples

    try:
        data = load_json(output_path)
    except Exception:
        return normalized_samples

    scores_section = data.get("scores", {}).get(backbone, {})
    pending: List[str] = []
    for sample_idx in normalized_samples:
        required_runs: Sequence[str]
        if sample_runs:
            required_runs = sample_runs.get(sample_idx, ())
        else:
            required_runs = ()
        if not required_runs:
            required_runs = run_labels
        if not required_runs:
            pending.append(sample_idx)
            continue
        complete = True
        for run_label in required_runs:
            per_run = scores_section.get(run_label)
            if not isinstance(per_run, dict) or sample_idx not in per_run:
                complete = False
                break
        if not complete:
            pending.append(sample_idx)
    return pending
