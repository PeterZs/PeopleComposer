#!/usr/bin/env python3
"""Binary (Yes/No) alignment decoding for the new benchmark prompts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.utils import (
    ANNOTATIONS_DEFAULT,
    RESULTS_ROOT_DEFAULT,
    RunSpec,
    build_annotation_views,
    collect_run_filters,
    determine_pending_samples,
    discover_run_specs,
    ensure_dir,
    filter_run_specs_by_set,
    load_json,
    load_sample_indices_from_bundle,
    load_sample_runs_from_bundle,
    normalize_prompt_types,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DECODE_SCRIPT = REPO_ROOT / "evaluation_tools" / "run_gpt4o_eval_decoding.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-based yes/no decoding for the new benchmark prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations", type=Path, default=Path("<YOUR_ANNOTATIONS_JSON>"), help="Base annotations file.")
    parser.add_argument("--results-root", type=Path, default=Path("<YOUR_RESULTS_ROOT>"), help="Root directory containing per-model prompt folders.")
    parser.add_argument("--backbone", default="gpt-5", help="OpenAI model used for decoding.")
    parser.add_argument("--api-key", help="API key for the decoding model.")
    parser.add_argument("--prompt-types", default="ABC", help="Subset of prompts to evaluate (e.g., 'A', 'BC', or 'ABC').")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model directories to include.")
    parser.add_argument("--template-ab", default='Does this image show "{}"? Please answer yes or no.', help="Question template for prompts A/B.")
    parser.add_argument("--template-c", default="{}", help="Question template for prompt C (questions already in yes/no form).")
    parser.add_argument("--system-prompt", help="Optional override for the decoder system prompt.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for decoding (omit to use model default).")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per question.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Seconds between retries.")
    parser.add_argument("--image-detail", default="high", help="Image detail level forwarded to the API.")
    parser.add_argument("--model-option", action="append", default=[], metavar="KEY=VALUE", help="Additional --model-option values passed through.")
    parser.add_argument("--human-quality-questions", type=Path, help="Optional JSON file overriding the default human-quality question list.")
    parser.add_argument("--output-dir", type=Path, default=Path("<YOUR_OUTPUT_DIR>"), help="Directory for per-run JSON outputs and summaries.")
    parser.add_argument("--force", action="store_true", help="Re-run evaluations even if outputs already exist.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort immediately if any question fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging noise when invoking the decoder.")
    parser.add_argument("--disable-human-quality", action="store_true", help="Skip the extra human-quality limb checks.")
    parser.add_argument("--run-filter", action="append", default=[], metavar="METHOD:PROMPT", help="Only evaluate specific method/prompt pairs (e.g., 'flux_dev:C').")
    parser.add_argument("--run-filter-file", type=Path, help="JSON file containing METHOD:PROMPT entries.")
    parser.add_argument("--prompts-from", type=Path, help="Optional JSON/list file specifying sample indices to evaluate.")
    return parser.parse_args()


def select_template(args: argparse.Namespace, prompt_type: str) -> str:
    return args.template_c if prompt_type.upper() == "C" else args.template_ab


def format_command(
    args: argparse.Namespace,
    spec: RunSpec,
    annotations_path: Path,
    output_path: Path,
    question_template: str,
    prompts_from: Optional[Path],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(DECODE_SCRIPT),
        "--annotations",
        str(annotations_path),
        "--selected-bundle",
        str(spec.bundle_path),
        "--backbone",
        args.backbone,
        "--output",
        str(output_path),
        "--question-template",
        question_template,
        "--disable-general-question",
        "--max-retries",
        str(args.max_retries),
        "--retry-delay",
        str(args.retry_delay),
        "--image-detail",
        args.image_detail,
    ]
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if spec.run_labels:
        cmd.append("--image-sources")
        cmd.extend(spec.run_labels)
    if args.system_prompt:
        cmd.extend(["--system-prompt", args.system_prompt])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    for opt in args.model_option:
        cmd.extend(["--model-option", opt])
    if args.fail_fast:
        cmd.append("--fail-fast")
    if args.force:
        cmd.append("--overwrite")
    if args.quiet:
        cmd.append("--quiet")
    if not args.disable_human_quality:
        cmd.append("--enable-human-quality")
        if args.human_quality_questions:
            cmd.extend(["--human-quality-questions", str(args.human_quality_questions)])
    if prompts_from is not None:
        cmd.extend(["--prompts-from", str(prompts_from)])
    return cmd


def run_command(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def summarize_yesno(per_sample: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, float]:
    seed_scores: List[float] = []
    sample_scores: List[float] = []
    sample_all_yes: List[float] = []
    human_seed_scores: List[float] = []
    human_sample_scores: List[float] = []
    seed_count = 0
    sample_count = 0

    for finals in per_sample.values():
        per_seed_yes: List[float] = []
        per_seed_all: List[float] = []
        per_seed_human: List[float] = []
        for record in finals.values():
            avg = float(record.get("average_score", 0.0))
            question_count = record.get("question_count")
            if not question_count and record.get("questions"):
                question_count = len(record["questions"])
            yes_count = record.get("yes_count")
            if yes_count is None and question_count:
                yes_count = avg * question_count
            all_yes_flag = (
                1.0 if question_count and yes_count is not None and yes_count == question_count else 0.0
            )
            per_seed_yes.append(avg)
            per_seed_all.append(all_yes_flag)
            seed_scores.append(avg)
            seed_count += 1
            human_score = record.get("human_quality_score")
            if isinstance(human_score, (int, float)):
                value = float(human_score)
                per_seed_human.append(value)
                human_seed_scores.append(value)
        if not per_seed_yes:
            continue
        sample_count += 1
        sample_scores.append(sum(per_seed_yes) / len(per_seed_yes))
        sample_all_yes.append(sum(per_seed_all) / len(per_seed_all))
        if per_seed_human:
            human_sample_scores.append(sum(per_seed_human) / len(per_seed_human))

    mean_seed_yes = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
    mean_sample_yes = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    mean_sample_all = sum(sample_all_yes) / len(sample_all_yes) if sample_all_yes else 0.0
    mean_seed_human = sum(human_seed_scores) / len(human_seed_scores) if human_seed_scores else 0.0
    mean_sample_human = (
        sum(human_sample_scores) / len(human_sample_scores) if human_sample_scores else 0.0
    )
    return {
        "samples": sample_count,
        "seeds": seed_count,
        "mean_seed_yes": mean_seed_yes,
        "mean_sample_yes": mean_sample_yes,
        "mean_sample_all_yes": mean_sample_all,
        "mean_seed_human": mean_seed_human,
        "mean_sample_human": mean_sample_human,
    }


def summarize_output(backbone: str, spec: RunSpec, result_path: Path) -> List[Dict[str, object]]:
    if not result_path.exists():
        return []
    payload = load_json(result_path)
    scores_section = payload.get("scores", {}).get(backbone, {})
    rows: List[Dict[str, object]] = []
    for run_label, per_sample in scores_section.items():
        stats = summarize_yesno(per_sample)
        rows.append(
            {
                "model": spec.method,
                "prompt": spec.prompt_type,
                "run_label": run_label,
                "samples": stats["samples"],
                "seeds": stats["seeds"],
                "mean_seed_yes": stats["mean_seed_yes"],
                "mean_sample_yes": stats["mean_sample_yes"],
                "mean_sample_all_yes": stats["mean_sample_all_yes"],
                "mean_seed_human": stats["mean_seed_human"],
                "mean_sample_human": stats["mean_sample_human"],
                "output": str(result_path),
            }
        )
    return rows


def collect_completed_rows(backbone: str, output_dir: Path) -> Dict[str, List[Dict[str, object]]]:
    per_prompt_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    if not output_dir.exists():
        return per_prompt_rows

    sample_cache: Dict[Path, Tuple[List[str], Dict[str, Tuple[str, ...]]]] = {}
    for path in sorted(output_dir.glob("*__prompt_?.json")):
        if path.name.startswith("summary"):
            continue
        payload = load_json(path)
        meta = payload.get("meta") or {}
        bundle_raw = meta.get("selected_bundle_path")
        if not bundle_raw:
            print(f"[summary-skip] Missing bundle reference in {path}")
            continue
        bundle_path = Path(bundle_raw)
        if not bundle_path.exists():
            print(f"[summary-skip] Bundle path not found for {path}: {bundle_path}")
            continue
        scores_section = payload.get("scores", {}).get(backbone, {})
        if not scores_section:
            continue
        method, prompt_fragment = path.stem.split("__prompt_", 1)
        prompt_type = prompt_fragment.upper()
        run_labels = tuple(sorted(scores_section.keys()))
        spec = RunSpec(
            method=method,
            prompt_type=prompt_type,
            bundle_path=bundle_path,
            run_labels=run_labels,
        )
        if bundle_path not in sample_cache:
            try:
                samples = load_sample_indices_from_bundle(bundle_path)
                sample_runs = load_sample_runs_from_bundle(bundle_path)
                sample_cache[bundle_path] = (samples, sample_runs)
            except Exception as exc:
                print(f"[summary-skip] Failed to load bundle {bundle_path}: {exc}")
                continue
        target_samples, sample_run_map = sample_cache[bundle_path]
        pending = determine_pending_samples(
            path,
            backbone,
            spec.run_labels,
            target_samples,
            force=False,
            sample_runs=sample_run_map,
        )
        if pending:
            complete = len(target_samples) - len(pending)
            total = len(target_samples)
            print(
                f"[summary-pending] {spec.describe()} incomplete ({complete}/{total} samples); skipping."
            )
            continue
        rows = summarize_output(backbone, spec, path)
        for row in rows:
            per_prompt_rows[row["prompt"]].append(row)
    return per_prompt_rows


def render_table(rows: List[Dict[str, object]], prompt: str) -> str:
    header = "| Model | Run Label | Samples | Seeds | Mean Yes Rate | Sample Avg Yes | Sample All-Yes | Human Quality | Output |\n"
    header += "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    body_lines: List[str] = []
    for row in rows:
        body_lines.append(
            "| {model} | {run_label} | {samples} | {seeds} | {mean_seed_yes:.4f} | {mean_sample_yes:.4f} | {mean_sample_all_yes:.4f} | {mean_sample_human:.4f} | `{output}` |".format(
                **row
            )
        )
    if not body_lines:
        body_lines.append("| _No data_ | | | | | | | | |")
    return f"### Prompt {prompt}\n" + header + "\n".join(body_lines) + "\n"


def main() -> None:
    args = parse_args()
    if "{}" not in args.template_ab or "{}" not in args.template_c:
        raise SystemExit("Both question templates must include '{}' placeholder.")

    prompt_types = normalize_prompt_types(args.prompt_types)
    run_filters = collect_run_filters(args.run_filter, args.run_filter_file)
    prompts_from_path: Optional[Path] = None
    prompts_from_list: Optional[List[str]] = None
    if args.prompts_from:
        prompts_from_path = args.prompts_from.resolve()
        if not prompts_from_path.exists():
            raise FileNotFoundError(f"--prompts-from file not found: {prompts_from_path}")
        prompts_from_list = load_json(prompts_from_path)
        if not isinstance(prompts_from_list, list):
            raise ValueError("--prompts-from must point to a JSON list of sample indices.")
        prompts_from_list = [str(idx) for idx in prompts_from_list]
    annotation_views = build_annotation_views(args.annotations, prompt_types)
    discovered_specs = discover_run_specs(
        args.results_root, prompt_types=prompt_types, model_filter=args.models
    )
    run_specs = (
        filter_run_specs_by_set(discovered_specs, run_filters)
        if run_filters
        else list(discovered_specs)
    )
    if not run_specs:
        raise SystemExit("No matching model/prompt runs found under the provided results root.")

    results_dir = args.output_dir
    ensure_dir(results_dir)

    for spec in run_specs:
        annotation_path = annotation_views.get(spec.prompt_type)
        if not annotation_path:
            continue
        output_dir = results_dir / args.backbone
        ensure_dir(output_dir)
        output_path = output_dir / f"{spec.slug}.json"
        template = select_template(args, spec.prompt_type)

        bundle_samples = load_sample_indices_from_bundle(spec.bundle_path)
        sample_run_map = load_sample_runs_from_bundle(spec.bundle_path)
        target_samples = prompts_from_list or bundle_samples
        pending_samples = determine_pending_samples(
            output_path,
            args.backbone,
            spec.run_labels,
            target_samples,
            force=args.force,
            sample_runs=sample_run_map,
        )
        if not pending_samples:
            print(f"[resume] {spec.describe()} already complete; skipping.")
            continue

        pending_file: Optional[Path] = None
        prompts_arg: Optional[Path] = None
        if prompts_from_list is not None and len(pending_samples) == len(target_samples):
            prompts_arg = prompts_from_path
        elif len(pending_samples) != len(target_samples) or prompts_from_list is not None:
            cache_dir = output_dir / "pending_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                delete=False,
                dir=cache_dir,
                prefix=f"{spec.slug}_pending_",
                suffix=".json",
            ) as tmp:
                json.dump(pending_samples, tmp, ensure_ascii=False, indent=2)
                tmp.write("\n")
                pending_file = Path(tmp.name)
            prompts_arg = pending_file

        if args.dry_run:
            cmd = format_command(args, spec, annotation_path, output_path, template, prompts_arg)
            print(f"[dry-run] {' '.join(str(part) for part in cmd)}")
        else:
            if (
                output_path.exists()
                and not args.force
                and prompts_arg is None
                and len(pending_samples) == len(target_samples)
            ):
                print(f"[skip] {spec.describe()} -> {output_path} (already exists)")
            else:
                cmd = format_command(args, spec, annotation_path, output_path, template, prompts_arg)
                print(f"[run] {spec.describe()} -> {output_path}")
                run_command(cmd)
        if pending_file and pending_file.exists():
            pending_file.unlink(missing_ok=True)

    if args.dry_run:
        print("Dry run complete. No summaries generated.")
        return

    summary_dir = results_dir / args.backbone
    ensure_dir(summary_dir)
    per_prompt_rows = collect_completed_rows(args.backbone, summary_dir)

    summary_md_lines = [f"# Yes/No Decoding Summary ({args.backbone})"]
    summary_json: Dict[str, object] = {"backbone": args.backbone, "prompt_types": {}}
    if run_filters:
        summary_json["run_filters"] = [
            f"{method}:{prompt}" for method, prompt in sorted(run_filters)
        ]

    for prompt in sorted(per_prompt_rows.keys()):
        table = render_table(per_prompt_rows[prompt], prompt)
        summary_md_lines.append(table)
        summary_json["prompt_types"][prompt] = per_prompt_rows[prompt]

    summary_md = "\n".join(summary_md_lines)
    summary_dir = results_dir / args.backbone
    ensure_dir(summary_dir)
    summary_md_path = summary_dir / "summary.md"
    summary_json_path = summary_dir / "summary.json"
    summary_md_path.write_text(summary_md, encoding="utf-8")
    write_json(summary_json_path, summary_json)
    print(f"Wrote summary markdown to {summary_md_path}")
    print(f"Wrote summary JSON to {summary_json_path}")


if __name__ == "__main__":
    main()
