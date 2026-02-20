#!/usr/bin/env python
"""Step 3: compare full aggregated Basic Pitch outputs between TensorFlow and CoreML."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple


def require_python_compat() -> None:
    if sys.version_info >= (3, 12):
        print(
            "Warning: Python 3.12+ detected. Basic Pitch dependencies may fail. "
            "Use Python 3.10 or 3.11 for reliable setup."
        )


def map_coreml_outputs(raw_outputs: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    mapped: Dict[str, Any] = {}
    assigned_88: List[str] = []
    for raw_name, raw_value in raw_outputs.items():
        arr = np.asarray(raw_value, dtype=np.float32)
        lower = raw_name.lower()

        if "onset" in lower or raw_name == "Identity_2":
            mapped["onset"] = arr
            continue
        if "contour" in lower or raw_name == "Identity":
            mapped["contour"] = arr
            continue
        if "note" in lower or raw_name == "Identity_1":
            mapped["note"] = arr
            continue

        if arr.ndim >= 3 and arr.shape[-1] == 264:
            mapped["contour"] = arr
            continue
        if arr.ndim >= 3 and arr.shape[-1] == 88:
            assigned_88.append(raw_name)

    if "note" not in mapped or "onset" not in mapped:
        if len(assigned_88) == 2:
            first = np.asarray(raw_outputs[assigned_88[0]], dtype=np.float32)
            second = np.asarray(raw_outputs[assigned_88[1]], dtype=np.float32)
            mapped["note"] = first
            mapped["onset"] = second
        elif len(assigned_88) == 1:
            mapped.setdefault("note", np.asarray(raw_outputs[assigned_88[0]], dtype=np.float32))

    missing = [k for k in ("note", "onset", "contour") if k not in mapped]
    if missing:
        raise RuntimeError(f"Unable to map CoreML outputs. Missing: {missing}. Raw keys: {list(raw_outputs.keys())}")
    return mapped


def run_full_pipeline_tf(audio_path: Path, model_path: Path, overlap_frames: int) -> Tuple[Dict[str, Any], Dict[str, int]]:
    import numpy as np
    from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
    from basic_pitch.inference import Model, get_audio_input, unwrap_output

    overlap_len = overlap_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    model = Model(model_path)
    output: Dict[str, List[Any]] = {"note": [], "onset": [], "contour": []}
    audio_original_length = 0
    num_windows = 0
    for audio_windowed, _, this_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        pred = model.predict(audio_windowed.astype(np.float32))
        for key in output.keys():
            output[key].append(np.asarray(pred[key], dtype=np.float32))
        audio_original_length = this_original_length
        num_windows += 1

    if num_windows == 0:
        raise RuntimeError("No audio windows were generated for TensorFlow pipeline.")

    aggregated = {
        key: unwrap_output(np.concatenate(values, axis=0), audio_original_length, overlap_frames, hop_size).astype(
            np.float32
        )
        for key, values in output.items()
    }
    meta = {
        "num_windows": num_windows,
        "audio_original_length_samples": int(audio_original_length),
        "overlap_len_samples": int(overlap_len),
        "hop_size_samples": int(hop_size),
    }
    return aggregated, meta


def run_full_pipeline_coreml(
    audio_path: Path, coreml_model_path: Path, overlap_frames: int, compute_unit_name: str
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    import numpy as np
    import coremltools as ct
    from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
    from basic_pitch.inference import get_audio_input, unwrap_output

    overlap_len = overlap_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    compute_unit = getattr(ct.ComputeUnit, compute_unit_name)
    mlmodel = ct.models.MLModel(str(coreml_model_path), compute_units=compute_unit)
    spec = mlmodel.get_spec()
    if not spec.description.input:
        raise RuntimeError("CoreML model has no inputs.")
    input_name = spec.description.input[0].name

    output: Dict[str, List[Any]] = {"note": [], "onset": [], "contour": []}
    audio_original_length = 0
    num_windows = 0
    for audio_windowed, _, this_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        raw_pred = mlmodel.predict({input_name: audio_windowed.astype(np.float32)})
        pred = map_coreml_outputs(raw_pred)
        for key in output.keys():
            output[key].append(np.asarray(pred[key], dtype=np.float32))
        audio_original_length = this_original_length
        num_windows += 1

    if num_windows == 0:
        raise RuntimeError(f"No audio windows were generated for CoreML pipeline ({compute_unit_name}).")

    aggregated = {
        key: unwrap_output(np.concatenate(values, axis=0), audio_original_length, overlap_frames, hop_size).astype(
            np.float32
        )
        for key, values in output.items()
    }
    meta = {
        "num_windows": num_windows,
        "audio_original_length_samples": int(audio_original_length),
        "overlap_len_samples": int(overlap_len),
        "hop_size_samples": int(hop_size),
        "execution_mode": compute_unit_name,
    }
    return aggregated, meta


def probe_cpu_only_predict(coreml_model_path: Path) -> Dict[str, Any]:
    code = textwrap.dedent(
        f"""
        import numpy as np
        import coremltools as ct
        model = ct.models.MLModel({str(coreml_model_path)!r}, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        x = np.zeros((1, 43844, 1), dtype=np.float32)
        _ = model.predict({{input_name: x}})
        print("CPU_ONLY_OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode == 0 and "CPU_ONLY_OK" in proc.stdout:
        return {"ok": True}

    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-8:]) if proc.stderr else ""
    stdout_tail = "\n".join(proc.stdout.strip().splitlines()[-8:]) if proc.stdout else ""
    signal_hint = ""
    if proc.returncode < 0:
        signal_hint = f"terminated by signal {-proc.returncode}"
    return {
        "ok": False,
        "returncode": proc.returncode,
        "signal_hint": signal_hint,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def compare_outputs(
    tf_outputs: Dict[str, Any], coreml_outputs: Dict[str, Any], threshold: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import numpy as np

    per_output: Dict[str, Any] = {}
    aligned: Dict[str, Dict[str, Any]] = {}
    for key in ("note", "onset", "contour"):
        tf_arr = np.asarray(tf_outputs[key], dtype=np.float32)
        cm_arr = np.asarray(coreml_outputs[key], dtype=np.float32)

        common_time = min(tf_arr.shape[0], cm_arr.shape[0])
        common_freq = min(tf_arr.shape[1], cm_arr.shape[1])
        tf_trim = tf_arr[:common_time, :common_freq]
        cm_trim = cm_arr[:common_time, :common_freq]
        diff = np.abs(tf_trim - cm_trim)
        per_output[key] = {
            "tf_shape": list(tf_arr.shape),
            "coreml_shape": list(cm_arr.shape),
            "aligned_shape": [int(common_time), int(common_freq)],
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
        }
        aligned[key] = {"tf": tf_trim, "coreml": cm_trim}

    tf_note = aligned["note"]["tf"]
    cm_note = aligned["note"]["coreml"]
    tf_note_active = tf_note >= threshold
    cm_note_active = cm_note >= threshold
    activation_flips = np.logical_xor(tf_note_active, cm_note_active)

    tf_active_activations = int(np.sum(tf_note_active))
    activation_flip_count = int(np.sum(activation_flips))
    activation_flip_pct = (
        (activation_flip_count / tf_active_activations) * 100.0 if tf_active_activations > 0 else 0.0
    )

    tf_active_frames = np.any(tf_note_active, axis=1)
    frame_has_flip = np.any(activation_flips, axis=1)
    flipped_active_frames = int(np.sum(np.logical_and(frame_has_flip, tf_active_frames)))
    total_active_frames = int(np.sum(tf_active_frames))
    frame_flip_pct = (flipped_active_frames / total_active_frames) * 100.0 if total_active_frames > 0 else 0.0

    note_flip = {
        "threshold": float(threshold),
        "active_frames_tf": total_active_frames,
        "flipped_active_frames": flipped_active_frames,
        "frame_flip_rate_pct_of_active_frames": float(frame_flip_pct),
        "active_activations_tf": tf_active_activations,
        "flipped_activations": activation_flip_count,
        "activation_flip_rate_pct_of_active_activations": float(activation_flip_pct),
    }

    return per_output, note_flip


def classify_gate(frame_flip_rate_pct: float) -> Tuple[str, str]:
    if frame_flip_rate_pct < 2.0:
        return "GREEN", "Frame-flip rate is <2%; conversion differences are unlikely to affect note extraction."
    if frame_flip_rate_pct <= 5.0:
        return "YELLOW", "Frame-flip rate is 2-5%; threshold tuning may be needed."
    return "RED", "Frame-flip rate is >5%; conversion parity issue needs debugging."


def worst_divergence_details(tf_outputs: Dict[str, Any], coreml_outputs: Dict[str, Any], top_k: int = 12) -> Dict[str, Any]:
    import numpy as np

    best_name = None
    best_max = -1.0
    best_diff = None
    for key in ("note", "onset", "contour"):
        tf_arr = np.asarray(tf_outputs[key], dtype=np.float32)
        cm_arr = np.asarray(coreml_outputs[key], dtype=np.float32)
        common_time = min(tf_arr.shape[0], cm_arr.shape[0])
        common_freq = min(tf_arr.shape[1], cm_arr.shape[1])
        diff = np.abs(tf_arr[:common_time, :common_freq] - cm_arr[:common_time, :common_freq])
        this_max = float(np.max(diff))
        if this_max > best_max:
            best_max = this_max
            best_name = key
            best_diff = diff

    if best_name is None or best_diff is None:
        return {"error": "Unable to determine divergence details."}

    frame_mean = best_diff.mean(axis=1)
    top_idx = frame_mean.argsort()[::-1][:top_k]
    top_frames = [{"frame_index": int(i), "mean_abs_diff": float(frame_mean[i])} for i in top_idx]
    return {
        "worst_output": best_name,
        "worst_output_max_abs_diff": best_max,
        "top_divergent_frames_by_mean_abs_diff": top_frames,
    }


def save_outputs(output_dir: Path, prefix: str, outputs: Dict[str, Any]) -> Dict[str, str]:
    import numpy as np

    saved: Dict[str, str] = {}
    for key in ("note", "onset", "contour"):
        path = output_dir / f"{prefix}_{key}.npy"
        np.save(path, np.asarray(outputs[key], dtype=np.float32))
        saved[key] = str(path)
    return saved


def write_report(
    path: Path,
    audio_path: Path,
    tf_paths: Dict[str, str],
    coreml_gpu_paths: Dict[str, str],
    per_output_gpu: Dict[str, Any],
    note_flip_gpu: Dict[str, Any],
    gate_gpu: Tuple[str, str],
    metadata: Dict[str, Any],
    gpu_divergence: Dict[str, Any] | None = None,
    cpu_only: Dict[str, Any] | None = None,
) -> None:
    lines: List[str] = [
        "# Step 3 Full Pipeline Comparison",
        "",
        "## Inputs",
        f"- Audio: `{audio_path}`",
        f"- Overlap frames: `{metadata['overlap_frames']}`",
        f"- Total windows: `{metadata['num_windows']}`",
        "",
        "## Saved Aggregated Tensors",
        "- Tensor layout: `(time_frames, frequency_bins)`",
    ]
    for key in ("note", "onset", "contour"):
        lines.append(f"- TensorFlow `{key}`: `{tf_paths[key]}`")
    for key in ("note", "onset", "contour"):
        lines.append(f"- CoreML CPU+GPU `{key}`: `{coreml_gpu_paths[key]}`")

    lines += ["", "## CPU+GPU Aggregated Diff Metrics"]
    for key in ("note", "onset", "contour"):
        m = per_output_gpu[key]
        lines.append(
            f"- `{key}`: max_abs_diff={m['max_abs_diff']:.9f}, "
            f"mean_abs_diff={m['mean_abs_diff']:.9f}, aligned_shape={m['aligned_shape']}"
        )

    lines += [
        "",
        "## Note Threshold Flip Metrics (0.5)",
        f"- Active frames (TF): `{note_flip_gpu['active_frames_tf']}`",
        f"- Flipped active frames: `{note_flip_gpu['flipped_active_frames']}`",
        f"- Frame-flip rate (% of active frames): `{note_flip_gpu['frame_flip_rate_pct_of_active_frames']:.6f}`",
        f"- Active activations (TF): `{note_flip_gpu['active_activations_tf']}`",
        f"- Flipped activations: `{note_flip_gpu['flipped_activations']}`",
        (
            "- Activation-flip rate (% of active activations): "
            f"`{note_flip_gpu['activation_flip_rate_pct_of_active_activations']:.6f}`"
        ),
        "",
        "## Gate",
        f"- Status: `{gate_gpu[0]}`",
        f"- Reason: {gate_gpu[1]}",
    ]

    if cpu_only is not None:
        lines += [
            "",
            "## RED Follow-Up (CPU_ONLY)",
        ]
        if cpu_only.get("blocked", False):
            lines += [
                "- CPU_ONLY run status: `BLOCKED`",
                f"- CPU_ONLY error: `{cpu_only['error']}`",
                f"- CPU_ONLY details: {cpu_only.get('details', 'n/a')}",
            ]
        else:
            lines += [
                f"- CPU_ONLY gate: `{cpu_only['gate'][0]}`",
                f"- CPU_ONLY reason: {cpu_only['gate'][1]}",
                "- CPU_ONLY metrics:",
            ]
            for key in ("note", "onset", "contour"):
                m = cpu_only["per_output"][key]
                lines.append(
                    f"  - `{key}`: max_abs_diff={m['max_abs_diff']:.9f}, "
                    f"mean_abs_diff={m['mean_abs_diff']:.9f}"
                )
            nf = cpu_only["note_flip"]
            lines += [
                (
                    "- CPU_ONLY frame-flip rate (% of active frames): "
                    f"`{nf['frame_flip_rate_pct_of_active_frames']:.6f}`"
                ),
                (
                    "- CPU_ONLY activation-flip rate (% of active activations): "
                    f"`{nf['activation_flip_rate_pct_of_active_activations']:.6f}`"
                ),
                f"- CPU_ONLY interpretation: {cpu_only['interpretation']}",
            ]
            if "divergence" in cpu_only:
                div = cpu_only["divergence"]
                lines += [
                    f"- Worst diverging tensor on CPU_ONLY: `{div.get('worst_output')}` "
                    f"(max_abs_diff={div.get('worst_output_max_abs_diff', 0.0):.9f})",
                    "- Most divergent frame indices (by mean abs diff):",
                ]
                for row in div.get("top_divergent_frames_by_mean_abs_diff", []):
                    lines.append(f"  - frame `{row['frame_index']}`: `{row['mean_abs_diff']:.9f}`")

    if gpu_divergence is not None:
        lines += [
            "",
            "## Divergence Localization (CPU+GPU)",
            f"- Worst diverging tensor: `{gpu_divergence.get('worst_output')}` "
            f"(max_abs_diff={gpu_divergence.get('worst_output_max_abs_diff', 0.0):.9f})",
            "- Most divergent frame indices (by mean abs diff):",
        ]
        for row in gpu_divergence.get("top_divergent_frames_by_mean_abs_diff", []):
            lines.append(f"- frame `{row['frame_index']}`: mean_abs_diff=`{row['mean_abs_diff']:.9f}`")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default="cqt_spike/synthetic_g_major.wav", help="Input audio path.")
    parser.add_argument("--basic-pitch-repo", default=".tmp/basic-pitch", help="Local clone path for basic-pitch.")
    parser.add_argument("--output-dir", default="cqt_spike", help="Output directory.")
    parser.add_argument(
        "--coreml-model",
        default="cqt_spike/BasicPitch_from_tf.mlpackage",
        help="CoreML model path to use for comparison.",
    )
    parser.add_argument("--overlap-frames", type=int, default=30, help="Number of overlapping frames.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for note activation flips.")
    args = parser.parse_args()

    require_python_compat()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(args.basic_pitch_repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(
            f"basic-pitch repo not found at {repo}. Clone it first: "
            "git clone https://github.com/spotify/basic-pitch .tmp/basic-pitch"
        )
    sys.path.insert(0, str(repo))

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    coreml_model_path = Path(args.coreml_model).resolve()
    if not coreml_model_path.exists():
        raise FileNotFoundError(f"CoreML model not found: {coreml_model_path}")

    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception as exc:
        raise RuntimeError(
            "Failed to import basic_pitch runtime. Use Python 3.10/3.11 and install requirements-spike.txt."
        ) from exc

    tf_outputs, tf_meta = run_full_pipeline_tf(audio_path, Path(ICASSP_2022_MODEL_PATH), args.overlap_frames)
    tf_paths = save_outputs(output_dir, "step3_tf_full", tf_outputs)

    coreml_gpu_outputs, cm_gpu_meta = run_full_pipeline_coreml(
        audio_path, coreml_model_path, args.overlap_frames, "CPU_AND_GPU"
    )
    coreml_gpu_paths = save_outputs(output_dir, "step3_coreml_cpu_and_gpu_full", coreml_gpu_outputs)

    per_output_gpu, note_flip_gpu = compare_outputs(tf_outputs, coreml_gpu_outputs, args.threshold)
    gate_gpu = classify_gate(note_flip_gpu["frame_flip_rate_pct_of_active_frames"])
    gpu_divergence = worst_divergence_details(tf_outputs, coreml_gpu_outputs) if gate_gpu[0] == "RED" else None

    summary: Dict[str, Any] = {
        "audio_path": str(audio_path),
        "coreml_model_path": str(coreml_model_path),
        "overlap_frames": int(args.overlap_frames),
        "tf_meta": tf_meta,
        "coreml_cpu_and_gpu_meta": cm_gpu_meta,
        "cpu_and_gpu": {
            "per_output": per_output_gpu,
            "note_flip": note_flip_gpu,
            "gate": {"status": gate_gpu[0], "reason": gate_gpu[1]},
            "saved_outputs": coreml_gpu_paths,
        },
        "tf_saved_outputs": tf_paths,
    }

    cpu_only_section: Dict[str, Any] | None = None
    if gate_gpu[0] == "RED":
        probe = probe_cpu_only_predict(coreml_model_path)
        if not probe["ok"]:
            details = ""
            if probe.get("stderr_tail"):
                details = probe["stderr_tail"]
            elif probe.get("stdout_tail"):
                details = probe["stdout_tail"]
            signal_hint = probe.get("signal_hint", "")
            if signal_hint:
                details = f"{signal_hint}. {details}".strip()
            cpu_only_section = {
                "blocked": True,
                "error": "CPU_ONLY prediction probe crashed.",
                "details": details or f"subprocess return code={probe.get('returncode')}",
            }
            summary["cpu_only"] = {
                "status": "blocked",
                "error": cpu_only_section["error"],
                "details": cpu_only_section["details"],
            }
        else:
            coreml_cpu_outputs, cm_cpu_meta = run_full_pipeline_coreml(
                audio_path, coreml_model_path, args.overlap_frames, "CPU_ONLY"
            )
            cpu_only_paths = save_outputs(output_dir, "step3_coreml_cpu_only_full", coreml_cpu_outputs)
            per_output_cpu, note_flip_cpu = compare_outputs(tf_outputs, coreml_cpu_outputs, args.threshold)
            gate_cpu = classify_gate(note_flip_cpu["frame_flip_rate_pct_of_active_frames"])

            gpu_rate = note_flip_gpu["frame_flip_rate_pct_of_active_frames"]
            cpu_rate = note_flip_cpu["frame_flip_rate_pct_of_active_frames"]
            drop = gpu_rate - cpu_rate
            significant_drop = drop >= max(1.0, 0.3 * gpu_rate)
            if significant_drop:
                interpretation = (
                    "CPU_ONLY reduced frame-flip rate significantly; likely precision/compute-unit behavior. "
                    "Use CPU/GPU path for stability."
                )
            else:
                interpretation = (
                    "CPU_ONLY did not materially reduce divergence; conversion parity issue is likely in coremltools conversion."
                )

            cpu_only_section = {
                "meta": cm_cpu_meta,
                "saved_outputs": cpu_only_paths,
                "per_output": per_output_cpu,
                "note_flip": note_flip_cpu,
                "gate": gate_cpu,
                "interpretation": interpretation,
            }
            if not significant_drop:
                cpu_only_section["divergence"] = worst_divergence_details(tf_outputs, coreml_cpu_outputs)

            summary["cpu_only"] = {
                "meta": cm_cpu_meta,
                "per_output": per_output_cpu,
                "note_flip": note_flip_cpu,
                "gate": {"status": gate_cpu[0], "reason": gate_cpu[1]},
                "saved_outputs": cpu_only_paths,
                "interpretation": interpretation,
            }
            if "divergence" in cpu_only_section:
                summary["cpu_only"]["divergence"] = cpu_only_section["divergence"]

    if gpu_divergence is not None:
        summary["cpu_and_gpu"]["divergence"] = gpu_divergence

    metadata = {
        "overlap_frames": int(args.overlap_frames),
        "num_windows": int(tf_meta["num_windows"]),
    }
    report_path = output_dir / "step3_full_pipeline_comparison.md"
    write_report(
        report_path,
        audio_path,
        tf_paths,
        coreml_gpu_paths,
        per_output_gpu,
        note_flip_gpu,
        gate_gpu,
        metadata,
        gpu_divergence=gpu_divergence,
        cpu_only=cpu_only_section,
    )

    summary_path = output_dir / "step3_full_pipeline_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote report: {report_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
