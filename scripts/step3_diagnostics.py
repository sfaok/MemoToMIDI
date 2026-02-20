#!/usr/bin/env python
"""Step 3 diagnostics: isolate input parity, no-stitch parity, and activation distribution."""

from __future__ import annotations

import argparse
import json
import sys
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


def run_tf_windows(audio_path: Path, model_path: Path, overlap_frames: int) -> Dict[str, Any]:
    import numpy as np
    from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
    from basic_pitch.inference import Model, get_audio_input

    overlap_len = overlap_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    model = Model(model_path)
    inputs: List[Any] = []
    outputs: Dict[str, List[Any]] = {"note": [], "onset": [], "contour": []}
    audio_original_length = 0
    for audio_windowed, _, this_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        x = audio_windowed.astype(np.float32)
        pred = model.predict(x)
        inputs.append(np.asarray(x, dtype=np.float32))
        for key in outputs.keys():
            outputs[key].append(np.asarray(pred[key], dtype=np.float32))
        audio_original_length = int(this_original_length)

    if not inputs:
        raise RuntimeError("No audio windows were generated for TensorFlow inference.")

    return {
        "inputs": inputs,
        "outputs": outputs,
        "meta": {
            "overlap_len_samples": int(overlap_len),
            "hop_size_samples": int(hop_size),
            "audio_original_length_samples": int(audio_original_length),
            "num_windows": int(len(inputs)),
        },
    }


def run_coreml_windows(audio_path: Path, coreml_model_path: Path, overlap_frames: int) -> Dict[str, Any]:
    import numpy as np
    import coremltools as ct
    from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP
    from basic_pitch.inference import get_audio_input

    overlap_len = overlap_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    mlmodel = ct.models.MLModel(str(coreml_model_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()
    if not spec.description.input:
        raise RuntimeError("CoreML model has no inputs.")
    input_name = spec.description.input[0].name

    inputs: List[Any] = []
    outputs: Dict[str, List[Any]] = {"note": [], "onset": [], "contour": []}
    audio_original_length = 0
    for audio_windowed, _, this_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        x = audio_windowed.astype(np.float32)
        raw_pred = mlmodel.predict({input_name: x})
        pred = map_coreml_outputs(raw_pred)
        inputs.append(np.asarray(x, dtype=np.float32))
        for key in outputs.keys():
            outputs[key].append(np.asarray(pred[key], dtype=np.float32))
        audio_original_length = int(this_original_length)

    if not inputs:
        raise RuntimeError("No audio windows were generated for CoreML inference.")

    return {
        "inputs": inputs,
        "outputs": outputs,
        "meta": {
            "overlap_len_samples": int(overlap_len),
            "hop_size_samples": int(hop_size),
            "audio_original_length_samples": int(audio_original_length),
            "num_windows": int(len(inputs)),
        },
    }


def compare_inputs(tf_inputs: List[Any], coreml_inputs: List[Any]) -> Dict[str, Any]:
    import numpy as np

    total = min(len(tf_inputs), len(coreml_inputs))
    per_window: List[Dict[str, Any]] = []
    global_max = 0.0
    global_mean_sum = 0.0
    for i in range(total):
        tf_arr = np.asarray(tf_inputs[i], dtype=np.float32)
        cm_arr = np.asarray(coreml_inputs[i], dtype=np.float32)
        if tf_arr.shape != cm_arr.shape:
            per_window.append(
                {
                    "window_index": i,
                    "tf_shape": list(tf_arr.shape),
                    "coreml_shape": list(cm_arr.shape),
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "shape_mismatch": True,
                }
            )
            continue
        diff = np.abs(tf_arr - cm_arr)
        w_max = float(np.max(diff))
        w_mean = float(np.mean(diff))
        global_max = max(global_max, w_max)
        global_mean_sum += w_mean
        per_window.append(
            {
                "window_index": i,
                "shape": list(tf_arr.shape),
                "max_abs_diff": w_max,
                "mean_abs_diff": w_mean,
                "shape_mismatch": False,
            }
        )

    shape_mismatch = any(row["shape_mismatch"] for row in per_window)
    compared_windows = sum(1 for row in per_window if not row["shape_mismatch"])
    global_mean = global_mean_sum / compared_windows if compared_windows else 0.0
    return {
        "num_tf_windows": int(len(tf_inputs)),
        "num_coreml_windows": int(len(coreml_inputs)),
        "num_compared_windows": int(total),
        "shape_mismatch_present": shape_mismatch,
        "per_window": per_window,
        "global_max_abs_diff": float(global_max),
        "global_mean_abs_diff": float(global_mean),
        "threshold_for_identity": 1e-6,
        "inputs_identical_under_threshold": (not shape_mismatch) and (global_max <= 1e-6),
    }


def save_window_inputs(output_dir: Path, prefix: str, inputs: List[Any]) -> Dict[str, Any]:
    import numpy as np

    saved: List[str] = []
    for i, arr in enumerate(inputs):
        path = output_dir / f"{prefix}_window_{i:03d}.npy"
        np.save(path, np.asarray(arr, dtype=np.float32))
        saved.append(str(path))

    stacked = np.concatenate([np.asarray(arr, dtype=np.float32) for arr in inputs], axis=0)
    stacked_path = output_dir / f"{prefix}_all_windows.npy"
    np.save(stacked_path, stacked)
    return {"per_window_paths": saved, "stacked_path": str(stacked_path), "stacked_shape": list(stacked.shape)}


def frame_flip_metrics(tf_note: Any, coreml_note: Any, threshold: float) -> Dict[str, Any]:
    import numpy as np

    tf_arr = np.asarray(tf_note, dtype=np.float32)
    cm_arr = np.asarray(coreml_note, dtype=np.float32)
    common_time = min(tf_arr.shape[0], cm_arr.shape[0])
    common_freq = min(tf_arr.shape[1], cm_arr.shape[1])
    tf_trim = tf_arr[:common_time, :common_freq]
    cm_trim = cm_arr[:common_time, :common_freq]

    tf_active = tf_trim >= threshold
    cm_active = cm_trim >= threshold
    flips = np.logical_xor(tf_active, cm_active)
    frame_has_flip = np.any(flips, axis=1)
    tf_active_frames = np.any(tf_active, axis=1)
    flipped_active_frames = int(np.sum(np.logical_and(frame_has_flip, tf_active_frames)))
    total_active_frames = int(np.sum(tf_active_frames))
    frame_flip_rate = (flipped_active_frames / total_active_frames) * 100.0 if total_active_frames > 0 else 0.0

    return {
        "threshold": float(threshold),
        "active_frames_tf": total_active_frames,
        "flipped_active_frames": flipped_active_frames,
        "frame_flip_rate_pct_of_active_frames": float(frame_flip_rate),
    }


def check_single_window_no_stitch(tf_outputs: Dict[str, List[Any]], coreml_outputs: Dict[str, List[Any]]) -> Dict[str, Any]:
    import numpy as np

    tf_note = np.asarray(tf_outputs["note"][0], dtype=np.float32)[0]
    cm_note = np.asarray(coreml_outputs["note"][0], dtype=np.float32)[0]
    common_time = min(tf_note.shape[0], cm_note.shape[0])
    common_freq = min(tf_note.shape[1], cm_note.shape[1])
    tf_trim = tf_note[:common_time, :common_freq]
    cm_trim = cm_note[:common_time, :common_freq]
    diff = np.abs(tf_trim - cm_trim)

    thresholds = [0.3, 0.5, 0.7]
    flips = [frame_flip_metrics(tf_trim, cm_trim, t) for t in thresholds]
    return {
        "tf_shape": list(tf_note.shape),
        "coreml_shape": list(cm_note.shape),
        "aligned_shape": [int(common_time), int(common_freq)],
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "frame_flip_rates": flips,
    }


def aggregate_tf_note(tf_outputs: Dict[str, List[Any]], audio_original_length: int, overlap_frames: int, hop_size: int) -> Any:
    import numpy as np
    from basic_pitch.inference import unwrap_output

    concatenated = np.concatenate([np.asarray(arr, dtype=np.float32) for arr in tf_outputs["note"]], axis=0)
    return unwrap_output(concatenated, audio_original_length, overlap_frames, hop_size).astype(np.float32)


def histogram_0_to_1_10bins(arr: Any) -> Dict[str, Any]:
    import numpy as np

    values = np.asarray(arr, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    hist, edges = np.histogram(values, bins=bins)

    in_04_06 = int(np.sum((values >= 0.4) & (values < 0.6)))
    total = int(values.size)
    active = int(np.sum(values >= 0.5))
    borderline_active = int(np.sum((values >= 0.5) & (values < 0.6)))

    rows: List[Dict[str, Any]] = []
    for i in range(len(hist)):
        rows.append(
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(hist[i]),
            }
        )

    return {
        "bin_count": 10,
        "range_min": 0.0,
        "range_max": 1.0,
        "histogram": rows,
        "total_activations": total,
        "count_in_0_4_to_0_6": in_04_06,
        "pct_in_0_4_to_0_6_of_all": float((in_04_06 / total) * 100.0 if total else 0.0),
        "active_activations_at_0_5_or_more": active,
        "active_activations_in_0_5_to_0_6": borderline_active,
        "pct_of_active_in_0_5_to_0_6": float((borderline_active / active) * 100.0 if active else 0.0),
    }


def write_report(
    report_path: Path,
    audio_path: Path,
    coreml_model_path: Path,
    overlap_frames: int,
    tf_input_paths: Dict[str, Any],
    coreml_input_paths: Dict[str, Any],
    check1: Dict[str, Any],
    check2: Dict[str, Any],
    check3: Dict[str, Any],
) -> None:
    lines: List[str] = [
        "# Step 3 Diagnostics",
        "",
        "## Inputs",
        f"- Audio: `{audio_path}`",
        f"- CoreML model: `{coreml_model_path}`",
        f"- Overlap frames: `{overlap_frames}`",
        "",
        "## Check 1 - Identical Input Tensor Per Window",
        "- Saved exact per-window model inputs:",
        f"- TensorFlow windows: `{tf_input_paths['per_window_paths']}`",
        f"- TensorFlow stacked: `{tf_input_paths['stacked_path']}` shape={tf_input_paths['stacked_shape']}",
        f"- CoreML windows: `{coreml_input_paths['per_window_paths']}`",
        f"- CoreML stacked: `{coreml_input_paths['stacked_path']}` shape={coreml_input_paths['stacked_shape']}",
        "",
        "- Per-window input diffs (TF vs CoreML):",
    ]

    for row in check1["per_window"]:
        if row["shape_mismatch"]:
            lines.append(
                f"- window `{row['window_index']}`: SHAPE MISMATCH "
                f"tf_shape={row['tf_shape']} coreml_shape={row['coreml_shape']}"
            )
        else:
            lines.append(
                f"- window `{row['window_index']}`: "
                f"max_abs_diff=`{row['max_abs_diff']:.12f}`, mean_abs_diff=`{row['mean_abs_diff']:.12f}`"
            )

    verdict = "IDENTICAL (<1e-6)" if check1["inputs_identical_under_threshold"] else "DIFFERENT (>1e-6 or shape mismatch)"
    lines += [
        f"- Global max abs diff: `{check1['global_max_abs_diff']:.12f}`",
        f"- Global mean abs diff: `{check1['global_mean_abs_diff']:.12f}`",
        f"- Check 1 verdict: `{verdict}`",
        "",
        "## Check 2 - Single Window, No Stitching",
        "- Compared first raw output window only (`note` head), no overlap averaging/unwrap.",
        f"- Max abs diff: `{check2['max_abs_diff']:.9f}`",
        f"- Mean abs diff: `{check2['mean_abs_diff']:.9f}`",
        "- Frame-flip rates (% of active TF frames):",
    ]

    for row in check2["frame_flip_rates"]:
        lines.append(
            f"- threshold `{row['threshold']:.1f}`: "
            f"active_frames_tf=`{row['active_frames_tf']}`, "
            f"flipped_active_frames=`{row['flipped_active_frames']}`, "
            f"frame_flip_rate_pct=`{row['frame_flip_rate_pct_of_active_frames']:.6f}`"
        )

    lines += [
        "",
        "## Check 3 - TF Note Activation Histogram",
        "- Source tensor: full aggregated TF note output (after normal Step 3 unwrapping).",
        "- Histogram bins: `[0.0, 0.1), ..., [0.9, 1.0]`",
        "| Bin | Count |",
        "| --- | ---: |",
    ]

    for row in check3["histogram"]:
        lines.append(f"| {row['bin_start']:.1f}-{row['bin_end']:.1f} | {row['count']} |")

    lines += [
        "",
        f"- Total activations: `{check3['total_activations']}`",
        f"- Activations in 0.4-0.6: `{check3['count_in_0_4_to_0_6']}` "
        f"(`{check3['pct_in_0_4_to_0_6_of_all']:.6f}%` of all activations)",
        f"- Active activations (>=0.5): `{check3['active_activations_at_0_5_or_more']}`",
        f"- Active activations in 0.5-0.6: `{check3['active_activations_in_0_5_to_0_6']}` "
        f"(`{check3['pct_of_active_in_0_5_to_0_6']:.6f}%` of active activations)",
    ]

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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

    tf_run = run_tf_windows(audio_path, Path(ICASSP_2022_MODEL_PATH), args.overlap_frames)
    coreml_run = run_coreml_windows(audio_path, coreml_model_path, args.overlap_frames)

    tf_input_paths = save_window_inputs(output_dir, "step3_diag_tf_input", tf_run["inputs"])
    coreml_input_paths = save_window_inputs(output_dir, "step3_diag_coreml_input", coreml_run["inputs"])

    check1 = compare_inputs(tf_run["inputs"], coreml_run["inputs"])
    check2 = check_single_window_no_stitch(tf_run["outputs"], coreml_run["outputs"])
    check3 = histogram_0_to_1_10bins(
        aggregate_tf_note(
            tf_run["outputs"],
            tf_run["meta"]["audio_original_length_samples"],
            args.overlap_frames,
            tf_run["meta"]["hop_size_samples"],
        )
    )

    report_path = output_dir / "step3_diagnostics.md"
    write_report(
        report_path,
        audio_path,
        coreml_model_path,
        args.overlap_frames,
        tf_input_paths,
        coreml_input_paths,
        check1,
        check2,
        check3,
    )

    summary = {
        "audio_path": str(audio_path),
        "coreml_model_path": str(coreml_model_path),
        "overlap_frames": int(args.overlap_frames),
        "tf_meta": tf_run["meta"],
        "coreml_meta": coreml_run["meta"],
        "check1_identical_input": check1,
        "check2_single_window_no_stitch": check2,
        "check3_tf_note_histogram": check3,
        "saved_tf_inputs": tf_input_paths,
        "saved_coreml_inputs": coreml_input_paths,
    }
    summary_path = output_dir / "step3_diagnostics.json"
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
