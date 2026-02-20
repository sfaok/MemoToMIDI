#!/usr/bin/env python
"""Step 3 real-audio comparison: TF vs CoreML Basic Pitch diagnostics and note extraction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


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


def prepare_audio_wav(source_path: Path, wav_path: Path, target_sr: int) -> Dict[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf

    y, sr = librosa.load(str(source_path), sr=target_sr, mono=True)
    if y.size == 0:
        raise RuntimeError(f"Decoded audio is empty: {source_path}")

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), y, target_sr, subtype="PCM_16")

    return {
        "source_path": str(source_path),
        "wav_path": str(wav_path),
        "decoded_sr": int(sr),
        "saved_sr": int(target_sr),
        "num_samples": int(y.shape[0]),
        "duration_seconds": float(y.shape[0] / float(target_sr)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "rms": float(np.sqrt(np.mean(np.square(y)))),
    }


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


def compare_inputs(tf_inputs: Sequence[Any], coreml_inputs: Sequence[Any]) -> Dict[str, Any]:
    import numpy as np

    total = min(len(tf_inputs), len(coreml_inputs))
    per_window: List[Dict[str, Any]] = []
    global_max = 0.0
    global_mean_sum = 0.0
    compared_windows = 0

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
        compared_windows += 1
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
    return {
        "num_tf_windows": int(len(tf_inputs)),
        "num_coreml_windows": int(len(coreml_inputs)),
        "num_compared_windows": int(total),
        "shape_mismatch_present": bool(shape_mismatch),
        "per_window": per_window,
        "global_max_abs_diff": float(global_max),
        "global_mean_abs_diff": float(global_mean_sum / compared_windows if compared_windows else 0.0),
        "threshold_for_identity": 1e-6,
        "inputs_identical_under_threshold": (not shape_mismatch) and (global_max <= 1e-6),
    }


def aggregate_outputs(
    outputs: Dict[str, Sequence[Any]], audio_original_length: int, overlap_frames: int, hop_size: int
) -> Dict[str, Any]:
    import numpy as np
    from basic_pitch.inference import unwrap_output

    return {
        key: unwrap_output(
            np.concatenate([np.asarray(arr, dtype=np.float32) for arr in values], axis=0),
            audio_original_length,
            overlap_frames,
            hop_size,
        ).astype(np.float32)
        for key, values in outputs.items()
    }


def note_tensor_diff_metrics(tf_note: Any, coreml_note: Any, thresholds: Sequence[float]) -> Dict[str, Any]:
    import numpy as np

    tf_arr = np.asarray(tf_note, dtype=np.float32)
    cm_arr = np.asarray(coreml_note, dtype=np.float32)

    common_time = min(tf_arr.shape[0], cm_arr.shape[0])
    common_freq = min(tf_arr.shape[1], cm_arr.shape[1])
    tf_trim = tf_arr[:common_time, :common_freq]
    cm_trim = cm_arr[:common_time, :common_freq]

    diff = np.abs(tf_trim - cm_trim)

    frame_flip_rates: List[Dict[str, Any]] = []
    for threshold in thresholds:
        tf_active = tf_trim >= threshold
        cm_active = cm_trim >= threshold
        flips = np.logical_xor(tf_active, cm_active)
        frame_has_flip = np.any(flips, axis=1)
        tf_active_frames = np.any(tf_active, axis=1)

        flipped_active_frames = int(np.sum(np.logical_and(frame_has_flip, tf_active_frames)))
        total_active_frames = int(np.sum(tf_active_frames))
        frame_flip_rate = (flipped_active_frames / total_active_frames) * 100.0 if total_active_frames > 0 else 0.0

        frame_flip_rates.append(
            {
                "threshold": float(threshold),
                "active_frames_tf": total_active_frames,
                "flipped_active_frames": flipped_active_frames,
                "frame_flip_rate_pct_of_active_frames": float(frame_flip_rate),
            }
        )

    return {
        "tf_shape": list(tf_arr.shape),
        "coreml_shape": list(cm_arr.shape),
        "aligned_shape": [int(common_time), int(common_freq)],
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "frame_flip_rates": frame_flip_rates,
    }


def histogram_0_to_1_10bins(arr: Any) -> Dict[str, Any]:
    import numpy as np

    values = np.asarray(arr, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    hist, edges = np.histogram(values, bins=bins)

    rows: List[Dict[str, Any]] = []
    for i in range(len(hist)):
        rows.append({"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(hist[i])})

    return {
        "histogram": rows,
        "total_activations": int(values.size),
        "peak_activation": float(np.max(values)),
        "count_above_0_7": int(np.sum(values > 0.7)),
        "count_above_0_8": int(np.sum(values > 0.8)),
    }


def extract_notes(
    output: Dict[str, Any], onset_threshold: float, frame_threshold: float, min_note_len_frames: int
) -> List[Dict[str, Any]]:
    import pretty_midi
    from basic_pitch.note_creation import model_output_to_notes

    _midi, note_events = model_output_to_notes(
        output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len_frames,
    )

    formatted: List[Dict[str, Any]] = []
    for start, end, pitch, amplitude, _bends in sorted(note_events, key=lambda x: (x[0], x[2], x[1])):
        duration = max(0.0, float(end) - float(start))
        p = int(pitch)
        formatted.append(
            {
                "pitch_midi": p,
                "pitch_name": pretty_midi.note_number_to_name(p),
                "start_time_s": float(start),
                "duration_s": duration,
                "amplitude": float(amplitude),
            }
        )

    return formatted


def compare_note_lists(
    tf_notes: Sequence[Dict[str, Any]],
    coreml_notes: Sequence[Dict[str, Any]],
    onset_tolerance_s: float = 0.05,
    duration_tolerance_s: float = 0.15,
) -> Dict[str, Any]:
    used_coreml: set[int] = set()
    matches: List[Dict[str, Any]] = []
    tf_only: List[Dict[str, Any]] = []

    for i, tf_note in enumerate(tf_notes):
        best_j = -1
        best_cost = math.inf
        best_onset_diff = 0.0
        best_dur_diff = 0.0
        for j, cm_note in enumerate(coreml_notes):
            if j in used_coreml:
                continue
            if int(tf_note["pitch_midi"]) != int(cm_note["pitch_midi"]):
                continue

            onset_diff = abs(float(tf_note["start_time_s"]) - float(cm_note["start_time_s"]))
            dur_diff = abs(float(tf_note["duration_s"]) - float(cm_note["duration_s"]))
            if onset_diff <= onset_tolerance_s and dur_diff <= duration_tolerance_s:
                cost = onset_diff + dur_diff
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
                    best_onset_diff = onset_diff
                    best_dur_diff = dur_diff

        if best_j >= 0:
            used_coreml.add(best_j)
            matches.append(
                {
                    "tf_index": i,
                    "coreml_index": best_j,
                    "pitch_midi": int(tf_note["pitch_midi"]),
                    "pitch_name": str(tf_note["pitch_name"]),
                    "onset_diff_s": float(best_onset_diff),
                    "duration_diff_s": float(best_dur_diff),
                }
            )
        else:
            tf_only.append(dict(tf_note))

    coreml_only = [dict(coreml_notes[j]) for j in range(len(coreml_notes)) if j not in used_coreml]

    matched = len(matches)
    tf_count = len(tf_notes)
    coreml_count = len(coreml_notes)
    precision = matched / coreml_count if coreml_count else 0.0
    recall = matched / tf_count if tf_count else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    mismatch_count = len(tf_only) + len(coreml_only)

    return {
        "onset_tolerance_s": float(onset_tolerance_s),
        "duration_tolerance_s": float(duration_tolerance_s),
        "tf_note_count": tf_count,
        "coreml_note_count": coreml_count,
        "matched_count": matched,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mismatch_count": int(mismatch_count),
        "matches": matches,
        "tf_only": tf_only,
        "coreml_only": coreml_only,
    }


def sweep_coreml_onset_thresholds(
    coreml_output: Dict[str, Any],
    tf_reference_notes: Sequence[Dict[str, Any]],
    frame_threshold: float,
    min_note_len_frames: int,
) -> Dict[str, Any]:
    thresholds = [round(0.2 + i * 0.05, 2) for i in range(9)]

    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_sort_key: Tuple[int, float, float] | None = None

    for threshold in thresholds:
        notes = extract_notes(coreml_output, threshold, frame_threshold, min_note_len_frames)
        cmp = compare_note_lists(tf_reference_notes, notes)
        row = {
            "threshold": float(threshold),
            "note_count": int(len(notes)),
            "matched_count": int(cmp["matched_count"]),
            "mismatch_count": int(cmp["mismatch_count"]),
            "precision": float(cmp["precision"]),
            "recall": float(cmp["recall"]),
            "f1": float(cmp["f1"]),
            "tf_only_count": int(len(cmp["tf_only"])),
            "coreml_only_count": int(len(cmp["coreml_only"])),
            "comparison": cmp,
            "notes": notes,
        }
        rows.append(row)

        sort_key = (
            int(cmp["mismatch_count"]),
            -float(cmp["f1"]),
            abs(float(threshold) - 0.5),
        )
        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_row = row

    if best_row is None:
        raise RuntimeError("Threshold sweep produced no rows.")

    return {
        "rows": rows,
        "best": best_row,
        "selection_rule": "min mismatch_count, then max F1, then closest threshold to 0.5",
    }


def note_rows_to_markdown_rows(notes: Sequence[Dict[str, Any]]) -> List[str]:
    rows: List[str] = []
    for note in notes:
        rows.append(
            "| {pitch_name} ({pitch_midi}) | {start:.3f} | {dur:.3f} | {amp:.3f} |".format(
                pitch_name=note["pitch_name"],
                pitch_midi=note["pitch_midi"],
                start=note["start_time_s"],
                dur=note["duration_s"],
                amp=note["amplitude"],
            )
        )
    return rows


def write_report(
    report_path: Path,
    audio_info: Dict[str, Any],
    coreml_model_path: Path,
    overlap_frames: int,
    input_parity: Dict[str, Any],
    histogram: Dict[str, Any],
    single_window_metrics: Dict[str, Any],
    full_pipeline_metrics: Dict[str, Any],
    default_thresholds: Dict[str, float],
    tf_notes_default: Sequence[Dict[str, Any]],
    coreml_notes_default: Sequence[Dict[str, Any]],
    default_note_comparison: Dict[str, Any],
    threshold_sweep: Dict[str, Any],
) -> None:
    lines: List[str] = [
        "# Step 3 Real Audio Comparison",
        "",
        "## Part A - Real Audio Used",
        f"- Source audio: `{audio_info['source_path']}`",
        f"- Converted WAV for test pipeline: `{audio_info['wav_path']}`",
        f"- Duration (seconds): `{audio_info['duration_seconds']:.3f}`",
        f"- Sample rate (Hz): `{audio_info['saved_sr']}`",
        f"- Amplitude range: min=`{audio_info['min']:.6f}`, max=`{audio_info['max']:.6f}`, rms=`{audio_info['rms']:.6f}`",
        "",
        "## Inputs",
        f"- CoreML model: `{coreml_model_path}`",
        f"- Overlap frames: `{overlap_frames}`",
        "",
        "## Input Parity (Same Windowed Pipeline)",
        f"- TF windows: `{input_parity['num_tf_windows']}`",
        f"- CoreML windows: `{input_parity['num_coreml_windows']}`",
        f"- Global max abs diff (input tensors): `{input_parity['global_max_abs_diff']:.12f}`",
        f"- Global mean abs diff (input tensors): `{input_parity['global_mean_abs_diff']:.12f}`",
        (
            "- Verdict: `IDENTICAL (<1e-6)`"
            if input_parity["inputs_identical_under_threshold"]
            else "- Verdict: `DIFFERENT (>1e-6 or shape mismatch)`"
        ),
        "",
        "## Part B.1 - TF Note Activation Histogram (Full Pipeline)",
        "- Histogram bins: `[0.0, 0.1), ..., [0.9, 1.0]`",
        "| Bin | Count |",
        "| --- | ---: |",
    ]

    for row in histogram["histogram"]:
        lines.append(f"| {row['bin_start']:.1f}-{row['bin_end']:.1f} | {row['count']} |")

    lines += [
        "",
        f"- Total activations: `{histogram['total_activations']}`",
        f"- Peak activation value: `{histogram['peak_activation']:.6f}`",
        f"- Activations above 0.7: `{histogram['count_above_0_7']}`",
        f"- Activations above 0.8: `{histogram['count_above_0_8']}`",
        "",
        "## Part B.2 - Single Window (No Stitching, `note` head)",
        f"- Max abs diff: `{single_window_metrics['max_abs_diff']:.9f}`",
        f"- Mean abs diff: `{single_window_metrics['mean_abs_diff']:.9f}`",
        "- Frame-flip rates (% of active TF frames):",
    ]

    for row in single_window_metrics["frame_flip_rates"]:
        lines.append(
            f"- threshold `{row['threshold']:.1f}`: "
            f"active_frames_tf=`{row['active_frames_tf']}`, "
            f"flipped_active_frames=`{row['flipped_active_frames']}`, "
            f"frame_flip_rate_pct=`{row['frame_flip_rate_pct_of_active_frames']:.6f}`"
        )

    lines += [
        "",
        "## Part B.3 - Full Pipeline (With Stitching, `note` head)",
        f"- Max abs diff: `{full_pipeline_metrics['max_abs_diff']:.9f}`",
        f"- Mean abs diff: `{full_pipeline_metrics['mean_abs_diff']:.9f}`",
        "- Frame-flip rates (% of active TF frames):",
    ]

    for row in full_pipeline_metrics["frame_flip_rates"]:
        lines.append(
            f"- threshold `{row['threshold']:.1f}`: "
            f"active_frames_tf=`{row['active_frames_tf']}`, "
            f"flipped_active_frames=`{row['flipped_active_frames']}`, "
            f"frame_flip_rate_pct=`{row['frame_flip_rate_pct_of_active_frames']:.6f}`"
        )

    lines += [
        "",
        "## Part B.4 - Musical Validation (Basic Pitch Post-Processing)",
        "- Default post-processing settings used:",
        f"  - onset_threshold=`{default_thresholds['onset_threshold']}`",
        f"  - frame_threshold=`{default_thresholds['frame_threshold']}`",
        f"  - minimum_note_length_ms=`{default_thresholds['minimum_note_length_ms']:.1f}`",
        f"- TF extracted notes: `{len(tf_notes_default)}`",
        f"- CoreML extracted notes: `{len(coreml_notes_default)}`",
        (
            "- Same detected-note set within tolerance: `YES`"
            if default_note_comparison["mismatch_count"] == 0
            else "- Same detected-note set within tolerance: `NO`"
        ),
        f"- Matching tolerance: onset `±{default_note_comparison['onset_tolerance_s']:.3f}s`, "
        f"duration `±{default_note_comparison['duration_tolerance_s']:.3f}s`",
        f"- Matched notes: `{default_note_comparison['matched_count']}`",
        f"- Notes only in TF: `{len(default_note_comparison['tf_only'])}`",
        f"- Notes only in CoreML: `{len(default_note_comparison['coreml_only'])}`",
        "",
        "### TF Extracted Notes",
        "| Pitch | Start (s) | Duration (s) | Amplitude |",
        "| --- | ---: | ---: | ---: |",
    ]

    if tf_notes_default:
        lines.extend(note_rows_to_markdown_rows(tf_notes_default))
    else:
        lines.append("| (none) | - | - | - |")

    lines += [
        "",
        "### CoreML Extracted Notes",
        "| Pitch | Start (s) | Duration (s) | Amplitude |",
        "| --- | ---: | ---: | ---: |",
    ]

    if coreml_notes_default:
        lines.extend(note_rows_to_markdown_rows(coreml_notes_default))
    else:
        lines.append("| (none) | - | - | - |")

    lines += [
        "",
        "### Notes Present In One Output But Not The Other",
        "- Only in TF:",
    ]

    if default_note_comparison["tf_only"]:
        for note in default_note_comparison["tf_only"]:
            lines.append(
                f"  - {note['pitch_name']} ({note['pitch_midi']}), start={note['start_time_s']:.3f}s, "
                f"duration={note['duration_s']:.3f}s"
            )
    else:
        lines.append("  - (none)")

    lines.append("- Only in CoreML:")
    if default_note_comparison["coreml_only"]:
        for note in default_note_comparison["coreml_only"]:
            lines.append(
                f"  - {note['pitch_name']} ({note['pitch_midi']}), start={note['start_time_s']:.3f}s, "
                f"duration={note['duration_s']:.3f}s"
            )
    else:
        lines.append("  - (none)")

    lines += [
        "",
        "## Part C - CoreML Threshold Sensitivity (Onset Threshold Sweep)",
        "- Sweep range: `0.20` to `0.60` in steps of `0.05`",
        f"- Frame threshold held at default: `{default_thresholds['frame_threshold']}`",
        "- Selection rule:",
        f"  - `{threshold_sweep['selection_rule']}`",
        "",
        "| Threshold | CoreML Notes | Matched | TF-only | CoreML-only | Mismatch | Precision | Recall | F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in threshold_sweep["rows"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['note_count']} | {row['matched_count']} | "
            f"{row['tf_only_count']} | {row['coreml_only_count']} | {row['mismatch_count']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )

    best = threshold_sweep["best"]
    lines += [
        "",
        f"- Adjusted CoreML onset threshold candidate: `{best['threshold']:.2f}`",
        f"- At that threshold: matched=`{best['matched_count']}`, mismatch=`{best['mismatch_count']}`, F1=`{best['f1']:.3f}`",
    ]

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-source",
        default="/Users/jamesrobertson/Downloads/Cedarcrest Blvd 955.m4a",
        help="Real guitar source recording path (wav/m4a/etc).",
    )
    parser.add_argument(
        "--audio-wav-output",
        default="cqt_spike/step3_real_audio_input.wav",
        help="Converted WAV path used for inference.",
    )
    parser.add_argument("--basic-pitch-repo", default=".tmp/basic-pitch", help="Local clone path for basic-pitch.")
    parser.add_argument("--output-dir", default="cqt_spike", help="Output directory.")
    parser.add_argument(
        "--coreml-model",
        default="cqt_spike/BasicPitch_from_tf.mlpackage",
        help="CoreML model path to use for comparison.",
    )
    parser.add_argument("--overlap-frames", type=int, default=30, help="Number of overlapping frames.")
    parser.add_argument(
        "--report-name",
        default="step3_real_audio_comparison.md",
        help="Markdown report filename under output-dir.",
    )
    parser.add_argument(
        "--summary-name",
        default="step3_real_audio_comparison.json",
        help="JSON summary filename under output-dir.",
    )
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

    audio_source_path = Path(args.audio_source).resolve()
    if not audio_source_path.exists():
        raise FileNotFoundError(f"Audio source not found: {audio_source_path}")

    coreml_model_path = Path(args.coreml_model).resolve()
    if not coreml_model_path.exists():
        raise FileNotFoundError(f"CoreML model not found: {coreml_model_path}")

    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
        from basic_pitch.inference import (
            DEFAULT_FRAME_THRESHOLD,
            DEFAULT_MINIMUM_NOTE_LENGTH_MS,
            DEFAULT_ONSET_THRESHOLD,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import basic_pitch runtime. Use Python 3.10/3.11 and install requirements-spike.txt."
        ) from exc

    wav_output_path = Path(args.audio_wav_output).resolve()
    audio_info = prepare_audio_wav(audio_source_path, wav_output_path, AUDIO_SAMPLE_RATE)

    tf_run = run_tf_windows(wav_output_path, Path(ICASSP_2022_MODEL_PATH), args.overlap_frames)
    coreml_run = run_coreml_windows(wav_output_path, coreml_model_path, args.overlap_frames)

    input_parity = compare_inputs(tf_run["inputs"], coreml_run["inputs"])

    thresholds_for_flip = [0.3, 0.4, 0.5, 0.6]
    single_window_metrics = note_tensor_diff_metrics(
        tf_run["outputs"]["note"][0][0],
        coreml_run["outputs"]["note"][0][0],
        thresholds_for_flip,
    )

    tf_agg = aggregate_outputs(
        tf_run["outputs"],
        tf_run["meta"]["audio_original_length_samples"],
        args.overlap_frames,
        tf_run["meta"]["hop_size_samples"],
    )
    coreml_agg = aggregate_outputs(
        coreml_run["outputs"],
        coreml_run["meta"]["audio_original_length_samples"],
        args.overlap_frames,
        coreml_run["meta"]["hop_size_samples"],
    )

    full_pipeline_metrics = note_tensor_diff_metrics(
        tf_agg["note"],
        coreml_agg["note"],
        thresholds_for_flip,
    )

    histogram = histogram_0_to_1_10bins(tf_agg["note"])

    min_note_len_frames = int(round(DEFAULT_MINIMUM_NOTE_LENGTH_MS / 1000.0 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    tf_notes_default = extract_notes(tf_agg, DEFAULT_ONSET_THRESHOLD, DEFAULT_FRAME_THRESHOLD, min_note_len_frames)
    coreml_notes_default = extract_notes(
        coreml_agg,
        DEFAULT_ONSET_THRESHOLD,
        DEFAULT_FRAME_THRESHOLD,
        min_note_len_frames,
    )
    default_note_comparison = compare_note_lists(tf_notes_default, coreml_notes_default)

    threshold_sweep = sweep_coreml_onset_thresholds(
        coreml_agg,
        tf_notes_default,
        frame_threshold=DEFAULT_FRAME_THRESHOLD,
        min_note_len_frames=min_note_len_frames,
    )

    report_path = output_dir / args.report_name
    default_thresholds = {
        "onset_threshold": float(DEFAULT_ONSET_THRESHOLD),
        "frame_threshold": float(DEFAULT_FRAME_THRESHOLD),
        "minimum_note_length_ms": float(DEFAULT_MINIMUM_NOTE_LENGTH_MS),
    }
    write_report(
        report_path,
        audio_info,
        coreml_model_path,
        args.overlap_frames,
        input_parity,
        histogram,
        single_window_metrics,
        full_pipeline_metrics,
        default_thresholds,
        tf_notes_default,
        coreml_notes_default,
        default_note_comparison,
        threshold_sweep,
    )

    summary = {
        "audio_info": audio_info,
        "coreml_model_path": str(coreml_model_path),
        "overlap_frames": int(args.overlap_frames),
        "tf_meta": tf_run["meta"],
        "coreml_meta": coreml_run["meta"],
        "input_parity": input_parity,
        "single_window_metrics": single_window_metrics,
        "full_pipeline_metrics": full_pipeline_metrics,
        "tf_histogram": histogram,
        "default_thresholds": default_thresholds,
        "default_note_comparison": default_note_comparison,
        "tf_notes_default": tf_notes_default,
        "coreml_notes_default": coreml_notes_default,
        "threshold_sweep": {
            "selection_rule": threshold_sweep["selection_rule"],
            "best": {
                k: v
                for k, v in threshold_sweep["best"].items()
                if k not in ("comparison", "notes")
            },
            "rows": [
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("comparison", "notes")
                }
                for row in threshold_sweep["rows"]
            ],
        },
    }

    summary_path = output_dir / args.summary_name
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
