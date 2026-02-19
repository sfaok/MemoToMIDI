#!/usr/bin/env python
"""Step 2: run Basic Pitch pipeline, capture reference tensors, and try CoreML conversion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def require_python_compat() -> None:
    # Basic Pitch officially supports Python <= 3.11.
    if sys.version_info >= (3, 12):
        print(
            "Warning: Python 3.12+ detected. Basic Pitch dependencies may fail. "
            "Use Python 3.10 or 3.11 for reliable setup."
        )


def to_jsonable_note_events(note_events: List[Tuple[float, float, int, float, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for start, end, pitch, velocity, bends in note_events:
        out.append(
            {
                "pitch": int(pitch),
                "start_time_s": float(start),
                "end_time_s": float(end),
                "duration_s": float(end - start),
                "velocity": float(velocity),
                "pitch_bends": [int(x) for x in bends] if bends else None,
            }
        )
    return out


def shape_dtype_min_max(arr: Any) -> Dict[str, Any]:
    import numpy as np

    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def generate_synthetic_chord(audio_path: Path, sample_rate: int = 44100, duration_s: float = 3.0) -> None:
    import numpy as np
    from scipy.io import wavfile

    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    freqs = [196.0, 246.9, 293.7]  # G3, B3, D4

    signal = np.zeros_like(t, dtype=np.float32)
    for freq in freqs:
        signal += (1.0 / len(freqs)) * np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    # Short fade in/out to avoid clicks.
    fade_len = int(0.02 * sample_rate)
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    signal[:fade_len] *= fade
    signal[-fade_len:] *= fade[::-1]

    peak = float(np.max(np.abs(signal))) or 1.0
    signal = 0.9 * (signal / peak)

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(audio_path, sample_rate, signal.astype(np.float32))


def run_coreml_conversion(
    output_dir: Path,
    reference_input_first_window: Any,
    reference_batched_outputs: Dict[str, Any],
    tf_model_path: Path,
    tflite_model_path: Path,
) -> Dict[str, Any]:
    conversion_report: Dict[str, Any] = {"attempts": []}
    try:
        import coremltools as ct
    except Exception as exc:
        conversion_report["status"] = "skipped"
        conversion_report["reason"] = f"coremltools import failed: {exc!r}"
        return conversion_report

    def io_spec(mlmodel: Any) -> Dict[str, Any]:
        spec = mlmodel.get_spec()
        io: Dict[str, Any] = {"inputs": [], "outputs": []}

        for tensor in spec.description.input:
            entry: Dict[str, Any] = {"name": tensor.name, "type": tensor.type.WhichOneof("Type")}
            if entry["type"] == "multiArrayType":
                entry["shape"] = list(tensor.type.multiArrayType.shape)
            io["inputs"].append(entry)

        for tensor in spec.description.output:
            entry = {"name": tensor.name, "type": tensor.type.WhichOneof("Type")}
            if entry["type"] == "multiArrayType":
                entry["shape"] = list(tensor.type.multiArrayType.shape)
            io["outputs"].append(entry)

        return io

    def map_output_name(name: str, shape: Tuple[int, ...], assigned: Dict[str, str]) -> str:
        lower = name.lower()
        if "onset" in lower or name == "Identity_2":
            return "onset"
        if "contour" in lower or name == "Identity":
            return "contour"
        if "note" in lower or name == "Identity_1":
            return "note"
        if shape and shape[-1] == 264:
            return "contour"
        if shape and shape[-1] == 88:
            if "note" not in assigned:
                return "note"
            return "onset"
        return name

    for label, source_path in [("tf", tf_model_path), ("tflite", tflite_model_path)]:
        attempt: Dict[str, Any] = {"source": label, "path": str(source_path)}
        if not source_path.exists():
            attempt["status"] = "skipped"
            attempt["reason"] = "source model path not found"
            conversion_report["attempts"].append(attempt)
            continue

        try:
            mlmodel = ct.convert(str(source_path))
            out_path = output_dir / f"BasicPitch_from_{label}.mlpackage"
            mlmodel.save(str(out_path))
            attempt["status"] = "success"
            attempt["saved_model"] = str(out_path)
            attempt["io"] = io_spec(mlmodel)
        except Exception as exc:
            attempt["status"] = "failed"
            attempt["error"] = repr(exc)
            conversion_report["attempts"].append(attempt)
            continue

        try:
            input_name = attempt["io"]["inputs"][0]["name"]
            pred = mlmodel.predict({input_name: reference_input_first_window})
            diffs: Dict[str, float] = {}
            assigned: Dict[str, str] = {}
            for out_name, out_val in pred.items():
                mapped_name = map_output_name(out_name, tuple(out_val.shape), assigned)
                assigned[mapped_name] = out_name
                if mapped_name in reference_batched_outputs:
                    target = reference_batched_outputs[mapped_name][:1]
                    import numpy as np

                    diffs[mapped_name] = float(np.max(np.abs(out_val - target)))
            attempt["prediction_compare"] = {
                "status": "ok",
                "max_abs_diff": diffs,
            }
        except Exception as exc:
            attempt["prediction_compare"] = {
                "status": "failed",
                "error": repr(exc),
            }

        conversion_report["attempts"].append(attempt)

    conversion_report["status"] = "done"
    return conversion_report


def write_markdown_report(
    report_path: Path,
    audio_path: Path,
    artifacts: Dict[str, Path],
    tensor_info: Dict[str, Any],
    conversion_report: Dict[str, Any],
) -> None:
    lines: List[str] = [
        "# CQT Spike Report",
        "",
        "## Inputs",
        f"- Audio source: `{audio_path}`",
        "",
        "## Saved Artifacts",
    ]

    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")

    lines += ["", "## Tensor Shapes/Dtypes"]
    for name, info in tensor_info.items():
        lines.append(
            f"- `{name}`: shape={info['shape']}, dtype={info['dtype']}, min={info['min']:.6f}, max={info['max']:.6f}"
        )

    lines += ["", "## CoreML Conversion"]
    lines.append(f"- Status: `{conversion_report.get('status', 'unknown')}`")
    if "reason" in conversion_report:
        lines.append(f"- Reason: `{conversion_report['reason']}`")
    for attempt in conversion_report.get("attempts", []):
        lines.append(f"- Attempt `{attempt.get('source')}`: `{attempt.get('status')}`")
        if "error" in attempt:
            lines.append(f"  - Error: `{attempt['error']}`")
        if "saved_model" in attempt:
            lines.append(f"  - Saved model: `{attempt['saved_model']}`")
        if "prediction_compare" in attempt:
            lines.append(f"  - Compare: `{attempt['prediction_compare']}`")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=str, default=None, help="Optional input wav/mp3/flac path.")
    parser.add_argument("--basic-pitch-repo", default=".tmp/basic-pitch", help="Local clone path of basic-pitch.")
    parser.add_argument("--output-dir", default="cqt_spike", help="Output directory for spike artifacts.")
    parser.add_argument(
        "--skip-coreml-convert", action="store_true", help="Skip CoreML conversion attempts even if coremltools exists."
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

    try:
        import librosa
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "Missing required Python packages. Install with: pip install -r requirements-spike.txt"
        ) from exc

    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH, FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP
        from basic_pitch.inference import DEFAULT_OVERLAPPING_FRAMES, Model, get_audio_input, predict, run_inference
    except Exception as exc:
        raise RuntimeError(
            "Failed to import basic_pitch runtime stack. "
            "Use Python 3.10/3.11 and install requirements-spike.txt."
        ) from exc

    if args.audio:
        audio_path = Path(args.audio).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    else:
        audio_path = output_dir / "synthetic_g_major.wav"
        generate_synthetic_chord(audio_path)

    resampled_audio, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    reference_audio_path = output_dir / "reference_audio.npy"
    np.save(reference_audio_path, resampled_audio.astype(np.float32))

    overlap_len = DEFAULT_OVERLAPPING_FRAMES * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    windows: List[np.ndarray] = []
    original_length = 0
    for audio_windowed, _window_time, this_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        windows.append(audio_windowed.astype(np.float32))
        original_length = this_original_length

    if not windows:
        raise RuntimeError("No inference windows were generated from input audio.")

    reference_input = np.concatenate(windows, axis=0)
    reference_input_path = output_dir / "reference_input.npy"
    np.save(reference_input_path, reference_input.astype(np.float32))

    model = Model(ICASSP_2022_MODEL_PATH)

    raw_batched_output_lists: Dict[str, List[np.ndarray]] = {"note": [], "onset": [], "contour": []}
    for window in windows:
        pred = model.predict(window)
        for key in raw_batched_output_lists.keys():
            raw_batched_output_lists[key].append(pred[key])

    raw_batched_outputs: Dict[str, np.ndarray] = {}
    for key, values in raw_batched_output_lists.items():
        raw_batched_outputs[key] = np.concatenate(values, axis=0)
        np.save(output_dir / f"reference_{key}_output_batched.npy", raw_batched_outputs[key].astype(np.float32))

    unwrapped_output = run_inference(audio_path, model)
    reference_note_path = output_dir / "reference_note_output.npy"
    reference_onset_path = output_dir / "reference_onset_output.npy"
    reference_contour_path = output_dir / "reference_contour_output.npy"
    np.save(reference_note_path, unwrapped_output["note"].astype(np.float32))
    np.save(reference_onset_path, unwrapped_output["onset"].astype(np.float32))
    np.save(reference_contour_path, unwrapped_output["contour"].astype(np.float32))

    model_output, midi_data, note_events = predict(audio_path, model)
    reference_midi_path = output_dir / "reference_output.mid"
    midi_data.write(str(reference_midi_path))
    note_events_path = output_dir / "reference_note_events.json"
    note_events_path.write_text(json.dumps(to_jsonable_note_events(note_events), indent=2), encoding="utf-8")

    conversion_report: Dict[str, Any]
    if args.skip_coreml_convert:
        conversion_report = {"status": "skipped", "reason": "--skip-coreml-convert enabled"}
    else:
        conversion_report = run_coreml_conversion(
            output_dir=output_dir,
            reference_input_first_window=reference_input[:1].astype(np.float32),
            reference_batched_outputs=raw_batched_outputs,
            tf_model_path=build_icassp_2022_model_path(FilenameSuffix.tf),
            tflite_model_path=build_icassp_2022_model_path(FilenameSuffix.tflite),
        )

    metadata = {
        "audio_path": str(audio_path),
        "model_path": str(ICASSP_2022_MODEL_PATH),
        "audio_original_length_samples": int(original_length),
        "num_windows": len(windows),
        "overlap_len_samples": overlap_len,
        "hop_size_samples": hop_size,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    conversion_path = output_dir / "conversion_report.json"
    conversion_path.write_text(json.dumps(conversion_report, indent=2), encoding="utf-8")

    report_path = output_dir / "report.md"
    artifacts = {
        "reference_audio": reference_audio_path,
        "reference_input": reference_input_path,
        "reference_note_output": reference_note_path,
        "reference_onset_output": reference_onset_path,
        "reference_contour_output": reference_contour_path,
        "reference_note_output_batched": output_dir / "reference_note_output_batched.npy",
        "reference_onset_output_batched": output_dir / "reference_onset_output_batched.npy",
        "reference_contour_output_batched": output_dir / "reference_contour_output_batched.npy",
        "reference_midi": reference_midi_path,
        "reference_note_events": note_events_path,
        "metadata": metadata_path,
        "conversion_report": conversion_path,
    }
    tensor_info = {
        "reference_audio": shape_dtype_min_max(resampled_audio),
        "reference_input": shape_dtype_min_max(reference_input),
        "reference_note_output": shape_dtype_min_max(unwrapped_output["note"]),
        "reference_onset_output": shape_dtype_min_max(unwrapped_output["onset"]),
        "reference_contour_output": shape_dtype_min_max(unwrapped_output["contour"]),
        "model_output_note": shape_dtype_min_max(model_output["note"]),
        "model_output_onset": shape_dtype_min_max(model_output["onset"]),
        "model_output_contour": shape_dtype_min_max(model_output["contour"]),
    }
    write_markdown_report(report_path, audio_path, artifacts, tensor_info, conversion_report)

    print(f"Wrote artifacts to {output_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
