#!/usr/bin/env python
"""Step 1: inspect basic-pitch source and write a structured report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def find_line(path: Path, needle: str) -> int:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return line_number
    raise ValueError(f"Could not find '{needle}' in {path}")


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--basic-pitch-repo",
        default=".tmp/basic-pitch",
        help="Path to a local clone of spotify/basic-pitch.",
    )
    parser.add_argument(
        "--out",
        default="cqt_spike/step1_report.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    repo = Path(args.basic_pitch_repo).resolve()
    out = Path(args.out).resolve()

    if not repo.exists():
        raise FileNotFoundError(
            f"basic-pitch repo not found at {repo}. Clone it first: "
            "git clone https://github.com/spotify/basic-pitch .tmp/basic-pitch"
        )

    inference_py = repo / "basic_pitch" / "inference.py"
    constants_py = repo / "basic_pitch" / "constants.py"
    models_py = repo / "basic_pitch" / "models.py"
    nn_py = repo / "basic_pitch" / "nn.py"
    init_py = repo / "basic_pitch" / "__init__.py"

    for path in [inference_py, constants_py, models_py, nn_py, init_py]:
        if not path.exists():
            raise FileNotFoundError(f"Expected file missing: {path}")

    ln_load = find_line(inference_py, "librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)")
    ln_model_predict = find_line(inference_py, "model.predict(audio_windowed)")
    ln_input_shape = find_line(models_py, "inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))")
    ln_get_cqt = find_line(models_py, "x = get_cqt(inputs, n_harmonics, True)")
    ln_cqt_layer = find_line(models_py, "x = nnaudio.CQT(")
    ln_norm = find_line(models_py, "x = signal.NormalizedLog()(x)")
    ln_harmonics = find_line(models_py, "[0.5] + list(range(1, n_harmonics))")

    ln_sr = find_line(constants_py, "AUDIO_SAMPLE_RATE = 22050")
    ln_fft_hop = find_line(constants_py, "FFT_HOP = 256")
    ln_frames = find_line(constants_py, "ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH")
    ln_audio_samples = find_line(constants_py, "AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP")
    ln_freq_note_bins = find_line(constants_py, "N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE")
    ln_freq_contour_bins = find_line(
        constants_py, "N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE"
    )

    ln_model_outputs = find_line(models_py, 'outputs = {"onset": x_onset, "contour": x_contours, "note": x_notes}')
    ln_note_sigmoid = find_line(models_py, "activation=\"sigmoid\",")
    ln_flatten_onset = find_line(models_py, "x_onset = nn.FlattenFreqCh(")
    ln_flatten_note = find_line(models_py, "x_notes = nn.FlattenFreqCh(name=note_name)(x_notes_pre)")
    ln_flatten_contour = find_line(models_py, "x_contours = nn.FlattenFreqCh(name=contour_name)(x_contours)")

    ln_saved_model = find_line(init_py, 'tf = "nmp"')
    ln_coreml = find_line(init_py, 'coreml = "nmp.mlpackage"')
    ln_tflite = find_line(init_py, 'tflite = "nmp.tflite"')
    ln_onnx = find_line(init_py, 'onnx = "nmp.onnx"')

    ln_harmonic_shifts = find_line(nn_py, "self.shifts = [")
    ln_shift_formula = find_line(nn_py, "log_base_b(float(h), 2)")

    report_lines = [
        "# Step 1 Report: Basic Pitch Source Inspection",
        "",
        "## Repository",
        f"- Source: `{repo}`",
        "",
        "## 1) Preprocessing Location",
        "- Result: preprocessing is inside the model graph for TensorFlow-based execution.",
        f"- Audio is loaded/resampled in `{rel(inference_py, repo)}:{ln_load}`.",
        f"- Inference passes raw windowed audio into `model.predict(...)` in `{rel(inference_py, repo)}:{ln_model_predict}`.",
        f"- Model input is raw audio `(batch, AUDIO_N_SAMPLES, 1)` in `{rel(models_py, repo)}:{ln_input_shape}`.",
        f"- CQT is computed in-model via `get_cqt(...)` in `{rel(models_py, repo)}:{ln_get_cqt}` and `nnaudio.CQT(...)` in `{rel(models_py, repo)}:{ln_cqt_layer}`.",
        f"- Log normalization is also in-model in `{rel(models_py, repo)}:{ln_norm}`.",
        "",
        "## 2) Input Spec",
        f"- Sample rate: `22050 Hz` in `{rel(constants_py, repo)}:{ln_sr}`.",
        f"- Hop size: `256` samples in `{rel(constants_py, repo)}:{ln_fft_hop}`.",
        f"- Model input window samples: `AUDIO_N_SAMPLES = 43844` in `{rel(constants_py, repo)}:{ln_audio_samples}`.",
        f"- Model input tensor shape: `(batch, 43844, 1)` from `{rel(models_py, repo)}:{ln_input_shape}`.",
        "- Input tensor dtype at inference: `float32` (numpy arrays from librosa).",
        "- Input is raw audio, not precomputed spectrogram.",
        "- In-model spectrogram path:",
        "  - CQT via `nnaudio.CQT`.",
        "  - `fmin=27.5`, `hop_length=256`, `bins_per_octave=36`.",
        "  - `n_bins = n_semitones * 3` where `n_semitones = ceil(12*log2(n_harmonics)) + 88` capped by Nyquist.",
        f"  - Harmonic channels for default `n_harmonics=8`: `[0.5, 1, 2, 3, 4, 5, 6, 7]` in `{rel(models_py, repo)}:{ln_harmonics}`.",
        f"  - Harmonic shifts are integer bin shifts in `{rel(nn_py, repo)}:{ln_harmonic_shifts}` and `{rel(nn_py, repo)}:{ln_shift_formula}`.",
        "- Normalization in-model: power -> dB -> per-example min-max scaling to [0, 1].",
        "",
        "## 3) Output Spec",
        f"- Model outputs are `onset`, `contour`, `note` in `{rel(models_py, repo)}:{ln_model_outputs}`.",
        f"- Output activations are sigmoid in `{rel(models_py, repo)}:{ln_note_sigmoid}` (applies to note/onset/contour heads).",
        "- Output value range: [0, 1].",
        f"- Frequency bins: notes/onsets `88` (`{rel(constants_py, repo)}:{ln_freq_note_bins}`), contour `264` (`{rel(constants_py, repo)}:{ln_freq_contour_bins}`).",
        f"- Window time frames: `ANNOT_N_FRAMES = 172` in `{rel(constants_py, repo)}:{ln_frames}`.",
        f"- Flattening to `(batch, time, freq)` is done in `{rel(models_py, repo)}:{ln_flatten_note}`, `{rel(models_py, repo)}:{ln_flatten_onset}`, `{rel(models_py, repo)}:{ln_flatten_contour}`.",
        "",
        "## 4) Model Formats",
        "- Available serializations are in `basic_pitch/saved_models/icassp_2022/`:",
        f"  - TensorFlow SavedModel (`nmp`) in `{rel(init_py, repo)}:{ln_saved_model}`.",
        f"  - CoreML (`nmp.mlpackage`) in `{rel(init_py, repo)}:{ln_coreml}`.",
        f"  - TFLite (`nmp.tflite`) in `{rel(init_py, repo)}:{ln_tflite}`.",
        f"  - ONNX (`nmp.onnx`) in `{rel(init_py, repo)}:{ln_onnx}`.",
        "",
        "## 5) Harmonic CQT Details",
        "- Harmonic stacking is performed after CQT+log normalization.",
        "- Implementation uses one base CQT then channel-stacks frequency-shifted copies.",
        "- Default harmonic multipliers: `[0.5, 1, 2, 3, 4, 5, 6, 7]`.",
        "- Shift formula: `round(12 * bins_per_semitone * log2(harmonic))`.",
        "- CQT config (default model):",
        "  - `fmin=27.5 Hz`",
        "  - `bins_per_octave=36` (3 bins/semitone)",
        "  - `hop_length=256`",
        "  - `sr=22050`",
        "",
        "## Initial Gate Result",
        "- Status: GREEN for preprocessing location risk.",
        "- Reason: CQT + normalization + harmonic stacking are embedded in model graph for TF model path.",
        "- Remaining risk: verify converted CoreML model keeps equivalent behavior and outputs.",
    ]

    write_report(out, report_lines)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
