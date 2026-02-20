#!/usr/bin/env python
"""Generate precomputed Basic Pitch preprocessing assets for Swift/Accelerate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.constants import (
    ANNOTATIONS_BASE_FREQUENCY,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    N_FREQ_BINS_CONTOURS,
)
from basic_pitch.layers import nnaudio
from basic_pitch.models import MAX_N_SEMITONES


def write_bin(path: Path, arr: np.ndarray, dtype: np.dtype) -> Dict[str, object]:
    cast = np.asarray(arr, dtype=dtype)
    le = cast.astype(cast.dtype.newbyteorder("<"), copy=False)
    path.write_bytes(le.tobytes(order="C"))
    return {
        "path": str(path),
        "dtype": str(cast.dtype),
        "byte_order": "little_endian",
        "shape": list(cast.shape),
        "layout": "C",
        "num_values": int(cast.size),
        "num_bytes": int(cast.nbytes),
    }


def compute_harmonic_maps(
    shifts: List[int], n_input_bins: int, n_output_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    src_index = np.zeros((len(shifts), n_output_bins), dtype=np.int32)
    src_mask = np.zeros((len(shifts), n_output_bins), dtype=np.float32)

    for h, shift in enumerate(shifts):
        for b in range(n_output_bins):
            src = b + shift
            if 0 <= src < n_input_bins:
                src_index[h, b] = src
                src_mask[h, b] = 1.0
            else:
                src_index[h, b] = 0
                src_mask[h, b] = 0.0

    return src_index, src_mask


def extract_preprocess_batchnorm_constants(model_path: Path) -> Dict[str, float]:
    saved_model = tf.saved_model.load(str(model_path))
    by_name = {var.name: var for var in saved_model.variables}

    gamma = float(by_name["batch_normalization/gamma:0"].numpy().reshape(-1)[0])
    beta = float(by_name["batch_normalization/beta:0"].numpy().reshape(-1)[0])
    moving_mean = float(by_name["batch_normalization/moving_mean:0"].numpy().reshape(-1)[0])
    moving_variance = float(by_name["batch_normalization/moving_variance:0"].numpy().reshape(-1)[0])
    epsilon = 1e-3

    fused_scale = gamma / math.sqrt(moving_variance + epsilon)
    fused_bias = beta - (moving_mean * fused_scale)

    return {
        "gamma": gamma,
        "beta": beta,
        "moving_mean": moving_mean,
        "moving_variance": moving_variance,
        "epsilon": epsilon,
        "fused_scale": fused_scale,
        "fused_bias": fused_bias,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="cqt_spike/precomputed",
        help="Directory where binary assets and metadata JSON are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_harmonics = 8
    harmonics = [0.5] + list(range(1, n_harmonics))
    bins_per_octave = 12 * CONTOURS_BINS_PER_SEMITONE
    filter_scale = 1.0
    basis_norm = 1
    n_output_freq_bins = N_FREQ_BINS_CONTOURS
    fmin = ANNOTATIONS_BASE_FREQUENCY

    n_semitones = min(int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES), MAX_N_SEMITONES)
    cqt_n_bins = n_semitones * CONTOURS_BINS_PER_SEMITONE
    n_filters = min(bins_per_octave, cqt_n_bins)
    n_octaves = int(np.ceil(float(cqt_n_bins) / bins_per_octave))
    Q = filter_scale / (2 ** (1 / bins_per_octave) - 1)

    fmin_t = fmin * 2 ** (n_octaves - 1)
    remainder = cqt_n_bins % bins_per_octave
    if remainder == 0:
        fmax_t = fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
    else:
        fmax_t = fmin_t * 2 ** ((remainder - 1) / bins_per_octave)
    fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave)

    sr_for_kernels, hop_for_cqt, downsample_factor, early_filter, early_enabled = nnaudio.get_early_downsample_params(
        AUDIO_SAMPLE_RATE,
        256,
        fmax_t,
        Q,
        n_octaves,
        np.float32,
    )

    top_octave_basis, n_fft, _top_lengths, _top_freqs = nnaudio.create_cqt_kernels(
        Q=Q,
        fs=sr_for_kernels,
        fmin=fmin_t,
        n_bins=n_filters,
        bins_per_octave=bins_per_octave,
        norm=basis_norm,
        topbin_check=False,
    )
    cqt_kernel_real = np.asarray(top_octave_basis.real, dtype=np.float32)
    cqt_kernel_imag = np.asarray(top_octave_basis.imag, dtype=np.float32)

    lowpass_filter = np.asarray(
        nnaudio.create_lowpass_filter(band_center=0.5, kernel_length=256, transition_bandwidth=0.001).numpy(),
        dtype=np.float32,
    )

    freqs = fmin * 2.0 ** (np.arange(cqt_n_bins) / float(bins_per_octave))
    lengths = np.ceil(Q * sr_for_kernels / freqs).astype(np.float32)
    librosa_norm_sqrt_lengths = np.sqrt(lengths).astype(np.float32)

    harmonic_shifts = np.asarray(
        [int(np.round(12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(float(h)))) for h in harmonics], dtype=np.int32
    )
    harmonic_src_index, harmonic_src_mask = compute_harmonic_maps(
        shifts=harmonic_shifts.tolist(),
        n_input_bins=cqt_n_bins,
        n_output_bins=n_output_freq_bins,
    )

    bn = extract_preprocess_batchnorm_constants(Path(ICASSP_2022_MODEL_PATH))
    bn_fused = np.asarray([bn["fused_scale"], bn["fused_bias"]], dtype=np.float32)

    files: Dict[str, Dict[str, object]] = {}
    files["cqt_top_octave_kernel_real_f32le.bin"] = write_bin(
        output_dir / "cqt_top_octave_kernel_real_f32le.bin", cqt_kernel_real, np.float32
    )
    files["cqt_top_octave_kernel_imag_f32le.bin"] = write_bin(
        output_dir / "cqt_top_octave_kernel_imag_f32le.bin", cqt_kernel_imag, np.float32
    )
    files["cqt_lowpass_filter_256_f32le.bin"] = write_bin(
        output_dir / "cqt_lowpass_filter_256_f32le.bin", lowpass_filter, np.float32
    )
    files["cqt_librosa_norm_sqrt_lengths_f32le.bin"] = write_bin(
        output_dir / "cqt_librosa_norm_sqrt_lengths_f32le.bin", librosa_norm_sqrt_lengths, np.float32
    )
    files["harmonic_shifts_i32le.bin"] = write_bin(output_dir / "harmonic_shifts_i32le.bin", harmonic_shifts, np.int32)
    files["harmonic_source_index_i32le.bin"] = write_bin(
        output_dir / "harmonic_source_index_i32le.bin", harmonic_src_index, np.int32
    )
    files["harmonic_source_mask_f32le.bin"] = write_bin(
        output_dir / "harmonic_source_mask_f32le.bin", harmonic_src_mask, np.float32
    )
    files["preprocess_batchnorm_fused_scale_bias_f32le.bin"] = write_bin(
        output_dir / "preprocess_batchnorm_fused_scale_bias_f32le.bin", bn_fused, np.float32
    )

    if early_enabled and early_filter is not None:
        files["cqt_early_downsample_filter_256_f32le.bin"] = write_bin(
            output_dir / "cqt_early_downsample_filter_256_f32le.bin",
            np.asarray(early_filter.numpy(), dtype=np.float32),
            np.float32,
        )

    metadata = {
        "version": 1,
        "source_model_path": str(ICASSP_2022_MODEL_PATH),
        "parameters": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE,
            "hop_length_samples": 256,
            "fmin_hz": fmin,
            "bins_per_octave": bins_per_octave,
            "n_harmonics": n_harmonics,
            "harmonics": harmonics,
            "n_semitones_cqt": n_semitones,
            "n_cqt_bins": cqt_n_bins,
            "n_output_freq_bins": n_output_freq_bins,
            "n_octaves": n_octaves,
            "n_filters_top_octave": n_filters,
            "n_fft_top_octave": int(n_fft),
            "Q": float(Q),
            "early_downsample_enabled": bool(early_enabled),
            "early_downsample_factor": int(downsample_factor),
            "sr_for_kernels_hz": float(sr_for_kernels),
            "hop_for_cqt": int(hop_for_cqt),
            "harmonic_shifts_bins": harmonic_shifts.tolist(),
        },
        "batchnorm_preprocess": bn,
        "files": files,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {len(files)} binary assets to: {output_dir}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
