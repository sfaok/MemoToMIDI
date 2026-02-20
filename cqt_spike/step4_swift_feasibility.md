# Step 4 Swift Feasibility

## Scope Note
- `cqt_spike/standalone_preprocess.py` is not present in this workspace.
- This assessment reconstructs the standalone preprocessing path from the exact Basic Pitch runtime/model code used by Step 3:
  - `.venv/lib/python3.9/site-packages/basic_pitch/inference.py`
  - `.venv/lib/python3.9/site-packages/basic_pitch/models.py`
  - `.venv/lib/python3.9/site-packages/basic_pitch/layers/nnaudio.py`
  - `.venv/lib/python3.9/site-packages/basic_pitch/layers/signal.py`
  - `.venv/lib/python3.9/site-packages/basic_pitch/nn.py`
- Also note: current `cqt_spike/BasicPitch_from_tf.mlpackage` input is `[1, 43844, 1]` (raw waveform window). This Step 4 document targets the external-preprocessing variant you requested (CQT + log norm + BN + harmonic stack).

## Part A - DSP Breakdown (In Order)
Assumed per-window input: `float32` mono waveform, shape `[1, 43844, 1]`.

| # | Operation | Python Function(s) | Shape In -> Out | Accelerate/vDSP Direct Equivalent | Swift Workaround if No Direct Equivalent |
| ---: | --- | --- | --- | --- | --- |
| 1 | Decode audio, force mono, resample to 22050 Hz | `librosa.load(path, sr=22050, mono=True)` | `[n_src]` -> `[n_22050]` | No exact librosa-equivalent in vDSP | Implement polyphase sinc resampler with precomputed taps + `vDSP_desamp`, or constrain capture path to native 22050 Hz mono to skip resampling. |
| 2 | Add front context for overlap pipeline | `np.concatenate([zeros(overlap/2), audio])` | `[n_22050]` -> `[n_22050 + 3840]` | Yes (`vDSP_vclr` + memcpy) | N/A |
| 3 | Window stream with fixed hop | loop in `window_audio_file(...)` | `[n_stream]` -> windows of length `<=43844` | No single API | Manual slicing/indexing in Swift arrays. |
| 4 | Right-pad final short window with zeros | `np.pad(window, [0, 43844-len])` | `[<=43844]` -> `[43844]` | Yes (`vDSP_vclr` + copy) | N/A |
| 5 | Add batch/channel dims for model-style tensor | `np.expand_dims(..., axis=-1)` and `np.expand_dims(..., axis=0)` | `[43844]` -> `[1, 43844, 1]` | N/A (metadata only) | N/A |
| 6 | Remove channel dim before CQT | `nn.FlattenAudioCh()` | `[1, 43844, 1]` -> `[1, 43844]` | N/A | N/A |
| 7 | Prepare CQT input rank | `CQT2010v2.reshape_input` | `[1, 43844]` -> `[1, 1, 43844]` | N/A | N/A |
| 8 | Build CQT kernels/filters (offline math) | `create_cqt_kernels`, `create_lowpass_filter`, `get_early_downsample_params` | params -> kernels | No single runtime equivalent | Precompute once in Python and bundle binaries (done in Part D). |
| 9 | Reflection pad before conv | `ReflectionPad1D(...)->tf.pad(..., mode="REFLECT")` | `[1,1,43844]` -> `[1,1,44100]` | No one-shot reflect-pad helper in vDSP | Custom mirror-copy pad routine in Swift. |
| 10 | Complex CQT conv for top octave (real + imag banks) | `get_cqt_complex` using `tf.nn.conv1d` twice | `[1,1,44100]` -> `[1,36,172,2]` | No direct multi-filter complex CQT op | Use `BNNS` 1D conv or im2col + `cblas_sgemm`; keep real/imag kernels separate. |
| 11 | Multi-octave recursion: lowpass+downsample by 2, repeat CQT, concat | `downsampling_by_n`, `get_cqt_complex`, `tf.concat` loop | top octave tensor -> `[1,324,172,2]` then crop | Partial (`vDSP_desamp` for FIR decimation) | Implement octave loop manually, then concatenate octave blocks. |
| 12 | Keep target CQT bins | `CQT = CQT[:, -n_bins:, :]` with `n_bins=309` | `[1,324,172,2]` -> `[1,309,172,2]` | Yes (strided copy) | N/A |
| 13 | Librosa-style magnitude normalization | `CQT *= sqrt(lengths)` (+ downsample factor) | `[1,309,172,2]` -> same | Yes (`vDSP_vmul`) | N/A |
| 14 | Magnitude + layout change to time-major | `sqrt(sum(CQT^2, axis=-1))`, then `tf.transpose(...,[0,2,1])` | `[1,309,172,2]` -> `[1,172,309]` | Yes (`vDSP_zvabs` or real/imag squared sum + `vvsqrtf`) | N/A |
| 15 | Log-power min-max normalization to [0,1] | `signal.NormalizedLog.call` (`square`, `log10`, per-example min/max normalize) | `[1,172,309]` -> `[1,172,309]` | Yes (`vDSP_vsq`, `vvlog10f`, `vDSP_minv`, `vDSP_maxv`, `vDSP_vsadd`, `vDSP_vsdiv`) | N/A |
| 16 | Add channel dim | `tf.expand_dims(x, -1)` | `[1,172,309]` -> `[1,172,309,1]` | N/A | N/A |
| 17 | Apply preprocessing batch norm (inference mode) | `tf.keras.layers.BatchNormalization` | `[1,172,309,1]` -> same | Yes (`vDSP_vsmsa` with fused scale+bias) | N/A |
| 18 | Harmonic stacking by integer bin shifts and zero pad | `nn.HarmonicStacking.call` (`tf.pad`, `tf.concat`, crop to first 264 bins) | `[1,172,309,1]` -> `[1,172,264,8]` | No one-call op | Use precomputed source-index + mask maps; gather/copy per harmonic channel. |

### Model-Specific Constants (from current Basic Pitch config)
- `sample_rate = 22050`
- `hop = 256`
- `window_samples = 43844`
- `cqt_bins_before_harmonic = 309`
- `harmonic_output_bins = 264`
- `harmonics = [0.5, 1, 2, 3, 4, 5, 6, 7]`
- `harmonic_shifts = [-36, 0, 36, 57, 72, 84, 93, 101]`
- `early_downsample_enabled = false` for this exact config

## Part B - Hard Parts (Ranked)

| Rank | Hard Part | Why It Is Hard | Direct Accelerate Equivalent? | Precompute in Python? | Swift Runtime After Precompute |
| ---: | --- | --- | --- | --- | --- |
| 1 | Bit-exact librosa-style resampling | `librosa.load(..., sr=22050)` uses high-quality resampling behavior that is hard to match sample-for-sample | No | Not generally (input-dependent) | Either avoid runtime resampling (record/store at 22050 mono) or implement custom sinc/polyphase resampler with fixed taps. |
| 2 | CQT engine (complex conv banks across octaves with reflect padding) | Multi-stage conv/decimation pipeline with strict padding/stride behavior | No single op | Yes (kernels, filters, normalization vectors) | Mainly FIR/conv + concat + elementwise math; all doable with vDSP/BNNS/cblas. |
| 3 | Reflection padding semantics | Exact mirror indexing must match TensorFlow pad behavior | No one-shot high-level op | Not needed | Small custom pad function in Swift. |
| 4 | Harmonic stacking | Frequency shifts + zero-padding + channel concat; easy to get indexing wrong | No single op | Yes (source-index + mask lookup tables) | Indexed gather/copy into output channels. |
| 5 | BatchNorm parity | Requires exact inference constants | Yes (`vsmsa`) once constants known | Yes (fused scale/bias constants) | One vector affine transform. |

### About “fractional bin interpolation”
- In this Basic Pitch configuration, harmonic shifts are integer bin offsets (`round(12 * 3 * log2(h))`), so fractional interpolation is not required.

## Part C - Estimate

### 1) Realistic Swift LOC
- Core preprocessing implementation: `~750-950` LOC
- Breakdown:
  - windowing + tensor plumbing: `~120-180`
  - CQT conv/downsample engine: `~350-450`
  - normalization + BN + harmonic stack: `~150-220`
  - file loader + validation harness against Python assets: `~120-180`

### 2) Assets to Precompute and Bundle
- `cqt_top_octave_kernel_real_f32le.bin`
- `cqt_top_octave_kernel_imag_f32le.bin`
- `cqt_lowpass_filter_256_f32le.bin`
- `cqt_librosa_norm_sqrt_lengths_f32le.bin`
- `harmonic_shifts_i32le.bin`
- `harmonic_source_index_i32le.bin`
- `harmonic_source_mask_f32le.bin`
- `preprocess_batchnorm_fused_scale_bias_f32le.bin`
- `metadata.json` (shapes/constants/byte layout)

### 3) Expected Numerical Diff vs Python Reference
- If input is already `22050 Hz` mono float32 and Swift matches reflect pad + kernels + order of ops:
  - expected preprocess max abs diff: about `1e-4` to `5e-4`
  - expected preprocess mean abs diff: about `1e-6` to `1e-5`
- If runtime resampling differs from librosa:
  - max abs diff can reach `1e-2` in some bins/frames, and downstream note decisions can shift near threshold.

### 4) Overall Feasibility
- **YELLOW**
- Reason:
  - Performance target (`<2s` preprocessing for 30s audio on A15) is achievable with precomputed kernels and vectorized conv/decimation.
  - Main risk is parity drift from resampling and exact padding/conv semantics, not raw compute.

## Part D - Precomputed Assets

### Script Added
- `scripts/step4_precompute_assets.py`

### Run Command
```bash
.venv/bin/python scripts/step4_precompute_assets.py
```

### Output Directory
- `cqt_spike/precomputed/`

### File Formats
All binary files are raw, little-endian, row-major (`C` order), no header.

| File | DType | Shape | Bytes |
| --- | --- | --- | ---: |
| `cqt_top_octave_kernel_real_f32le.bin` | `float32` | `[36, 256]` | 36864 |
| `cqt_top_octave_kernel_imag_f32le.bin` | `float32` | `[36, 256]` | 36864 |
| `cqt_lowpass_filter_256_f32le.bin` | `float32` | `[256]` | 1024 |
| `cqt_librosa_norm_sqrt_lengths_f32le.bin` | `float32` | `[309]` | 1236 |
| `harmonic_shifts_i32le.bin` | `int32` | `[8]` | 32 |
| `harmonic_source_index_i32le.bin` | `int32` | `[8, 264]` | 8448 |
| `harmonic_source_mask_f32le.bin` | `float32` | `[8, 264]` | 8448 |
| `preprocess_batchnorm_fused_scale_bias_f32le.bin` | `float32` | `[2]` (`[scale, bias]`) | 8 |
| `metadata.json` | JSON | manifest/constants | 3914 |

### Metadata of Note
From `cqt_spike/precomputed/metadata.json`:
- `n_cqt_bins = 309`
- `n_octaves = 9`
- `n_fft_top_octave = 256`
- `early_downsample_enabled = false`
- `harmonic_shifts_bins = [-36, 0, 36, 57, 72, 84, 93, 101]`
- preprocess BN fused constants:
  - `scale = 2.4807409894987282`
  - `bias = -0.8769182627750702`
