# Step 1 Report: Basic Pitch Source Inspection

## Repository
- Source: `C:\Users\email\MemoToMIDI\.tmp\basic-pitch`

## 1) Preprocessing Location
- Result: preprocessing is inside the model graph for TensorFlow-based execution.
- Audio is loaded/resampled in `basic_pitch/inference.py:239`.
- Inference passes raw windowed audio into `model.predict(...)` in `basic_pitch/inference.py:309`.
- Model input is raw audio `(batch, AUDIO_N_SAMPLES, 1)` in `basic_pitch/models.py:210`.
- CQT is computed in-model via `get_cqt(...)` in `basic_pitch/models.py:211` and `nnaudio.CQT(...)` in `basic_pitch/models.py:179`.
- Log normalization is also in-model in `basic_pitch/models.py:186`.

## 2) Input Spec
- Sample rate: `22050 Hz` in `basic_pitch/constants.py:33`.
- Hop size: `256` samples in `basic_pitch/constants.py:25`.
- Model input window samples: `AUDIO_N_SAMPLES = 43844` in `basic_pitch/constants.py:47`.
- Model input tensor shape: `(batch, 43844, 1)` from `basic_pitch/models.py:210`.
- Input tensor dtype at inference: `float32` (numpy arrays from librosa).
- Input is raw audio, not precomputed spectrogram.
- In-model spectrogram path:
  - CQT via `nnaudio.CQT`.
  - `fmin=27.5`, `hop_length=256`, `bins_per_octave=36`.
  - `n_bins = n_semitones * 3` where `n_semitones = ceil(12*log2(n_harmonics)) + 88` capped by Nyquist.
  - Harmonic channels for default `n_harmonics=8`: `[0.5, 1, 2, 3, 4, 5, 6, 7]` in `basic_pitch/models.py:216`.
  - Harmonic shifts are integer bin shifts in `basic_pitch/nn.py:51` and `basic_pitch/nn.py:52`.
- Normalization in-model: power -> dB -> per-example min-max scaling to [0, 1].

## 3) Output Spec
- Model outputs are `onset`, `contour`, `note` in `basic_pitch/models.py:320`.
- Output activations are sigmoid in `basic_pitch/models.py:258` (applies to note/onset/contour heads).
- Output value range: [0, 1].
- Frequency bins: notes/onsets `88` (`basic_pitch/constants.py:35`), contour `264` (`basic_pitch/constants.py:36`).
- Window time frames: `ANNOT_N_FRAMES = 172` in `basic_pitch/constants.py:44`.
- Flattening to `(batch, time, freq)` is done in `basic_pitch/models.py:290`, `basic_pitch/models.py:316`, `basic_pitch/models.py:263`.

## 4) Model Formats
- Available serializations are in `basic_pitch/saved_models/icassp_2022/`:
  - TensorFlow SavedModel (`nmp`) in `basic_pitch/__init__.py:75`.
  - CoreML (`nmp.mlpackage`) in `basic_pitch/__init__.py:76`.
  - TFLite (`nmp.tflite`) in `basic_pitch/__init__.py:77`.
  - ONNX (`nmp.onnx`) in `basic_pitch/__init__.py:78`.

## 5) Harmonic CQT Details
- Harmonic stacking is performed after CQT+log normalization.
- Implementation uses one base CQT then channel-stacks frequency-shifted copies.
- Default harmonic multipliers: `[0.5, 1, 2, 3, 4, 5, 6, 7]`.
- Shift formula: `round(12 * bins_per_semitone * log2(harmonic))`.
- CQT config (default model):
  - `fmin=27.5 Hz`
  - `bins_per_octave=36` (3 bins/semitone)
  - `hop_length=256`
  - `sr=22050`

## Initial Gate Result
- Status: GREEN for preprocessing location risk.
- Reason: CQT + normalization + harmonic stacking are embedded in model graph for TF model path.
- Remaining risk: verify converted CoreML model keeps equivalent behavior and outputs.
