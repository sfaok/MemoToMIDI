# Step 3 Report: CoreML Numeric Validation Summary

## Run Date
- February 19, 2026

## Objective
- Complete the macOS-only CoreML runtime comparison that was blocked in Step 2.

## Inputs
- Audio: `cqt_spike/synthetic_g_major.wav`
- Reference tensors: generated from `scripts/step2_capture_reference.py`
- Model source: TensorFlow SavedModel (`basic_pitch` ICASSP 2022 model)

## Result
- CoreML conversion from TensorFlow SavedModel: SUCCESS
- CoreML runtime prediction on macOS: SUCCESS
- Comparison executed against first batched TensorFlow output window.

## Numeric Diff (Max Abs)
- `note`: `0.12656021118164062`
- `onset`: `0.13552778959274292`
- `contour`: `0.2006065547466278`

## Interpretation
- Functional gate (conversion + runtime) is now open on macOS.
- Numeric parity is not close to zero; drift needs explicit tolerance criteria before calling parity "green".

## Step 3 Gate
- YELLOW
- Reason: runtime validation completed, but numeric acceptance thresholds are not yet defined and current max deltas are significant.
