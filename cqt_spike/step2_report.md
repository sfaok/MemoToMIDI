# Step 2 Report: Reference Capture + CoreML Conversion/Runtime Comparison

## Run Date
- February 19, 2026

## Environment Used
- OS: macOS (Darwin 22, x86_64)
- Python: 3.9.6
- Virtual env: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/.venv`
- Dependencies: `pip install -r requirements-spike.txt`

## Commands Executed
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-spike.txt
git clone https://github.com/spotify/basic-pitch .tmp/basic-pitch
python scripts/step2_capture_reference.py
```

## Step 2 Result
- Status: COMPLETE
- Reference capture: SUCCESS
- CoreML conversion (TF SavedModel): SUCCESS
- CoreML runtime prediction compare (macOS): SUCCESS

## Artifacts Generated
- `cqt_spike/report.md`
- `cqt_spike/conversion_report.json`
- `cqt_spike/metadata.json`
- `cqt_spike/reference_audio.npy`
- `cqt_spike/reference_input.npy`
- `cqt_spike/reference_note_output.npy`
- `cqt_spike/reference_onset_output.npy`
- `cqt_spike/reference_contour_output.npy`
- `cqt_spike/reference_note_output_batched.npy`
- `cqt_spike/reference_onset_output_batched.npy`
- `cqt_spike/reference_contour_output_batched.npy`
- `cqt_spike/reference_output.mid`
- `cqt_spike/reference_note_events.json`
- `cqt_spike/synthetic_g_major.wav`
- `cqt_spike/BasicPitch_from_tf.mlpackage`

## CoreML Findings on macOS
- TF SavedModel -> CoreML conversion succeeded.
- Runtime compare completed via `mlmodel.predict(...)`.
- Max absolute diff vs TensorFlow batched window outputs:
  - `note`: `0.12656021118164062`
  - `onset`: `0.13552778959274292`
  - `contour`: `0.2006065547466278`
- TFLite conversion still failed with source-framework detection error in `coremltools`.

## Gate
- Step 2 Gate: YELLOW
- Reason: pipeline now runs end-to-end on macOS and runtime comparison is unblocked, but output drift is non-trivial and needs acceptance thresholds / follow-up validation.
