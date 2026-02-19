# Step 2 Report: Reference Capture + CoreML Conversion Attempt

## Run Date
- February 19, 2026

## Environment Used
- OS: Windows
- Python: 3.11.9
- Virtual env: `c:\Users\email\MemoToMIDI\.venv`
- Dependencies: `pip install -r requirements-spike.txt`

## Commands Executed
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-spike.txt
.\.venv\Scripts\python.exe scripts\step2_capture_reference.py
```

## Step 2 Result
- Status: PARTIAL
- Reference capture: SUCCESS
- CoreML conversion + numeric runtime comparison: BLOCKED ON WINDOWS

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

## CoreML Findings on Windows
- `scripts/step2_capture_reference.py` conversion attempts:
  - TF SavedModel -> failed with `RuntimeError('BlobWriter not loaded')`
  - TFLite -> failed with source framework detection error
- Additional probe:
  - TF SavedModel converted successfully to `.mlmodel` when forcing `convert_to='neuralnetwork'`
  - Output file: `cqt_spike/BasicPitch_from_tf.mlmodel`
- Runtime comparison is still blocked on Windows:
  - `coremltools.models.MLModel.predict(...)` raises macOS-only runtime error

## Handoff for macOS
Run on Mac to complete conversion validation and numeric diff:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-spike.txt
python scripts/step2_capture_reference.py --audio cqt_spike/synthetic_g_major.wav
```

Then inspect:
- `cqt_spike/conversion_report.json`
- `cqt_spike/report.md`

## Gate
- Step 2 Gate: YELLOW
- Reason: reference tensors are captured and reproducible; CoreML runtime equivalence check must be finished on macOS.
