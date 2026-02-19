# MemoToMIDI

Initial spike workspace for validating Basic Pitch preprocessing and CoreML viability before building the iOS app.

## Scope

This repo starts with the CQT spike workflow from `MemoToMIDI-CQT-Spike-Plan.md`:

1. Inspect Basic Pitch source to confirm preprocessing and tensor contracts.
2. Capture reference tensors from a known test signal.
3. Attempt CoreML conversion and numeric output comparison.

## Quick Start

Use Python 3.10 or 3.11. Basic Pitch is not officially compatible with Python 3.14.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-spike.txt
```

Generate a Step 1 source report:

```powershell
python scripts/step1_inspect_basic_pitch.py
```

Run Step 2 reference capture (uses synthetic G major chord if no audio is provided):

```powershell
python scripts/step2_capture_reference.py
```

Use a custom WAV input:

```powershell
python scripts/step2_capture_reference.py --audio path\to\input.wav
```

Artifacts are written to `cqt_spike/`.
