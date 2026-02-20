# MemoToMIDI

MemoToMIDI is an iOS app-in-progress that turns recorded audio into MIDI.  
This repo currently includes:

- A validated Basic Pitch/CoreML spike workflow (`scripts/` + `cqt_spike/` artifacts)
- A working iOS Phase 1 app shell (`record + playback + waveform`) in SwiftUI

## Current Status

- Spike validation: complete through CoreML comparison + diagnostics
- iOS app: Phase 1 implemented (`22,050 Hz mono capture`, file save, playback)
- Build plan: `cqt_spike/MemoToMIDI/UPDATED_BUILD_PLAN.md`

## Repository Layout

- `scripts/` Python spike scripts (Step 1 to Step 4)
- `cqt_spike/` generated artifacts, reports, model assets, and Swift spike code
- `cqt_spike/MemoToMIDI/` Xcode iOS project (`MemoToMIDI.xcodeproj`)
- `requirements-spike.txt` Python dependencies for spike scripts

## Prerequisites

- macOS (for Xcode/iOS app)
- Xcode 15+ (project deployment target: iOS 17.2)
- Python 3.10 or 3.11 for spike scripts

## Quick Start (Spike Workflow)

From repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-spike.txt
```

Run scripts:

```bash
python scripts/step1_inspect_basic_pitch.py
python scripts/step2_capture_reference.py
python scripts/step3_full_pipeline_comparison.py
python scripts/step3_diagnostics.py
python scripts/step3_real_audio_comparison.py --audio cqt_spike/step3_real_audio_input.wav
python scripts/step4_precompute_assets.py
```

Most outputs are written to `cqt_spike/`.

## Run the iOS App

1. Open `cqt_spike/MemoToMIDI/MemoToMIDI.xcodeproj` in Xcode.
2. Select an iPhone target (device preferred for microphone testing).
3. Build and run.
4. Grant microphone permission.
5. Use the Record/Stop and Play Last controls on the `Phase 1: Record + Playback` screen.

The app records WAV files to the app Documents directory at `22,050 Hz` mono float32.

## Notes

- `cqt_spike/` includes research artifacts and intermediate files from model validation.
- Root `.gitignore` excludes many generated spike artifact types (`.npy`, `.json`, `.wav`, etc.).
