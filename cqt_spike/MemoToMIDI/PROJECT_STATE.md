# PROJECT_STATE

Generated: 2026-02-21
Scope: `MemoToMIDI.xcodeproj` target sources under `MemoToMIDI/`
Build check: `xcodebuild -project MemoToMIDI/MemoToMIDI.xcodeproj -scheme MemoToMIDI -configuration Debug -destination 'generic/platform=iOS Simulator' build CODE_SIGNING_ALLOWED=NO` completed with `BUILD SUCCEEDED`.

## 1. FILE INVENTORY

### Audio
| File | Lines | Description | Status |
|---|---:|---|---|
| `Audio/AudioFileReader.swift` | 92 | Reads WAV files, converts/resamples to mono Float32 buffers, and reports file metadata. | COMPLETE |
| `Audio/AudioRecorder.swift` | 408 | Handles mic permission, recording to 22050 Hz mono WAV via `AVAudioEngine` + `AVAudioConverter`, playback, and waveform/playback timing state. | COMPLETE |

### MIDI
| File | Lines | Description | Status |
|---|---:|---|---|
| `MIDI/MIDIExporter.swift` | 137 | Writes MIDI data to temp file and presents share sheet via `UIActivityViewController`. | COMPLETE |
| `MIDI/MIDIFileWriter.swift` | 181 | Encodes SMF Type 0 MIDI with one tempo meta event and note on/off events using VLQ delta times. | COMPLETE |
| `MIDI/MIDIPlayer.swift` | 1148 | Provides MIDI-only preview and overlay playback, preset/soundfont loading, offline MIDI render, and pre-mixed single-buffer transport. | COMPLETE |

### ML
| File | Lines | Description | Status |
|---|---:|---|---|
| `ML/BasicPitchInference.swift` | 278 | Runs windowed CoreML Basic Pitch inference, stitches overlap-trimmed outputs, and removes prepend-alignment frames before extraction. | COMPLETE |

### App Entry
| File | Lines | Description | Status |
|---|---:|---|---|
| `MemoToMIDI/ContentView.swift` | 21 | Root SwiftUI container that routes to `RecordingView`. | COMPLETE |
| `MemoToMIDI/MemoToMIDIApp.swift` | 17 | App entry point defining the main window scene. | COMPLETE |

### Models
| File | Lines | Description | Status |
|---|---:|---|---|
| `Models/BeatMap.swift` | 71 | Stores tapped beat positions and derives BPM from median inter-tap intervals. | COMPLETE |
| `Models/Constants.swift` | 27 | Defines validated audio/model window constants and default MIDI/extraction constants. | COMPLETE |
| `Models/NoteEvent.swift` | 17 | Core note model (`pitch`, `startTime`, `duration`, `velocity`) used across extraction, UI, and MIDI export. | COMPLETE |

### Processing
| File | Lines | Description | Status |
|---|---:|---|---|
| `Processing/NoteExtractor.swift` | 189 | Converts note/onset activations into `NoteEvent` objects with thresholds, min length, merging, and curved velocity mapping. | COMPLETE |
| `Processing/PitchCorrector.swift` | 64 | Applies fractional pitch-bin tuning correction to activation matrices. | COMPLETE |
| `Processing/TransientRefiner.swift` | 129 | Refines note start times by snapping note clusters to nearby transient peaks in raw audio. | COMPLETE |
| `Processing/TuningDetector.swift` | 104 | Estimates cents offset and confidence from activation peak interpolation. | COMPLETE |

### Views
| File | Lines | Description | Status |
|---|---:|---|---|
| `Views/EditorView.swift` | 494 | Note-edit screen with piano roll, cleanup sliders, MIDI/overlay playback controls, beat tapping, and MIDI export. | COMPLETE |
| `Views/PianoRollView.swift` | 506 | Canvas-based piano roll with zoom, scrolling, selection, deletion, and playback scrubber/seek. | COMPLETE |
| `Views/RecordingView.swift` | 445 | Main workflow UI for record/stop/play, auto-inference after stop, caching, extraction, and navigation to editor. | COMPLETE |
| `Views/WaveformView.swift` | 51 | Lightweight bar waveform visualization of recent recording levels. | COMPLETE |

### Views/Components
| File | Lines | Description | Status |
|---|---:|---|---|
| `Views/Components/BeatTapView.swift` | 236 | Tap-along UI that captures beat positions during playback and returns `BeatMap`. | COMPLETE |
| `Views/Components/CleanupSlidersView.swift` | 133 | Expandable sliders for extraction sensitivity, min note length, merge gap, and sustain threshold. | COMPLETE |

Inventory changes since previous snapshot:
- Added inventory entries for `Models/BeatMap.swift` and `Views/Components/BeatTapView.swift`.
- Updated significant line-count growth in `MIDI/MIDIPlayer.swift`, `Views/EditorView.swift`, `Audio/AudioRecorder.swift`, `ML/BasicPitchInference.swift`, and `Processing/NoteExtractor.swift`.
- Current Swift target total: 4,748 lines.

## 2. IMPLEMENTED FEATURES

| Feature | Status | Note |
|---|---|---|
| Audio recording at 22050 Hz mono | YES | `AudioRecorder` converts input to 22050 Hz mono Float32 before writing WAV. |
| Audio playback | YES | `AudioRecorder.playLastRecording()` provides transport with current-time tracking. |
| CoreML Basic Pitch inference (windowed + stitched) | YES | `BasicPitchInference.process()` windows, predicts per window, trims overlap halves, concatenates, then trims to expected frame count. |
| Tuning detection | YES | `TuningDetector.detect(from:)` implemented and used after inference. |
| Pitch correction | YES | `PitchCorrector.correct(result:centsOffset:)` applied before extraction. |
| Note extraction with adjustable thresholds | YES | `NoteExtractor` supports onset/frame thresholds, min length, merge gap; sliders drive re-extraction in `EditorView`. |
| Velocity curve mapping | YES | Power curve with activation range normalization (0.25-0.95), exponent 0.75, velocity floor 30. |
| Transient timing refinement | YES | `TransientRefiner.refine(...)` runs after extraction. |
| MIDI file export (SMF Type 0) | YES | `MIDIFileWriter.write(notes:bpm:)` encodes valid Type 0 data with tempo + note events. |
| Share sheet / file export UI | YES | `MIDIExportShareButton` writes temp `.mid` then presents `UIActivityViewController`. |
| Piano roll display | YES | `PianoRollView` uses SwiftUI `Canvas` for rendering notes/grid/scrubber. |
| MIDI preview playback (AVAudioUnitSampler) | YES | SoundFont-backed sampler playback with preset options (Piano, Clean Guitar, Marimba/Sine slot) and program fallback when SoundFont loading fails. |
| Overlay playback | YES | Pre-mixed single-buffer approach, recording + MIDI sampler. |
| Beat tap mapping | YES | Tap along during playback, median BPM derivation, beat positions stored in `BeatMap`. |
| Cleanup sliders (sensitivity, min note length, merge distance) | YES | `CleanupSlidersView` exposes onset threshold, min length, and merge gap controls. |
| Flash tempo / visual metronome | NO | No dedicated flash-tempo component in target. |
| Auto BPM detection | NO | No automatic tempo detector in target sources; BPM is manual or tap-derived. |
| Soft quantization | NO | No `Quantizer` in target sources. |
| Recording library / persistence | NO | No `RecordingStore` / recording list model/view in target sources. |
| Waveform display | YES | `WaveformView` renders live downsampled waveform bars. |

## 3. DATA FLOW

Actual path from “user taps record” to “MIDI file exported”:

1. `MemoToMIDI/ContentView.swift` -> `RecordingView()` is shown.
2. User records in `RecordingView` via `AudioRecorder.startRecording()` / `stopRecording()`.
3. `RecordingView` launches `BasicPitchInference.process(audioSamples:progressCallback:)` after stop.
4. `BasicPitchInference` windows audio, runs CoreML, stitches overlap-trimmed outputs, and removes prepend frames so frame 0 aligns to recording start.
5. `RecordingView` caches `InferenceResult` and raw audio buffer.
6. Processing chain runs from cached ML output only: `TuningDetector` -> `PitchCorrector` -> `NoteExtractor` -> `TransientRefiner`.
7. `EditorView` re-extracts from cached corrected matrices when cleanup sliders change (no re-inference).
8. Playback options:
   - MIDI-only preview through `MIDIPlayer` sampler.
   - Overlay playback via offline rendered MIDI + recording mix into one buffer.
9. Tempo mapping options in `EditorView`:
   - Manual BPM entry.
   - Beat tapping (`BeatTapView`) persisted into `BeatMap` and exported using median-derived BPM.
10. Export path: `MIDIExportShareButton` -> `MIDIExporter.exportToFile(notes:bpm:)` -> `MIDIFileWriter.write(notes:bpm:)` -> share sheet.

Cache behavior vs recomputation:

- Inference caching: `RecordingView` skips re-inference when the last processed recording URL matches and cached inference exists.
- Slider edits / tuning changes: extraction and refinement rerun from cached matrices only.
- Export: writes from current note list + chosen BPM; no re-inference.

## 4. KNOWN ISSUES

- Deployment target remains iOS 17.2 in project settings (`MemoToMIDI/MemoToMIDI.xcodeproj/project.pbxproj`), while product constraints call for iOS 17.0.
- Transient refinement is always applied in `EditorView`; no OFF/ON toggle is exposed.
- Verbose debug logging remains in hot runtime paths (`RecordingView`, `MIDIPlayer`, `TransientRefiner`) and can flood console output.
- `PHASE-STATUS.md` is stale relative to implemented Phase 3-5+ code.
- Prepend offset (~174ms) was being applied to note times - FIXED. Subtracted in inference pipeline.

## 5. DEPENDENCIES CHECK

Third-party packages/dependencies:

- SPM package references in `project.pbxproj`: none
- `Package.resolved`: none
- CocoaPods files (`Podfile`, `Podfile.lock`): none
- Carthage files (`Cartfile`): none
- Vendored third-party source trees: none detected in target sources

Non-Apple imports found:

- None.

Apple framework imports found in source:

- `SwiftUI`, `UIKit`, `AVFoundation`, `CoreML`, `Accelerate`, `Combine`, `Foundation`
