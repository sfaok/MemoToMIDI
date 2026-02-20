# PROJECT_STATE

Generated: 2026-02-20
Scope: `MemoToMIDI.xcodeproj` target sources under `MemoToMIDI/`
Build check: `xcodebuild ... build CODE_SIGNING_ALLOWED=NO` on iOS Simulator completed with `BUILD SUCCEEDED`.

## 1. FILE INVENTORY

### Audio
| File | Lines | Description | Status |
|---|---:|---|---|
| `Audio/AudioFileReader.swift` | 92 | Reads WAV files, converts/resamples to mono Float32 buffers, and reports file metadata. | COMPLETE |
| `Audio/AudioRecorder.swift` | 289 | Handles mic permission, recording to 22050 Hz mono WAV via `AVAudioEngine` + `AVAudioConverter`, playback, and live waveform levels. | COMPLETE |

### MIDI
| File | Lines | Description | Status |
|---|---:|---|---|
| `MIDI/MIDIExporter.swift` | 137 | Writes MIDI data to temp file and presents share sheet via `UIActivityViewController`. | COMPLETE |
| `MIDI/MIDIFileWriter.swift` | 181 | Encodes SMF Type 0 MIDI with one tempo meta event and note on/off events using VLQ delta times. | COMPLETE |
| `MIDI/MIDIPlayer.swift` | 401 | Provides note-preview playback using `AVAudioEngine` + `AVAudioUnitSampler` with transport controls and seeking. | COMPLETE |

### ML
| File | Lines | Description | Status |
|---|---:|---|---|
| `ML/BasicPitchInference.swift` | 270 | Runs windowed CoreML Basic Pitch inference, stitches overlap-trimmed outputs, and returns full note/onset activation matrices. | COMPLETE |

### App Entry
| File | Lines | Description | Status |
|---|---:|---|---|
| `MemoToMIDI/ContentView.swift` | 21 | Root SwiftUI container that routes to `RecordingView`. | COMPLETE |
| `MemoToMIDI/MemoToMIDIApp.swift` | 17 | App entry point defining the main window scene. | COMPLETE |

### Models
| File | Lines | Description | Status |
|---|---:|---|---|
| `Models/Constants.swift` | 27 | Defines validated audio/model window constants and default MIDI/extraction constants. | COMPLETE |
| `Models/NoteEvent.swift` | 17 | Core note model (`pitch`, `startTime`, `duration`, `velocity`) used across extraction, UI, and MIDI export. | COMPLETE |

### Processing
| File | Lines | Description | Status |
|---|---:|---|---|
| `Processing/NoteExtractor.swift` | 179 | Converts note/onset activations into `NoteEvent` objects with thresholds, min length, merging, and velocity mapping. | COMPLETE |
| `Processing/PitchCorrector.swift` | 64 | Applies fractional pitch-bin tuning correction to activation matrices. | COMPLETE |
| `Processing/TransientRefiner.swift` | 129 | Refines note start times by snapping note clusters to nearby transient peaks in raw audio. | COMPLETE |
| `Processing/TuningDetector.swift` | 104 | Estimates cents offset and confidence from activation peak interpolation. | COMPLETE |

### Views
| File | Lines | Description | Status |
|---|---:|---|---|
| `Views/EditorView.swift` | 225 | Note-edit screen with piano roll, preview playback, cleanup sliders, and MIDI export. | COMPLETE |
| `Views/PianoRollView.swift` | 506 | Canvas-based piano roll with zoom, scrolling, selection, deletion, and playback scrubber/seek. | COMPLETE |
| `Views/RecordingView.swift` | 444 | Main workflow UI for record/stop/play, auto-inference after stop, caching, extraction, and navigation to editor. | COMPLETE |
| `Views/WaveformView.swift` | 51 | Lightweight bar waveform visualization of recent recording levels. | COMPLETE |

### Views/Components
| File | Lines | Description | Status |
|---|---:|---|---|
| `Views/Components/CleanupSlidersView.swift` | 133 | Expandable sliders for extraction sensitivity, min note length, merge gap, and sustain threshold. | COMPLETE |

## 2. IMPLEMENTED FEATURES

| Feature | Status | Note |
|---|---|---|
| Audio recording at 22050 Hz mono | YES | `AudioRecorder` converts input to 22050 Hz mono Float32 before writing WAV. |
| Audio playback | YES | `AudioRecorder.playLastRecording()` uses `AVAudioPlayer`. |
| CoreML Basic Pitch inference (windowed + stitched) | YES | `BasicPitchInference.process()` windows, predicts per window, trims overlap halves, concatenates, and trims final frame count. |
| Tuning detection | YES | `TuningDetector.detect(from:)` implemented and used after inference. |
| Pitch correction | YES | `PitchCorrector.correct(result:centsOffset:)` applied before extraction. |
| Note extraction with adjustable thresholds | YES | `NoteExtractor` supports onset/frame thresholds, min length, merge gap; sliders drive re-extraction in `EditorView`. |
| Transient timing refinement | YES | `TransientRefiner.refine(...)` runs after extraction. |
| MIDI file export (SMF Type 0) | YES | `MIDIFileWriter.write(notes:bpm:)` encodes valid Type 0 data with tempo + note events. |
| Share sheet / file export UI | YES | `MIDIExportShareButton` writes temp `.mid` then presents `UIActivityViewController`. |
| Piano roll display | YES | `PianoRollView` uses SwiftUI `Canvas` for rendering notes/grid/scrubber. |
| MIDI preview playback (AVAudioUnitSampler) | YES | `MIDIPlayer` schedules note start/stop events on sampler. |
| Cleanup sliders (sensitivity, min note length, merge distance) | YES | `CleanupSlidersView` exposes onset threshold, min length, and merge gap controls. |
| Transient refinement toggle | NO | Refinement is always applied in `EditorView`; no UI toggle/state for OFF/ON. |
| Flash tempo / visual metronome | NO | No `FlashTempoView` or equivalent UI in target. |
| Tap tempo | NO | No tap-tempo control implemented. |
| Auto BPM detection | NO | No `TempoDetector` in target sources. |
| Soft quantization | NO | No `Quantizer` in target sources. |
| Recording library / persistence | NO | No `RecordingStore` / recording list model/view in target sources. |
| Waveform display | YES | `WaveformView` renders live downsampled waveform bars. |

## 3. DATA FLOW

Actual path from “user taps record” to “MIDI file exported”:

1. `MemoToMIDI/ContentView.swift` -> `RecordingView()` is shown.
2. User taps **Record** in `RecordingView` -> `handleRecordButton()` (`Views/RecordingView.swift`) -> `audioRecorder.startRecording()` (`Audio/AudioRecorder.swift`).
3. `AudioRecorder.startRecording()` configures session, installs input tap, converts incoming buffers in `handleInputBuffer(_:)`, and writes 22050 Hz mono WAV.
4. User taps **Stop** -> `handleRecordButton()` calls `audioRecorder.stopRecording()`.
5. `RecordingView.onChange(of: audioRecorder.isRecording)` detects `true -> false`, calls `refreshLastFileDescription()`, increments `processingRequestID`.
6. `.task(id: processingRequestID)` triggers `runInferenceForLastRecording()` (`Views/RecordingView.swift`).
7. `runInferenceForLastRecording()` reads samples with `AudioFileReader.readSamples(from:)` -> `readMonoFloat32(...)` (`Audio/AudioFileReader.swift`).
8. `runInferenceForLastRecording()` calls `inferenceEngine.process(audioSamples:progressCallback:)` (`ML/BasicPitchInference.swift`).
9. `BasicPitchInference.processSync(...)` runs: `warmUpSync()` -> `loadModelIfNeeded()` -> per-window `predictWindow(...)` -> overlap trim + concatenate -> final frame trim -> returns `InferenceResult`.
10. `RecordingView` caches outputs in state: `cachedInferenceResult`, `cachedAudioBuffer`, `lastProcessedRecordingURL`.
11. Processing chain then runs from cached inference output: `TuningDetector.detect(from:)` -> `PitchCorrector.correct(...)` -> `NoteExtractor.extract(...)` -> `TransientRefiner.refine(...)` -> `extractedNotes`.
12. User opens editor via `NavigationLink` to `EditorView(inferenceResult:tuningResult:audioBuffer:)`.
13. `EditorView.init(...)` computes and stores `correctedResult` once from passed cached inference output.
14. `EditorView.initializeNotesIfNeeded()` and `reextractNotes()` run `NoteExtractor.extract(from:correctedResult, parameters:...)` + `TransientRefiner.refine(...)`.
15. User taps **Export MIDI** (`MIDIExportShareButton`) -> `exportAndShare()` (`MIDI/MIDIExporter.swift`).
16. `exportAndShare()` -> `MIDIExporter.exportToFile(notes:bpm:)` -> `MIDIFileWriter.write(notes:bpm:)` (`MIDI/MIDIFileWriter.swift`) -> `.mid` temp file written -> `MIDIShareSheet` presented.

Cache behavior vs recomputation:

- Inference caching: `RecordingView` skips re-inference when `lastProcessedRecordingURL == url && cachedInferenceResult != nil` (`Views/RecordingView.swift:299`).
- Slider edits: `EditorView` re-runs extraction/refinement from cached `correctedResult` + `audioBuffer`; it does not call `BasicPitchInference` (`Views/EditorView.swift:211`, `Views/EditorView.swift:214`).
- Export: uses current `notes` only; no re-extraction and no re-inference during export (`MIDI/MIDIExporter.swift:21`).

## 4. KNOWN ISSUES

- No explicit TODO/FIXME/XXX/HACK markers were found in Swift source.
- Deployment target is iOS 17.2, not 17.0 as required by project constraints (`MemoToMIDI/MemoToMIDI.xcodeproj/project.pbxproj:362`, `MemoToMIDI/MemoToMIDI.xcodeproj/project.pbxproj:419`).
- Transient refinement is always on; there is no OFF/ON toggle (`MemoToMIDI/Views/EditorView.swift:215`, `MemoToMIDI/Views/EditorView.swift:223`).
- Tempo is fixed to default 120 BPM in export buttons; no tap tempo/auto-BPM/tempo control integration (`MemoToMIDI/Views/RecordingView.swift:114`, `MemoToMIDI/Views/EditorView.swift:114`).
- Analysis and storage feature areas are not implemented in target (groups exist but contain no files), so flash tempo, BPM detection, quantization, and persistence/library are absent (`MemoToMIDI/MemoToMIDI.xcodeproj/project.pbxproj:165`, `MemoToMIDI/MemoToMIDI.xcodeproj/project.pbxproj:172`).
- `PHASE-STATUS.md` is stale and contradicts current code (lists Phase 3/4/5 as not started even though related files compile and run) (`MemoToMIDI/PHASE-STATUS.md:72`, `MemoToMIDI/PHASE-STATUS.md:94`, `MemoToMIDI/PHASE-STATUS.md:106`).
- Verbose debug logging remains in runtime paths, which can flood console during normal usage (`MemoToMIDI/Views/RecordingView.swift:338`, `MemoToMIDI/Views/RecordingView.swift:391`, `MemoToMIDI/Processing/TransientRefiner.swift:57`).

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
