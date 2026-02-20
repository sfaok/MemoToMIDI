# MemoToMIDI — Phase Status

> **Update this file after completing each phase.** Agents should read this before starting work to know what exists and what interfaces they can depend on.

Last updated: 2026-02-20

---

## Phase 0: Model Conversion ✅ COMPLETE

**Deliverables:**
- `BasicPitch_from_tf.mlpackage` — CoreML model, raw audio in → probability matrices out
- `cqt_spike/` directory with reference data (.npy files) and spike documentation
- `BasicPitchRunner.swift` — macOS CLI proof of concept (to be refactored for iOS)

**Key findings:**
- CQT preprocessing is INSIDE the model graph. No external DSP needed.
- Model input: `[1, 43844, 1]` Float32 audio at 22050 Hz
- Model output per window: `note [1, 172, 88]`, `onset [1, 172, 88]`, `contour [1, 172, 264]`
- Validated: zero diff against Python reference
- Onset threshold adjusted to 0.45 (not 0.5) to compensate for CoreML activation shift

---

## Phase 1: Audio Capture + Playback ✅ COMPLETE

**Completed:**
- `Audio/AudioRecorder.swift`
  - Implemented APIs: `requestMicrophoneAccess() async -> Bool`, `startRecording() throws`, `stopRecording()`, `playLastRecording() throws`, `stopPlayback()`
  - Records to 22050 Hz, mono, Float32 `.wav` in Documents (with converter path for hardware sample-rate mismatch)
- `Audio/AudioFileReader.swift`
  - Implemented APIs: `describeFile(at:)` and `readMonoFloat32(from:targetSampleRate:)`
- `Views/RecordingView.swift` — Record/Stop/Play UI
- `Views/WaveformView.swift` — Basic waveform during recording
- `Models/Constants.swift` — shared audio constants for Phase 1
- `BasicPitch_from_tf.mlpackage` is included in the Xcode target
- On-device checks passed (iPhone 13): recording/playback confirmed and recorded output validated at `22050 Hz`, mono, Float32

**Depends on:** Nothing

---

## Phase 2: ML Inference Pipeline ✅ COMPLETE

**Completed:**
- `ML/BasicPitchInference.swift`
  - `process(audioSamples:progressCallback:) async throws -> InferenceResult`
  - `warmUp() async throws` for one-time model warm-up
  - Windowing: prepend 3840 zeros, 43844 sample windows, 36164 sample hop, final zero-padding
  - CoreML inference per window using generated model API (`input_2`, `Identity_1`, `Identity_2`)
  - Overlap stitch: trim half-overlap per side and concatenate (no overlap averaging)
  - Final trim to real-audio frame count based on model hop size (256 samples/frame)
- `Audio/AudioFileReader.swift`
  - Added `readSamples(from:)` for Phase 2 call sites
- `Views/RecordingView.swift`
  - Auto-runs inference after recording stops
  - Progress indicator during processing
  - Temporary console diagnostics (shapes, frame duration, top activations)
  - Caches inference matrices in view state for downstream reuse

**Depends on:** Phase 1 (`AudioFileReader.readSamples()` / `readMonoFloat32()` to get `[Float]` buffer)

---

## Phase 3: Note Extraction + Tuning ⬜ NOT STARTED

**Will produce:**
- `Processing/NoteExtractor.swift`
  - Public API: `func extract(noteMatrix: [[Float]], onsetMatrix: [[Float]], params: ExtractionParams) -> [NoteEvent]`
  - Parameters: onset threshold, frame threshold, min duration, merge distance
- `Processing/TuningDetector.swift`
  - Public API: `func detectTuning(noteMatrix: [[Float]]) -> Double` (cents offset)
- `Processing/PitchCorrector.swift`
  - Public API: `func correct(noteMatrix: [[Float]], centsOffset: Double) -> [[Float]]`

**Depends on:** Phase 2 (cached note and onset matrices)

---

## Phase 4: MIDI Export ⬜ NOT STARTED

**Will produce:**
- `MIDI/MIDIFileWriter.swift`
  - Public API: `func writeMIDI(notes: [NoteEvent], bpm: Double, to url: URL) throws`
  - SMF Type 0, 480 PPQN, single tempo event, note-on/off only
- Share sheet integration via `UIActivityViewController`

**Depends on:** Phase 3 (`[NoteEvent]` array)

---

## Phase 5: Piano Roll + Preview ⬜ NOT STARTED

**Will produce:**
- `Views/PianoRollView.swift` — Canvas-based note renderer
- `Views/EditorView.swift` — Piano roll + sliders + export button
- `MIDI/MIDIPlayer.swift` — AVAudioEngine + AVAudioUnitSampler playback
- `Views/Components/CleanupSlidersView.swift` — Threshold/merge sliders

**Depends on:** Phase 3 (can run parallel with Phase 4)

---

## Phase 6: Tempo + Quantization ⬜ NOT STARTED

**Will produce:**
- `Views/Components/FlashTempoView.swift` — Visual metronome
- `Analysis/TempoDetector.swift` — Auto-BPM from onset intervals
- `Analysis/Quantizer.swift` — Soft quantization engine
- `Views/Components/TempoControlView.swift` — BPM input + tap tempo

**Depends on:** Phase 4 + Phase 5

---

## Phase 7: Library + Polish ⬜ NOT STARTED

**Will produce:**
- `Views/RecordingListView.swift` — Library screen
- `Storage/RecordingStore.swift` — File + JSON persistence
- `Models/Recording.swift` — Recording metadata model
- App icon, launch screen, dark theme, error handling

**Depends on:** All prior phases

---

## Interface Contracts

> As phases complete, document the actual function signatures here so downstream phases reference the real API, not the planned one.

### AudioFileReader (Phase 1)
```
Planned: func readAudio(from url: URL) throws -> [Float]
Actual:  static func describeFile(at url: URL) throws -> AudioFileDescription
         static func readMonoFloat32(from url: URL, targetSampleRate: Double = AudioConstants.sampleRate) throws -> [Float]
```

### AudioRecorder (Phase 1)
```
Planned: startRecording(), stopRecording() -> URL
Actual:  func requestMicrophoneAccess() async -> Bool
         func startRecording() throws
         func stopRecording()
         func playLastRecording() throws
         func stopPlayback()
```

### BasicPitchInference (Phase 2)
```
Planned: func process(audioBuffer: [Float]) async throws -> (note: [[Float]], onset: [[Float]])
Actual:  struct InferenceResult {
             let noteActivations: [[Float]]
             let onsetActivations: [[Float]]
             let frameDuration: Double
         }
         final class BasicPitchInference {
             func warmUp() async throws
             func process(audioSamples: [Float], progressCallback: ((Double) -> Void)? = nil) async throws -> InferenceResult
         }
```

### NoteExtractor (Phase 3)
```
Planned: func extract(noteMatrix: [[Float]], onsetMatrix: [[Float]], params: ExtractionParams) -> [NoteEvent]
Actual:  [pending]
```

### MIDIFileWriter (Phase 4)
```
Planned: func writeMIDI(notes: [NoteEvent], bpm: Double, to url: URL) throws
Actual:  [pending]
```
