# MemoToMIDI — Agent Instructions

> Read this file completely before writing any code. It is the single source of truth for how this project works.

## What This App Does

MemoToMIDI records a guitar performance on iPhone, runs ML inference to detect notes, and exports a MIDI file. The user's workflow is: record → stop → adjust → export to Logic Pro.

## Hard Constraints — Repeat in Every Session

- **Pure Swift, zero third-party dependencies.** Allowed frameworks: SwiftUI, AVFoundation, AVAudioEngine, CoreML, Accelerate/vDSP, Foundation. No CocoaPods. No SPM packages. No AudioKit. No third-party MIDI libraries.
- **SwiftUI for all UI.** No UIKit except `UIActivityViewController` for the share sheet.
- **Offline processing, not real-time.** ML inference runs AFTER recording stops.
- **iOS 17.0 deployment target.** Target device: iPhone 13 (A15 Bionic, 4GB RAM).
- **The CoreML model is pre-converted and bundled.** Never attempt model conversion in Swift. The model file is `BasicPitch_from_tf.mlpackage`.
- **No pitch bend in v1.** The `NoteEvent` struct has no `pitchBend` field. MIDI writer emits no pitch bend events. This is intentional — deferred to post-v1.
- **No tempo curve in v1.** Single tempo meta-event per MIDI file. No multiple tempo changes.

## The Pipeline — Understand This Before Touching Any Code

```
Record Audio (22050 Hz mono)
    → Window into overlapping chunks (43844 samples, hop 36164)
    → CoreML inference per window (raw audio in, probability matrices out)
    → Stitch windows (trim half-overlap per side, concatenate)
    → [CACHED] note matrix (frames × 88) + onset matrix (frames × 88)
    → Tuning detection + correction (operates on cached matrices)
    → Note extraction (threshold + connected components)
    → [NoteEvent] array
    → MIDI export (NoteEvent + BPM → .mid file)
```

**Critical rule:** Tuning changes, cleanup slider changes, and export mode changes all re-run from cached ML output. They NEVER re-run inference. If your code re-runs inference when a slider moves, it is wrong.

## Validated Model Constants

These are proven values from the CQT spike. Do not change them.

```swift
enum AudioConstants {
    static let sampleRate: Double = 22050
    static let windowSamples: Int = 43844
    static let hopSamples: Int = 36164       // windowSamples - overlapLen
    static let overlapLen: Int = 7680        // 30 frames × 256 hop
    static let prependSamples: Int = 3840    // overlapLen / 2
}

enum MIDIConstants {
    static let ppqn: UInt16 = 480
    static let defaultTempo: Double = 120.0
    static let defaultOnsetThreshold: Float = 0.45
    static let defaultFrameThreshold: Float = 0.3
    static let minNoteLengthMs: Double = 58.0
    static let guitarRangeLow: UInt8 = 40    // E2
    static let guitarRangeHigh: UInt8 = 84   // C6
}
```

The CoreML model input is `[1, 43844, 1]` (batch, samples, channels) of Float32 audio at 22050 Hz.
The model outputs per window: `"note" [1, 172, 88]`, `"onset" [1, 172, 88]`, `"contour" [1, 172, 264]` (contour is deferred).

## Core Data Model

Every file that processes or consumes notes depends on this. Paste it if the agent doesn't have it in context.

```swift
import Foundation

struct NoteEvent: Identifiable, Codable {
    let id: UUID
    var pitch: UInt8          // MIDI note number (0-127)
    var startTime: Double     // Seconds from recording start
    var duration: Double      // Seconds
    var velocity: UInt8       // 0-127

    init(id: UUID = UUID(), pitch: UInt8, startTime: Double, duration: Double, velocity: UInt8) {
        self.id = id
        self.pitch = pitch
        self.startTime = startTime
        self.duration = duration
        self.velocity = velocity
    }
}
```

## Project Structure

```
MemoToMIDI/
├── MemoToMIDIApp.swift
├── Models/
│   ├── NoteEvent.swift
│   ├── Recording.swift
│   └── Constants.swift
├── Audio/
│   ├── AudioRecorder.swift          # AVAudioEngine → .wav at 22050 Hz mono
│   └── AudioFileReader.swift        # .wav → [Float]
├── ML/
│   ├── BasicPitch_from_tf.mlpackage
│   └── BasicPitchInference.swift    # Window → CoreML → stitch
├── Processing/
│   ├── TuningDetector.swift
│   ├── PitchCorrector.swift
│   └── NoteExtractor.swift
├── MIDI/
│   ├── MIDIFileWriter.swift         # SMF Type 0, ~100 lines
│   └── MIDIPlayer.swift             # AVAudioEngine + AVAudioUnitSampler
├── Analysis/
│   ├── TempoDetector.swift
│   └── Quantizer.swift
├── Storage/
│   └── RecordingStore.swift
├── Views/
│   ├── RecordingListView.swift
│   ├── RecordingView.swift
│   ├── EditorView.swift
│   ├── PianoRollView.swift
│   ├── WaveformView.swift
│   └── Components/
│       ├── FlashTempoView.swift
│       ├── TuningIndicatorView.swift
│       ├── CleanupSlidersView.swift
│       ├── TempoControlView.swift
│       └── PreviewControlsView.swift
└── Resources/
    ├── Sounds/
    └── Assets.xcassets
```

Place files exactly where this structure says. Don't create new directories or reorganize.

## Phase Dependencies

```
Phase 0 ✓ (DONE — CoreML model converted and validated)
Phase 1: Audio capture + playback (no dependencies)
Phase 2: ML inference (needs Phase 1 audio files)
Phase 3: Note extraction + tuning (needs Phase 2 cached output)
Phase 4: MIDI export (needs Phase 3 NoteEvent array)
Phase 5: Piano roll + preview (needs Phase 3, parallel with Phase 4)
Phase 6: Flash tempo + quantization (needs Phase 4 + 5)
Phase 7: Library + polish (needs all above)
```

Check PHASE-STATUS.md for current progress before starting work.

## Known Pitfalls

### Audio Recording (Phase 1)
- iPhone 13 mic runs at 48000 Hz natively. AVAudioEngine's input node format is hardware-determined. You'll need `AVAudioConverter` to downsample to 22050 Hz, or set the tap format — test both, the tap format approach may silently fail.
- `AVAudioSession` category must be `.playAndRecord` with `.defaultToSpeaker`. Get this wrong and playback comes out of the earpiece.

### CoreML Inference (Phase 2)
- `MLMultiArray` allocation is slow if you copy data. Use `MLMultiArray(dataPointer:...)` for zero-copy.
- First inference after app launch is slow (model compilation). Warm the model at launch with a dummy input.
- The unwrap logic is: trim `overlapLen / 2` frames from each side of each window's output, then concatenate. No averaging of overlapping frames.

### Note Extraction (Phase 3)
- Default onset threshold is **0.45**, not 0.5. This compensates for CoreML's slight downward activation shift vs TensorFlow.
- Frame duration is ~11.6ms (256 hop / 22050 Hz). Minimum note length of 58ms ≈ 5 frames.
- Connected component extraction: iterate per-pitch across the time axis. A note starts when onset > threshold AND frame > frame_threshold. It continues while frame > frame_threshold. It ends when frame drops below threshold.

### MIDI Writer (Phase 4)
- Variable-length quantity encoding: values > 127 need multi-byte encoding. This is the only tricky part — the rest is straightforward binary packing.
- Delta times are in ticks: `ticks = seconds × (BPM / 60) × PPQN`. Watch for floating-point rounding — always round to nearest integer tick.
- Events must be sorted by absolute time, then encoded as delta times between consecutive events.
- SMF is big-endian. Swift's default integer byte order is little-endian on ARM. Use `.bigEndian` on all multi-byte integer writes.

### Piano Roll (Phase 5)
- SwiftUI Canvas is the right choice for rendering notes — don't try to make each note a separate SwiftUI view. Hundreds of views will kill performance.
- Pinch-to-zoom needs `MagnificationGesture` composed with `DragGesture`. Getting both axes zoomable independently is non-trivial.

## Code Style

- Prefer `async/await` over callbacks or Combine.
- Errors should be Swift `Error` enums, not string messages.
- No force unwraps (`!`) except on hardcoded resource loads that are known to exist (e.g., bundled .sf2 file).
- All audio processing functions should work on `[Float]` buffers (Float32). No Double audio buffers.
- Keep files focused. One type per file for models and major classes. Utilities can share a file if small.

## What Not to Build

- No pitch bend detection or export (post-v1)
- No tempo curve / multiple tempo events (post-v1)
- No chord name display (post-v1)
- No per-string tuning correction (post-v1)
- No Core Data or SwiftData — use file-based JSON persistence
- No network features, no cloud sync, no accounts
- No onboarding flow or tutorials
