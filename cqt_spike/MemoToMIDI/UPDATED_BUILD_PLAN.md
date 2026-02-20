# MemoToMIDI — Updated Build Plan

## Date: February 19, 2026

## What Changed from the Original Spec

### CQT Spike Results (the big one)
The CQT spectrogram preprocessing is **inside the CoreML model graph**, not external. The model takes raw audio `[1, 43844, 1]` at 22050 Hz and handles all DSP internally. This eliminates the entire DSP/ directory from the original spec — no `SpectrogramProcessor.swift`, no Accelerate/vDSP FFT code, no precomputed CQT kernels.

The Swift inference pipeline is ~150-200 lines: window the audio, feed to CoreML, stitch output. Validated at zero diff against Python reference.

### Validated Model Constants
```
sample_rate      = 22050
window_samples   = 43844
hop_samples      = 36164    (window - overlap_len)
overlap_len      = 7680     (30 frames × 256 hop)
prepend_samples  = 3840     (overlap_len / 2)
output: "note"   = [1, 172, 88]  per window
output: "onset"  = [1, 172, 88]  per window
output: "contour"= [1, 172, 264] per window (deferred)
```

### CoreML Conversion Notes
- Converted from TF SavedModel via coremltools — SUCCESS
- Musical validation: 88 of 95 notes matched TF at adjusted threshold
- Remaining "mismatches" are onset/duration boundary disagreements, not wrong notes
- Default onset threshold: **0.45** (not 0.5) compensates for CoreML's slight downward activation shift
- Unwrap logic: trim half-overlap per side, concatenate — no averaging

### Deferred to Post-v1
- **Pitch bend detection and MIDI export** — simplifies MIDI writer, avoids polyphonic channel rotation
- **Tempo Curve mode (Mode 3)** — beat tracking from onsets is unreliable for casual/rubato playing
- **Per-string tuning correction** — flag only, don't auto-correct
- **Chord display (F8)** — nice-to-have, build if time allows

### Simplified from Original Spec
- **No DSP/ directory.** Model handles CQT internally.
- **No Accelerate/vDSP dependency for preprocessing.** Still available if needed for other purposes (e.g., onset energy analysis for tempo detection), but not on the critical path.
- **MIDI writer is simpler** — no pitch bend events, no tempo curve (multiple tempo events). ~100 lines.
- **Export modes reduced to two:** Fixed BPM, Fixed BPM + Soft Quantization.

---

## Updated Project Structure

```
MemoToMIDI/
├── MemoToMIDIApp.swift
├── Models/
│   ├── NoteEvent.swift              # pitch, start, duration, velocity
│   └── Recording.swift              # Recording metadata
├── Audio/
│   ├── AudioRecorder.swift          # AVAudioEngine capture → .wav at 22050 Hz mono
│   └── AudioFileReader.swift        # Read .wav into float32 buffer
├── ML/
│   ├── BasicPitch_from_tf.mlpackage # Bundled CoreML model (raw audio in, probabilities out)
│   └── BasicPitchInference.swift    # Window → CoreML → stitch (from BasicPitchRunner.swift)
├── Processing/
│   ├── TuningDetector.swift         # Measure median cents offset from note activations
│   ├── PitchCorrector.swift         # Shift probability matrix by tuning offset
│   └── NoteExtractor.swift          # Threshold + connected components + cleanup
├── MIDI/
│   ├── MIDIFileWriter.swift         # SMF Type 0 encoder, no pitch bend, no tempo curve
│   └── MIDIPlayer.swift             # AVAudioEngine playback for preview
├── Analysis/
│   ├── TempoDetector.swift          # Auto-BPM from onset intervals
│   └── Quantizer.swift              # Soft quantization engine
├── Storage/
│   └── RecordingStore.swift         # File system + JSON metadata
├── Views/
│   ├── RecordingListView.swift
│   ├── RecordingView.swift          # Record screen + waveform + flash tempo
│   ├── EditorView.swift             # Piano roll + sliders + export
│   ├── PianoRollView.swift
│   ├── WaveformView.swift
│   └── Components/
│       ├── FlashTempoView.swift
│       ├── TuningIndicatorView.swift
│       ├── CleanupSlidersView.swift
│       ├── TempoControlView.swift
│       └── PreviewControlsView.swift
└── Resources/
    ├── Sounds/                      # Lightweight .sf2 for preview
    └── Assets.xcassets
```

---

## Updated Build Phases

### Phase 0: Model Conversion — DONE ✓
- CoreML model converted and validated
- Reference data captured
- Swift CLI proof of concept validated at zero diff
- Deliverable: `BasicPitch_from_tf.mlpackage` + spike documentation

### Phase 1: Xcode Project + Audio Capture + Playback
**Goal:** iPhone app that records audio at 22050 Hz mono, saves .wav, plays back.

**Build:**
- Create Xcode project (iOS 17.0, SwiftUI, iPhone 13 target)
- Bundle `BasicPitch_from_tf.mlpackage` in the project
- `AudioRecorder.swift`: AVAudioEngine with input node tap
  - Output format: 22050 Hz, mono, float32
  - If hardware won't do 22050 natively, record at device rate and use `AVAudioConverter` to downsample
  - Write to .wav in app Documents directory
- `AudioFileReader.swift`: Read .wav back into float32 buffer
- Microphone permission request (Info.plist + runtime)
- Simple SwiftUI screen: Record/Stop button, playback button
- Basic waveform display during recording (even rough)
- `AVAudioSession` category: `.playAndRecord` with `.defaultToSpeaker` option

**Test:**
- Record 10 seconds of guitar on iPhone 13
- Play it back — confirm clean audio, no clipping, correct sample rate
- Verify .wav file in Documents directory is 22050 Hz mono
- If using AVAudioConverter for resampling: verify output sample rate is exactly 22050

**Known pitfall:** iPhone 13 mic natively runs at 48000 Hz. AVAudioEngine's input node format is hardware-determined. You'll likely need `AVAudioConverter` or set the output format on the input node tap. Test both approaches — setting the tap format to 22050 may work and would be simplest.

### Phase 2: ML Inference Pipeline (iOS)
**Goal:** Port BasicPitchRunner.swift from macOS CLI to iOS, integrated with Phase 1 audio.

**Build:**
- `BasicPitchInference.swift`: refactored from BasicPitchRunner.swift
  - Public API: `func process(audioBuffer: [Float]) async throws -> (note: [[Float]], onset: [[Float]])`
  - Windowing: prepend 3840 zeros, window at 43844 with hop 36164, zero-pad final window
  - CoreML inference per window using `MLModel` with `MLMultiArray` input [1, 43844, 1]
  - Unwrap: trim half-overlap per side, concatenate
  - Return aggregated note and onset matrices
- Wire to Phase 1: after recording stops, auto-run inference
- Show progress indicator during processing
- Cache the raw model output (note + onset matrices) for downstream re-use

**Test:**
- Record an open G chord on iPhone, run inference
- Print top-5 activated pitches per time frame — verify G3(55), B3(59), D4(62) appear
- Processing time should be <2 seconds for a 15-second recording on iPhone 13
- Compare output shape against expected: (total_frames, 88) for notes

**Known pitfall:** `MLMultiArray` allocation and data copying can be slow if done naively. Use `MLMultiArray(dataPointer:...)` to avoid unnecessary copies. Also: CoreML first inference is slow (model compilation) — consider warming the model at app launch.

### Phase 3: Note Extraction + Cleanup
**Goal:** Turn raw probability matrices into `[NoteEvent]` arrays with user-adjustable parameters.

**Split into two sub-prompts:**

**Phase 3a: Note Extraction**
- Threshold note probabilities (default 0.45 for onset, 0.3 for frame activation)
- Use onset matrix to determine note start times
- Connected component extraction: group adjacent active frames per pitch into notes
- Minimum duration filter (default 58ms — the model's frame duration at 256 hop / 22050 Hz is ~11.6ms, so ~5 frames minimum)
- Velocity: derive from peak activation strength within each note, map to 0-127
- Note merge: if same pitch has gap < threshold (default ~30ms), merge into one note
- All parameters adjustable; re-extraction from cached model output (no re-inference)
- Return `[NoteEvent]`

**Phase 3b: Tuning Detection + Correction**
- Analyze raw pitch activations to measure median cents offset from MIDI note centers
- Report as cents (+/-) and equivalent reference pitch (e.g., A=443 Hz)
- Apply correction by shifting the activation matrix fractionally before note extraction
- Manual adjustment slider (±50 cents)
- Reference note mode: user plays one known pitch, app measures offset
- When tuning changes, re-run note extraction from cached output

**Test:**
- Record 4-5 distinct chords. Verify correct notes extracted for each.
- Adjust sensitivity slider: higher → more notes, lower → fewer. Confirm sensible behavior.
- Detune guitar ~20 cents. Record. Verify auto-detection shows ~20 cents. Verify correction fixes note names.

### Phase 4: MIDI File Writer + Export
**Goal:** `[NoteEvent]` + BPM → valid .mid file → share sheet → Logic.

**Build:**
- `MIDIFileWriter.swift`: SMF Type 0 binary encoder, pure Swift
  - Header chunk: format 0, 1 track, 480 PPQN
  - Single tempo meta-event (microseconds per quarter = 60,000,000 / BPM)
  - Note-on (0x90) and note-off (0x80) events with velocity
  - Variable-length quantity encoding for delta times
  - End-of-track meta-event
  - No pitch bend events (deferred)
  - No multiple tempo events (tempo curve deferred)
- Delta time calculation: `ticks = seconds × (BPM / 60) × 480`
- Export via `UIActivityViewController` (share sheet)
- File UTI registration for .mid files
- Default filename: `MemoToMIDI_[timestamp].mid`

**Test:**
- Generate a .mid from a recorded chord progression
- AirDrop to Mac → double-click → Logic opens
- Verify: correct pitches, correct timing, correct tempo in transport bar
- Verify: notes have varying velocity (not all 127)
- Edge case: single note, very long recording (2+ minutes), very fast passages

### Phase 5: Piano Roll + Preview Playback
**Goal:** Visual display of detected notes with playback and manual deletion.

**Build:**
- `PianoRollView.swift`: custom SwiftUI renderer
  - Dark background, notes as rounded rectangles
  - Color = velocity (brighter = louder)
  - Horizontal axis = time, vertical axis = pitch
  - Piano key labels on left edge
  - Time ruler on top (seconds, or beats/bars if BPM set)
  - Pinch-to-zoom horizontal + vertical
  - Scroll both axes
  - Tap to select (highlight), tap again to deselect
  - Multi-select by tapping additional notes
  - Delete button when notes selected
  - Vertical playback scrubber line
- `MIDIPlayer.swift`: playback using AVAudioEngine + AVAudioUnitSampler
  - Load a lightweight .sf2 SoundFont (bundled, <2MB)
  - 2-3 presets: Piano, Clean Guitar, Sine tone
  - Play/pause, scrub to position
- Cleanup sliders panel (collapsible bottom sheet):
  - Sensitivity (onset threshold)
  - Min note length
  - Note merge distance
  - Adjusting any slider re-runs extraction, updates piano roll live

**Test:**
- View detected notes from a recorded chord progression
- Play back — sounds match what was played
- Delete 2-3 spurious notes, re-export to Logic, confirm they're gone
- Zoom in to verify individual note timing looks correct

### Phase 6: Flash Tempo + Tempo Detection + Export Options
**Goal:** Visual metronome, auto-BPM, Fixed BPM + quantization export.

**Build:**
- `FlashTempoView.swift`: visual metronome
  - BPM input + tap tempo (average last 4 tap intervals)
  - Screen pulse on each beat using `mach_absolute_time` for precision
  - Beat-1 accent (brighter flash)
  - Count-in: 1 bar of flashes before recording starts
  - Quarter note and eighth note flash patterns
- `TempoDetector.swift`: auto-BPM from onset intervals
  - Analyze strong onsets from the onset probability matrix
  - Autocorrelation or histogram on inter-onset intervals
  - Report confidence level
- `Quantizer.swift`: soft quantization
  - Grid presets: 1/4, 1/8, 1/16, 1/8 triplet, 1/16 triplet
  - Strength slider: 0% (raw) to 100% (fully quantized)
  - Non-destructive — original timing preserved internally
- Export options UI:
  - Fixed BPM (default) — raw timing against grid
  - Fixed BPM + Quantization — nudged toward grid
  - BPM source: Flash Tempo setting, manual entry, or auto-detected

**Test:**
- Flash Tempo at 100 BPM: verify no visible drift over 60 seconds
- Record to Flash Tempo at 120 BPM → export → Logic shows 120 BPM, notes near grid
- Record without Flash Tempo → auto-detect BPM → verify within ±3 BPM of actual
- Export with 50% quantization to 1/8 notes → Logic shows notes nudged but not robotic

### Phase 7: Recording Library + Polish
**Goal:** Persistent storage, library UI, final polish.

**Build:**
- `RecordingListView.swift`: main screen, list of past recordings
  - Most recent first
  - Each row: name, date, duration, waveform thumbnail
  - Tap to re-open in editor
  - Swipe to delete (with confirmation)
  - Rename recordings
- `RecordingStore.swift`: persistence
  - .wav files in Documents directory
  - JSON metadata file per recording (name, date, duration, last-used settings)
  - Cached note events (optional, re-derivable)
- Polish:
  - App icon
  - Launch screen
  - Dark theme throughout
  - Smooth transitions between screens
  - Error handling (microphone denied, disk full, model load failure)

**Test:** Record 5+ ideas over several sessions. Re-open old recordings, re-process with different settings, export. Full workflow should feel like 3 taps: record → stop → export.

---

## Phase Dependencies + Parallelism

```
Phase 0 ✓ (DONE)
    │
    ├── Phase 1 (audio capture)  ←── can start immediately
    │       │
    │       └── Phase 2 (inference) ←── needs Phase 1 audio
    │               │
    │               ├── Phase 3 (note extraction) ←── needs Phase 2 output
    │               │       │
    │               │       ├── Phase 4 (MIDI export) ←── needs [NoteEvent]
    │               │       │
    │               │       └── Phase 5 (piano roll) ←── needs [NoteEvent], can parallel with 4
    │               │               │
    │               │               └── Phase 6 (tempo + quantization) ←── needs piano roll + export
    │               │                       │
    │               │                       └── Phase 7 (library + polish)
```

Phase 4 and Phase 5 can run in parallel — they both consume `[NoteEvent]` but don't depend on each other.

---

## Build Setup Instructions

### Prerequisites
- Mac with Xcode 15+ installed
- iPhone 13 (or newer) for on-device testing
- Apple Developer account (free is fine for personal device testing)
- The CQT spike workspace with `BasicPitch_from_tf.mlpackage`

### Step 1: Create Xcode Project
1. Open Xcode → File → New → Project
2. Choose: iOS → App
3. Settings:
   - Product Name: `MemoToMIDI`
   - Team: your Apple ID
   - Organization Identifier: `com.yourname` (or whatever you use)
   - Interface: SwiftUI
   - Language: Swift
   - Storage: None (we'll do file-based persistence, not Core Data or SwiftData)
   - Uncheck "Include Tests" for now (add later if wanted)
4. Deployment target: iOS 17.0
5. Save to your preferred location

### Step 2: Bundle the CoreML Model
1. Drag `BasicPitch_from_tf.mlpackage` from `cqt_spike/` into the Xcode project navigator
2. When prompted: check "Copy items if needed", add to target MemoToMIDI
3. Xcode will auto-generate a Swift class for the model — verify it appears in the project
4. Build once (Cmd+B) to confirm the model compiles without errors

### Step 3: Configure Info.plist
Add these keys (Xcode may have migrated some to the target's Info tab):
```
NSMicrophoneUsageDescription = "MemoToMIDI needs microphone access to record your musical ideas."
```

Also register the .mid file type for export. In your target's Info tab, under "Document Types" or in Info.plist:
```xml
<key>CFBundleDocumentTypes</key>
<array>
    <dict>
        <key>CFBundleTypeName</key>
        <string>MIDI File</string>
        <key>CFBundleTypeRole</key>
        <string>Editor</string>
        <key>LSItemContentTypes</key>
        <array>
            <string>public.midi-audio</string>
        </array>
    </dict>
</array>
```

### Step 4: Create the Directory Structure
In the Xcode project navigator, create groups matching the project structure:
- Models/
- Audio/
- ML/
- Processing/
- MIDI/
- Analysis/
- Storage/
- Views/
- Views/Components/
- Resources/
- Resources/Sounds/

### Step 5: Add NoteEvent Model
Create `Models/NoteEvent.swift` as the first file — almost every other file depends on it:

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

(No `pitchBend` field — deferred to post-v1.)

### Step 6: Copy BasicPitchRunner Logic
Copy `BasicPitchRunner.swift` from the spike into `ML/BasicPitchInference.swift`. It'll need refactoring from a CLI tool to an async class, but the windowing and stitching logic is proven and shouldn't change.

### Step 7: Verify Build
- Connect iPhone 13
- Select it as the run destination
- Build and run (Cmd+R)
- The app should launch (blank screen is fine at this point)
- Check the console for any CoreML model loading warnings

### You're Ready for Phase 1
At this point the Xcode project exists, the model is bundled, the data model is defined, and the inference logic is available to port. Phase 1 (audio capture) can begin.

---

## Key Constants Reference

Keep these in a shared file (e.g., `Models/Constants.swift`):

```swift
enum AudioConstants {
    static let sampleRate: Double = 22050
    static let windowSamples: Int = 43844
    static let hopSamples: Int = 36164
    static let overlapLen: Int = 7680
    static let prependSamples: Int = 3840
}

enum MIDIConstants {
    static let ppqn: UInt16 = 480
    static let defaultTempo: Double = 120.0
    static let defaultOnsetThreshold: Float = 0.45
    static let defaultFrameThreshold: Float = 0.3
    static let minNoteLengthMs: Double = 58.0
    static let guitarRangeLow: UInt8 = 40   // E2
    static let guitarRangeHigh: UInt8 = 84  // C6
}
```
