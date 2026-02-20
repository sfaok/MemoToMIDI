# CLAUDE.md — Claude Code Instructions for MemoToMIDI

> **Read AGENTS.md first.** This file adds Claude Code-specific guidance on top of the universal agent instructions.

## Project Overview

iOS app that records guitar → ML note detection → MIDI export. Pure Swift, no dependencies. See AGENTS.md for full architecture, constraints, and data models.

## Quick Reference

```
Build:    Xcode project — open MemoToMIDI.xcodeproj, Cmd+B
Run:      Requires physical iPhone 13+ (CoreML + microphone)
Test:     No formal test target yet — verification is manual (record, check output)
Lint:     Xcode's built-in warnings — treat all warnings as errors
```

## Before You Start Coding

1. Read `AGENTS.md` completely — it has the hard constraints, model constants, and known pitfalls.
2. Check `PHASE-STATUS.md` — it tells you what's already built and what interfaces exist.
3. If you're working on Phase N, confirm all Phase N-1 deliverables exist and match the expected interfaces.

## File Placement Rules

- Follow the project structure in AGENTS.md exactly. Don't create new directories.
- One primary type per file for models and major classes.
- Constants go in `Models/Constants.swift` — don't scatter magic numbers.
- Views go in `Views/` or `Views/Components/`. Components are reusable sub-views.

## Key Patterns to Follow

### Cached ML Output — Never Re-run Inference
```swift
// CORRECT: Re-extract from cached matrices
let notes = extractor.extract(from: cachedNoteMatrix, onsets: cachedOnsetMatrix, params: newParams)

// WRONG: Re-running inference when user adjusts a slider
let output = try await inference.process(audioBuffer: audio) // NO — this is expensive
```

### Async Processing Pattern
```swift
// Use async/await, show progress, don't block UI
func processRecording() async {
    isProcessing = true
    defer { isProcessing = false }
    do {
        let (notes, onsets) = try await inference.process(audioBuffer: audioData)
        self.cachedNotes = notes
        self.cachedOnsets = onsets
        self.noteEvents = extractor.extract(from: notes, onsets: onsets)
    } catch {
        self.error = error
    }
}
```

### MLMultiArray Zero-Copy Pattern
```swift
// Use dataPointer to avoid copying audio data into MLMultiArray
let inputArray = try MLMultiArray(
    dataPointer: UnsafeMutableRawPointer(mutating: windowBuffer),
    shape: [1, NSNumber(value: AudioConstants.windowSamples), 1],
    dataType: .float32,
    strides: [NSNumber(value: AudioConstants.windowSamples), 1, 1]
)
```

## Common Mistakes to Avoid

1. **Adding dependencies.** No SPM, no CocoaPods. If you catch yourself typing `import AudioKit` or adding a Package.swift, stop.
2. **Using UIKit for UI.** SwiftUI only. Exception: `UIActivityViewController` for share sheet via `UIViewControllerRepresentable`.
3. **Re-running inference on parameter changes.** Slider adjustments re-run note extraction from cached matrices. Never re-run CoreML.
4. **Wrong byte order in MIDI writer.** SMF is big-endian. Use `.bigEndian` on all multi-byte writes.
5. **Forgetting model warm-up.** First CoreML inference is slow. Warm at app launch.
6. **Using Double for audio buffers.** All audio is Float32 (`[Float]`). CoreML input is Float32.

## What Success Looks Like Per Phase

| Phase | Success Criteria |
|-------|-----------------|
| 1 | Record 10s guitar, play back clean, .wav is 22050 Hz mono |
| 2 | Record G chord, inference shows G3/B3/D4 activated, <2s for 15s audio |
| 3 | 4-5 chords correctly extracted, sliders change results without re-inference |
| 4 | .mid opens in Logic, correct pitches/timing/velocity, single tempo event |
| 5 | Piano roll shows notes, playback sounds right, delete works |
| 6 | Flash tempo no drift 60s, quantization nudges without roboticizing |
| 7 | Library persists across launches, full workflow is 3 taps |

## Don't Expand Scope

If you think something needs more than what AGENTS.md or the phase spec describes, flag it rather than building it. Especially:
- Don't add note editing beyond delete (no move, no resize in v1)
- Don't add multiple tracks or instruments
- Don't add iCloud sync
- Don't add undo/redo (nice-to-have, not v1)
