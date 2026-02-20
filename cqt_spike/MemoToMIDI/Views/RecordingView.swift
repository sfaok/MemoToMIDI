import SwiftUI

struct RecordingView: View {
    private struct TestMIDIShareItem: Identifiable {
        let id = UUID()
        let url: URL
    }

    @StateObject private var audioRecorder = AudioRecorder()
    @State private var inferenceEngine = BasicPitchInference()
    @State private var microphoneAllowed = false
    @State private var lastFileDescription: AudioFileDescription?
    @State private var cachedInferenceResult: InferenceResult?
    @State private var cachedAudioBuffer: [Float] = []
    @State private var tuningResult: TuningResult?
    @State private var extractedNotes: [NoteEvent] = []
    @State private var lastInferenceDuration: TimeInterval?
    @State private var isRunningInference = false
    @State private var inferenceProgress: Double = 0
    @State private var didAttemptWarmUp = false
    @State private var processingRequestID: Int = 0
    @State private var lastProcessedRecordingURL: URL?
    @State private var testMIDIShareItem: TestMIDIShareItem?
    @State private var testMIDIExportError: MIDIExporterError?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Phase 2: Record + Inference")
                .font(.headline)

            WaveformView(samples: audioRecorder.waveformSamples)
                .frame(height: 150)

            Text(statusText)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Button(audioRecorder.isRecording ? "Stop" : "Record") {
                    handleRecordButton()
                }
                .buttonStyle(.borderedProminent)
                .tint(audioRecorder.isRecording ? .red : .blue)
                .disabled((!microphoneAllowed && !audioRecorder.isRecording) || isRunningInference)

                Button(audioRecorder.isPlaying ? "Stop Playback" : "Play Last") {
                    handlePlaybackButton()
                }
                .buttonStyle(.bordered)
                .disabled(audioRecorder.lastRecordingURL == nil || audioRecorder.isRecording || isRunningInference)
            }

            Button("Generate Test MIDI") {
                generateKnownTestMIDI()
            }
            .buttonStyle(.bordered)
            .disabled(audioRecorder.isRecording || isRunningInference)

            if isRunningInference {
                ProgressView(value: inferenceProgress)
                    .progressViewStyle(.linear)
                Text(String(format: "Inference progress: %.0f%%", inferenceProgress * 100))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let lastRecordingURL = audioRecorder.lastRecordingURL {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Last Recording")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(lastRecordingURL.lastPathComponent)
                        .font(.callout)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let lastFileDescription {
                        Text(fileDetails(for: lastFileDescription))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }

            if let cachedInferenceResult {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Cached ML Output")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Note shape: [\(cachedInferenceResult.noteActivations.count), \(cachedInferenceResult.noteActivations.first?.count ?? 0)]")
                        .font(.callout)
                    Text("Onset shape: [\(cachedInferenceResult.onsetActivations.count), \(cachedInferenceResult.onsetActivations.first?.count ?? 0)]")
                        .font(.callout)
                    Text(String(format: "Frame duration: %.6f s", cachedInferenceResult.frameDuration))
                        .font(.callout)
                    if let lastInferenceDuration {
                        Text(String(format: "Processing time: %.3f s", lastInferenceDuration))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    let topActivationRows = topActivationRows(for: cachedInferenceResult)
                    if !topActivationRows.isEmpty {
                        Text("Top 5 activations (mid frame)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.top, 4)
                        ForEach(Array(topActivationRows.enumerated()), id: \.offset) { row in
                            Text(row.element)
                                .font(.caption.monospacedDigit())
                        }
                    }
                    MIDIExportShareButton(notes: extractedNotes, bpm: MIDIConstants.defaultTempo)
                        .padding(.top, 8)
                    if let tuningResult {
                        NavigationLink {
                            EditorView(
                                inferenceResult: cachedInferenceResult,
                                tuningResult: tuningResult,
                                audioBuffer: cachedAudioBuffer
                            )
                        } label: {
                            Label("View Notes", systemImage: "pianokeys")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .padding(.top, 8)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }

            if let errorMessage = audioRecorder.errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }

            Spacer(minLength: 0)
        }
        .padding()
        .task {
            microphoneAllowed = await audioRecorder.requestMicrophoneAccess()
            if !microphoneAllowed {
                audioRecorder.errorMessage = "Microphone access is denied. Enable it in Settings to record."
            }
            await warmUpModelIfNeeded()
        }
        .task(id: processingRequestID) {
            guard processingRequestID > 0 else { return }
            await runInferenceForLastRecording()
        }
        .onChange(of: audioRecorder.isRecording) { wasRecording, isRecording in
            guard wasRecording, !isRecording else { return }
            refreshLastFileDescription()
            processingRequestID += 1
        }
        .sheet(item: $testMIDIShareItem, onDismiss: {
            testMIDIShareItem = nil
        }) { item in
            MIDIShareSheet(items: [item.url])
        }
        .alert(
            "Test MIDI Export Failed",
            isPresented: Binding(
                get: { testMIDIExportError != nil },
                set: { shouldShow in
                    if !shouldShow {
                        testMIDIExportError = nil
                    }
                }
            ),
            actions: {
                Button("OK", role: .cancel) {
                    testMIDIExportError = nil
                }
            },
            message: {
                Text(testMIDIExportError?.errorDescription ?? "Unknown error")
            }
        )
    }

    private var statusText: String {
        if isRunningInference {
            return String(format: "Running ML inference (%.0f%%)", inferenceProgress * 100)
        }
        if audioRecorder.isRecording {
            return String(format: "Recording %.2f s @ %.0f Hz mono", audioRecorder.recordingDuration, AudioConstants.sampleRate)
        }
        if audioRecorder.isPlaying {
            return "Playing back last recording"
        }
        if cachedInferenceResult != nil {
            return "Inference cached and ready for Phase 3 extraction"
        }
        if audioRecorder.lastRecordingURL != nil {
            return "Ready to record again"
        }
        return "Press Record to capture audio"
    }

    private func handleRecordButton() {
        if audioRecorder.isRecording {
            audioRecorder.stopRecording()
            return
        }

        do {
            cachedInferenceResult = nil
            cachedAudioBuffer = []
            tuningResult = nil
            extractedNotes = []
            lastInferenceDuration = nil
            inferenceProgress = 0
            lastProcessedRecordingURL = nil
            try audioRecorder.startRecording()
        } catch {
            audioRecorder.errorMessage = error.localizedDescription
        }
    }

    private func handlePlaybackButton() {
        if audioRecorder.isPlaying {
            audioRecorder.stopPlayback()
            return
        }
        do {
            try audioRecorder.playLastRecording()
        } catch {
            audioRecorder.errorMessage = error.localizedDescription
        }
    }

    private func generateKnownTestMIDI() {
        let testNotes = [
            NoteEvent(pitch: 60, startTime: 0.0, duration: 0.5, velocity: 100),  // C4, beat 1
            NoteEvent(pitch: 64, startTime: 0.5, duration: 0.5, velocity: 90),   // E4, beat 2
            NoteEvent(pitch: 67, startTime: 1.0, duration: 0.5, velocity: 80),   // G4, beat 3
            NoteEvent(pitch: 72, startTime: 1.5, duration: 1.0, velocity: 110)   // C5, beat 4 (held 2 beats)
        ]

        let midiData = MIDIFileWriter.write(notes: testNotes, bpm: 120.0)
        print("MIDI file size: \(midiData.count) bytes")

        do {
            let exportedTestMIDIURL = try MIDIExporter.exportToFile(notes: testNotes, bpm: 120.0)
            guard FileManager.default.fileExists(atPath: exportedTestMIDIURL.path) else {
                throw MIDIExporterError.fileUnavailable(path: exportedTestMIDIURL.path)
            }
            print("Exported to: \(exportedTestMIDIURL.path)")
            testMIDIShareItem = TestMIDIShareItem(url: exportedTestMIDIURL)
        } catch let exporterError as MIDIExporterError {
            testMIDIExportError = exporterError
        } catch {
            testMIDIExportError = .failedToWriteFile(underlying: error)
        }
    }

    private func refreshLastFileDescription() {
        guard let url = audioRecorder.lastRecordingURL else {
            lastFileDescription = nil
            return
        }
        lastFileDescription = try? AudioFileReader.describeFile(at: url)
    }

    private func fileDetails(for description: AudioFileDescription) -> String {
        String(
            format: "%.0f Hz, %d ch, %.2f s (%lld frames)",
            description.sampleRate,
            description.channelCount,
            description.duration,
            description.frameCount
        )
    }

    @MainActor
    private func warmUpModelIfNeeded() async {
        guard !didAttemptWarmUp else { return }
        didAttemptWarmUp = true

        let started = Date()
        do {
            try await inferenceEngine.warmUp()
            let elapsed = Date().timeIntervalSince(started)
            print(String(format: "BasicPitch warm-up completed in %.3f s", elapsed))
        } catch {
            print("BasicPitch warm-up failed: \(error.localizedDescription)")
        }
    }

    @MainActor
    private func runInferenceForLastRecording() async {
        guard let url = audioRecorder.lastRecordingURL else { return }
        if lastProcessedRecordingURL == url, cachedInferenceResult != nil {
            return
        }

        isRunningInference = true
        inferenceProgress = 0
        defer { isRunningInference = false }

        do {
            let audioSamples = try await Task.detached(priority: .userInitiated) {
                try AudioFileReader.readSamples(from: url)
            }.value
            cachedAudioBuffer = audioSamples

            let startTime = Date()
            let result = try await inferenceEngine.process(
                audioSamples: audioSamples,
                progressCallback: { progress in
                    inferenceProgress = progress
                }
            )
            let elapsed = Date().timeIntervalSince(startTime)

            cachedInferenceResult = result
            let tuning = TuningDetector.detect(from: result)
            let corrected = PitchCorrector.correct(result: result, centsOffset: tuning.centsOffset)
            let extracted = NoteExtractor.extract(from: corrected)
            tuningResult = tuning
            extractedNotes = TransientRefiner.refine(notes: extracted, audioBuffer: audioSamples)
            lastInferenceDuration = elapsed
            lastProcessedRecordingURL = url
            printValidationSummary(result: result, elapsed: elapsed)
            printExtractionDiagnostics(result: result)
        } catch {
            audioRecorder.errorMessage = "Inference failed: \(error.localizedDescription)"
        }
    }

    private func printValidationSummary(result: InferenceResult, elapsed: TimeInterval) {
        print("Inference complete.")
        print("Note activations shape: [\(result.noteActivations.count), \(result.noteActivations.first?.count ?? 0)]")
        print("Onset activations shape: [\(result.onsetActivations.count), \(result.onsetActivations.first?.count ?? 0)]")

        guard !result.noteActivations.isEmpty else {
            print("No frames available for activation printout.")
            print("Frame duration: \(result.frameDuration) seconds")
            print("Total duration: \(Double(result.noteActivations.count) * result.frameDuration) seconds")
            print(String(format: "Processing time: %.3f seconds", elapsed))
            return
        }

        let midFrame = result.noteActivations.count / 2
        let frame = result.noteActivations[midFrame]
        let indexed = frame.enumerated().sorted { $0.element > $1.element }
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        print("Top 5 activations at frame \(midFrame):")
        for rank in 0..<min(5, indexed.count) {
            let (columnIndex, activation) = indexed[rank]
            let midiNote = columnIndex + 21
            let name = noteNames[midiNote % 12]
            let octave = (midiNote / 12) - 1
            print("  MIDI \(midiNote) (\(name)\(octave)): \(String(format: "%.4f", activation))")
        }

        print("Frame duration: \(result.frameDuration) seconds")
        print("Total duration: \(Double(result.noteActivations.count) * result.frameDuration) seconds")
        print(String(format: "Processing time: %.3f seconds", elapsed))
    }

    private func topActivationRows(for result: InferenceResult) -> [String] {
        guard !result.noteActivations.isEmpty else { return [] }

        let midFrame = result.noteActivations.count / 2
        let frame = result.noteActivations[midFrame]
        let indexed = frame.enumerated().sorted { $0.element > $1.element }
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        return indexed.prefix(5).map { entry in
            let columnIndex = entry.offset
            let activation = entry.element
            let midiNote = columnIndex + 21
            let name = noteNames[midiNote % 12]
            let octave = (midiNote / 12) - 1
            return "MIDI \(midiNote) (\(name)\(octave)): \(String(format: "%.4f", activation))"
        }
    }

    private func printExtractionDiagnostics(result: InferenceResult) {
        let notes = NoteExtractor.extract(from: result)
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        print("Extracted \(notes.count) notes (raw, pre-correction)")

        // Tuning detection test
        let tuning = TuningDetector.detect(from: result)
        print("Tuning: \(String(format: "%.1f", tuning.centsOffset)) cents")
        print("Reference: A=\(String(format: "%.1f", tuning.referenceFrequency)) Hz")
        print("Confidence: \(String(format: "%.2f", tuning.confidence))")

        // Correction test
        let corrected = PitchCorrector.correct(result: result, centsOffset: tuning.centsOffset)
        let notesCorrected = NoteExtractor.extract(from: corrected)
        print("Without correction: \(notes.count) notes")
        print("With correction: \(notesCorrected.count) notes")
        print("Corrected note list:")
        for n in notesCorrected {
            let correctedName = noteNames[Int(n.pitch) % 12]
            let correctedOctave = (Int(n.pitch) / 12) - 1
            print(
                "  \(correctedName)\(correctedOctave) (MIDI \(n.pitch)): " +
                "start=\(String(format: "%.3f", n.startTime))s " +
                "dur=\(String(format: "%.3f", n.duration))s " +
                "vel=\(n.velocity)"
            )
        }

        let sensitive = ExtractionParameters(
            onsetThreshold: 0.3,
            frameThreshold: 0.2,
            minNoteLengthMs: 30,
            mergeGapMs: 30
        )
        let strict = ExtractionParameters(
            onsetThreshold: 0.6,
            frameThreshold: 0.5,
            minNoteLengthMs: 100,
            mergeGapMs: 30
        )

        let notesSensitive = NoteExtractor.extract(from: result, parameters: sensitive)
        let notesStrict = NoteExtractor.extract(from: result, parameters: strict)
        print("Sensitive: \(notesSensitive.count) notes")
        print("Strict: \(notesStrict.count) notes")

        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = NoteExtractor.extract(from: result)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        print(
            "100 extractions in \(String(format: "%.1f", elapsed * 1000))ms " +
            "(avg \(String(format: "%.2f", elapsed * 10))ms each)"
        )
    }
}
