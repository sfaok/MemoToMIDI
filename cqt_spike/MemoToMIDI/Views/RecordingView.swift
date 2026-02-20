import SwiftUI

struct RecordingView: View {
    @StateObject private var audioRecorder = AudioRecorder()
    @State private var inferenceEngine = BasicPitchInference()
    @State private var microphoneAllowed = false
    @State private var lastFileDescription: AudioFileDescription?
    @State private var cachedInferenceResult: InferenceResult?
    @State private var lastInferenceDuration: TimeInterval?
    @State private var isRunningInference = false
    @State private var inferenceProgress: Double = 0
    @State private var didAttemptWarmUp = false

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
            refreshLastFileDescription()
            Task {
                await runInferenceForLastRecording()
            }
            return
        }

        do {
            cachedInferenceResult = nil
            lastInferenceDuration = nil
            inferenceProgress = 0
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

        isRunningInference = true
        inferenceProgress = 0
        defer { isRunningInference = false }

        do {
            let audioSamples = try await Task.detached(priority: .userInitiated) {
                try AudioFileReader.readSamples(from: url)
            }.value

            let startTime = Date()
            let result = try await inferenceEngine.process(
                audioSamples: audioSamples,
                progressCallback: { progress in
                    inferenceProgress = progress
                }
            )
            let elapsed = Date().timeIntervalSince(startTime)

            cachedInferenceResult = result
            lastInferenceDuration = elapsed
            printValidationSummary(result: result, elapsed: elapsed)
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
}
