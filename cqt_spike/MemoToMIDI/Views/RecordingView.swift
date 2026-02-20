import SwiftUI

struct RecordingView: View {
    @StateObject private var audioRecorder = AudioRecorder()
    @State private var microphoneAllowed = false
    @State private var lastFileDescription: AudioFileDescription?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Phase 1: Record + Playback")
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
                .disabled(!microphoneAllowed && !audioRecorder.isRecording)

                Button(audioRecorder.isPlaying ? "Stop Playback" : "Play Last") {
                    handlePlaybackButton()
                }
                .buttonStyle(.bordered)
                .disabled(audioRecorder.lastRecordingURL == nil || audioRecorder.isRecording)
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
        }
    }

    private var statusText: String {
        if audioRecorder.isRecording {
            return String(format: "Recording %.2f s @ %.0f Hz mono", audioRecorder.recordingDuration, AudioConstants.sampleRate)
        }
        if audioRecorder.isPlaying {
            return "Playing back last recording"
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
            return
        }
        do {
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
}
