import SwiftUI

struct BeatTapView: View {
    @Environment(\.dismiss) private var dismiss

    @ObservedObject var audioRecorder: AudioRecorder
    let recordingDuration: Double
    let initialBeatMap: BeatMap?
    let onDone: (BeatMap) -> Void

    @State private var taps: [Double]
    @State private var isTouchSequenceActive = false
    @State private var flashOpacity: Double = 0
    @State private var beatCounterScale: CGFloat = 1
    @State private var didReturnResult = false

    init(
        audioRecorder: AudioRecorder,
        recordingDuration: Double,
        initialBeatMap: BeatMap?,
        onDone: @escaping (BeatMap) -> Void
    ) {
        self.audioRecorder = audioRecorder
        self.recordingDuration = recordingDuration
        self.initialBeatMap = initialBeatMap
        self.onDone = onDone
        _taps = State(initialValue: initialBeatMap?.taps ?? [])
    }

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()

            VStack(spacing: 24) {
                topBar
                beatStats
                tapSurface
                bottomControls
            }
            .padding()

            Color.accentColor
                .opacity(flashOpacity)
                .ignoresSafeArea()
                .allowsHitTesting(false)
        }
        .onDisappear {
            audioRecorder.stopPlayback()
        }
        .onChange(of: audioRecorder.isPlaying) { wasPlaying, isPlaying in
            guard wasPlaying, !isPlaying else { return }
            guard !didReturnResult else { return }
            if hasReachedPlaybackEnd {
                returnResultAndDismiss()
            }
        }
    }

    private var topBar: some View {
        HStack {
            Button("Cancel") {
                dismiss()
            }
            .buttonStyle(.bordered)

            Spacer()

            Button("Done") {
                returnResultAndDismiss()
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private var beatStats: some View {
        VStack(spacing: 10) {
            Text(beatCountText)
                .font(.title2.weight(.semibold))
                .scaleEffect(beatCounterScale)

            Text(bpmText)
                .font(.system(size: 34, weight: .bold, design: .rounded))
                .monospacedDigit()

            Text("Tap on touch-down while the recording plays")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private var tapSurface: some View {
        RoundedRectangle(cornerRadius: 22, style: .continuous)
            .fill(Color(.secondarySystemBackground))
            .overlay(
                VStack(spacing: 8) {
                    Image(systemName: "hand.tap.fill")
                        .font(.system(size: 34))
                    Text("Tap Beat")
                        .font(.headline)
                    Text(timeLabel(for: audioRecorder.currentPlaybackTime))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            )
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
            )
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !isTouchSequenceActive else { return }
                        isTouchSequenceActive = true
                        registerTap()
                    }
                    .onEnded { _ in
                        isTouchSequenceActive = false
                    }
            )
    }

    private var bottomControls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Button {
                    togglePlayback()
                } label: {
                    Label(audioRecorder.isPlaying ? "Pause" : "Play", systemImage: audioRecorder.isPlaying ? "pause.fill" : "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)

                Button("Clear / Start Over") {
                    clearAndReset()
                }
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)
            }

            if let errorMessage = audioRecorder.errorMessage, !errorMessage.isEmpty {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var currentBeatMap: BeatMap {
        BeatMap(
            taps: taps.sorted(),
            timeSignature: initialBeatMap?.timeSignature ?? 4
        )
    }

    private var beatCountText: String {
        if taps.isEmpty {
            return "No beats tapped yet"
        }
        if taps.count <= 6 {
            return (1...taps.count).map { "Beat \($0)" }.joined(separator: " · ")
        }
        return "Beat \(taps.count)"
    }

    private var bpmText: String {
        guard let bpm = currentBeatMap.bpm else { return "-- BPM" }
        return "\(Int(bpm.rounded())) BPM"
    }

    private var hasReachedPlaybackEnd: Bool {
        let duration = audioRecorder.playbackDuration > 0 ? audioRecorder.playbackDuration : recordingDuration
        guard duration > 0 else { return false }
        return audioRecorder.currentPlaybackTime >= duration - 0.03
    }

    private func togglePlayback() {
        if audioRecorder.isPlaying {
            audioRecorder.pausePlayback()
            return
        }

        do {
            try audioRecorder.playLastRecording()
        } catch {
            audioRecorder.errorMessage = error.localizedDescription
        }
    }

    private func registerTap() {
        guard audioRecorder.isPlaying else { return }
        let time = audioRecorder.playbackTimeSnapshot()
        guard time.isFinite else { return }
        guard time >= 0 else { return }

        taps.append(time)
        triggerTapFeedback()
    }

    private func triggerTapFeedback() {
        flashOpacity = 0.18
        withAnimation(.easeOut(duration: 0.18)) {
            flashOpacity = 0
        }

        withAnimation(.spring(response: 0.2, dampingFraction: 0.45)) {
            beatCounterScale = 1.12
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            withAnimation(.spring(response: 0.26, dampingFraction: 0.8)) {
                beatCounterScale = 1.0
            }
        }
    }

    private func clearAndReset() {
        taps = []
        audioRecorder.stopPlayback()
    }

    private func returnResultAndDismiss() {
        guard !didReturnResult else { return }
        didReturnResult = true
        onDone(currentBeatMap)
        dismiss()
    }

    private func timeLabel(for time: Double) -> String {
        let clamped = max(time, 0)
        let minutes = Int(clamped) / 60
        let seconds = clamped.truncatingRemainder(dividingBy: 60)
        return String(format: "%d:%05.2f", minutes, seconds)
    }
}
