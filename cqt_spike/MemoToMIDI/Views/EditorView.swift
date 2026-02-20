import SwiftUI

struct EditorView: View {
    let inferenceResult: InferenceResult
    let tuningResult: TuningResult

    @StateObject private var player = MIDIPlayer()
    @State private var parameters: ExtractionParameters
    @State private var notes: [NoteEvent]
    @State private var isCleanupExpanded = true
    @State private var pianoRollFitRequestID = 0
    @State private var selectedPreset: PlaybackPreset = .sineWave

    private let correctedResult: InferenceResult

    private struct ExtractionSignature: Equatable {
        let onsetThreshold: Float
        let frameThreshold: Float
        let minNoteLengthMs: Double
        let mergeGapMs: Double
    }

    init(inferenceResult: InferenceResult, tuningResult: TuningResult) {
        self.inferenceResult = inferenceResult
        self.tuningResult = tuningResult

        let corrected = PitchCorrector.correct(result: inferenceResult, centsOffset: tuningResult.centsOffset)
        self.correctedResult = corrected

        let defaultParameters = ExtractionParameters.default
        _parameters = State(initialValue: defaultParameters)
        _notes = State(initialValue: NoteExtractor.extract(from: corrected, parameters: defaultParameters))
    }

    var body: some View {
        editorContent
            .navigationTitle("Editor")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                exportToolbarItem
            }
            .onChange(of: extractionSignature) { _, _ in
                reextractNotes()
            }
            .onChange(of: selectedPreset) { _, newPreset in
                player.setPreset(newPreset)
            }
            .onChange(of: noteIDSignature) { _, _ in
                player.stop()
            }
            .onAppear {
                do {
                    try player.setup()
                } catch {
                    // Playback controls remain visible even if setup fails.
                }
                player.setPreset(selectedPreset)
                pianoRollFitRequestID += 1
            }
            .onDisappear {
                player.stop()
            }
    }

    private var editorContent: some View {
        VStack(spacing: 10) {
            headerRow
            pianoRollSection
            playbackControls
                .padding(.horizontal)
            cleanupSection
        }
    }

    private var headerRow: some View {
        HStack {
            Text("\(notes.count) notes")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)
            Spacer()
            Text(tuningLabel)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal)
    }

    private var pianoRollSection: some View {
        PianoRollView(
            notes: $notes,
            fitRequestID: pianoRollFitRequestID,
            playbackTime: player.currentTime
        ) { time in
            player.seek(to: time)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .padding(.horizontal)
    }

    private var cleanupSection: some View {
        CleanupSlidersView(parameters: $parameters, isExpanded: $isCleanupExpanded)
            .padding(.horizontal)
            .padding(.bottom, 8)
    }

    @ToolbarContentBuilder
    private var exportToolbarItem: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            MIDIExportShareButton(notes: notes, bpm: MIDIConstants.defaultTempo) {
                Label("Export MIDI", systemImage: "square.and.arrow.up")
            }
            .disabled(notes.isEmpty)
        }
    }

    private var extractionSignature: ExtractionSignature {
        ExtractionSignature(
            onsetThreshold: parameters.onsetThreshold,
            frameThreshold: parameters.frameThreshold,
            minNoteLengthMs: parameters.minNoteLengthMs,
            mergeGapMs: parameters.mergeGapMs
        )
    }

    private var noteIDSignature: [UUID] {
        notes.map(\.id)
    }

    private var tuningLabel: String {
        let cents = Int(tuningResult.centsOffset.rounded())
        return "Tuning \(cents >= 0 ? "+" : "")\(cents)c"
    }

    private var playbackControls: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Button {
                    player.seek(to: max(player.currentTime - 2.0, 0))
                } label: {
                    Image(systemName: "backward.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.bordered)
                .disabled(notes.isEmpty)

                Button {
                    if player.isPlaying {
                        player.pause()
                    } else {
                        player.play(notes: notes)
                    }
                } label: {
                    Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderedProminent)
                .disabled(notes.isEmpty)

                Button {
                    player.stop()
                } label: {
                    Image(systemName: "stop.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.bordered)
                .disabled(notes.isEmpty)

                Picker("Instrument", selection: $selectedPreset) {
                    ForEach(PlaybackPreset.allCases, id: \.self) { preset in
                        Text(preset.rawValue).tag(preset)
                    }
                }
                .pickerStyle(.segmented)
            }

            HStack {
                Text(timeLabel(for: player.currentTime))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)

                Spacer()

                Text(timeLabel(for: maxNoteEndTime))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private var maxNoteEndTime: Double {
        notes.map { $0.startTime + $0.duration }.max() ?? 0
    }

    private func timeLabel(for time: Double) -> String {
        let clamped = max(time, 0)
        let minutes = Int(clamped) / 60
        let seconds = clamped.truncatingRemainder(dividingBy: 60)
        return String(format: "%d:%05.2f", minutes, seconds)
    }

    private func reextractNotes() {
        player.stop()
        notes = NoteExtractor.extract(from: correctedResult, parameters: parameters)
        pianoRollFitRequestID += 1
    }
}
