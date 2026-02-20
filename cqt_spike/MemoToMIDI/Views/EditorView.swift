import SwiftUI

struct EditorView: View {
    let inferenceResult: InferenceResult
    let tuningResult: TuningResult
    let audioBuffer: [Float]

    @ObservedObject private var audioRecorder: AudioRecorder
    @StateObject private var player = MIDIPlayer()
    @State private var parameters: ExtractionParameters
    @State private var notes: [NoteEvent]
    @State private var isCleanupExpanded = true
    @State private var pianoRollFitRequestID = 0
    @State private var selectedPreset: PlaybackPreset = .sineWave
    @State private var didInitializeNotes = false
    @State private var beatMap: BeatMap?
    @State private var manualBPMText: String
    @State private var isBeatTapPresented = false

    private let correctedResult: InferenceResult

    private struct ExtractionSignature: Equatable {
        let onsetThreshold: Float
        let frameThreshold: Float
        let minNoteLengthMs: Double
        let mergeGapMs: Double
    }

    init(inferenceResult: InferenceResult, tuningResult: TuningResult, audioBuffer: [Float], audioRecorder: AudioRecorder) {
        self.inferenceResult = inferenceResult
        self.tuningResult = tuningResult
        self.audioBuffer = audioBuffer
        _audioRecorder = ObservedObject(wrappedValue: audioRecorder)

        let corrected = PitchCorrector.correct(result: inferenceResult, centsOffset: tuningResult.centsOffset)
        self.correctedResult = corrected

        let defaultParameters = ExtractionParameters.default
        _parameters = State(initialValue: defaultParameters)
        _notes = State(initialValue: [])
        _manualBPMText = State(initialValue: String(format: "%.0f", MIDIConstants.defaultTempo))
    }

    var body: some View {
        editorContent
            .navigationTitle("Editor")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                exportToolbarItem
            }
            .sheet(isPresented: $isBeatTapPresented) {
                BeatTapView(
                    audioRecorder: audioRecorder,
                    recordingDuration: recordingDuration,
                    initialBeatMap: beatMap
                ) { newBeatMap in
                    beatMap = newBeatMap
                }
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
                initializeNotesIfNeeded()
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
            tempoSection
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
            MIDIExportShareButton(notes: notes, bpm: exportBPM) {
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

    private var tempoSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Tempo Mapping")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(tempoSummary)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 12) {
                Button {
                    player.stop()
                    isBeatTapPresented = true
                } label: {
                    Label(beatMap?.bpm == nil ? "Tap Beats" : "Retap Beats", systemImage: "hand.tap")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)

                TextField("BPM", text: $manualBPMText)
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                    .textFieldStyle(.roundedBorder)
            }

            Text("Export tempo: \(tempoExportLabel)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
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

    private var recordingDuration: Double {
        max(Double(audioBuffer.count) / AudioConstants.sampleRate, maxNoteEndTime)
    }

    private var manualBPM: Double? {
        let raw = manualBPMText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return nil }
        guard let value = Double(raw), value > 0 else { return nil }
        return value
    }

    private var exportBPM: Double {
        if let tappedBPM = beatMap?.bpm {
            return tappedBPM
        }
        if let manualBPM {
            return manualBPM
        }
        return MIDIConstants.defaultTempo
    }

    private var tempoSummary: String {
        if let tappedBPM = beatMap?.bpm {
            return "\(Int(tappedBPM.rounded())) BPM (tapped)"
        }
        if let manualBPM {
            return "\(Int(manualBPM.rounded())) BPM (manual)"
        }
        return "\(Int(MIDIConstants.defaultTempo.rounded())) BPM (default)"
    }

    private var tempoExportLabel: String {
        if beatMap?.bpm != nil {
            return "\(Int(exportBPM.rounded())) BPM (tapped)"
        }
        if manualBPM != nil {
            return "\(Int(exportBPM.rounded())) BPM (manual)"
        }
        return "\(Int(exportBPM.rounded())) BPM (default)"
    }

    private func timeLabel(for time: Double) -> String {
        let clamped = max(time, 0)
        let minutes = Int(clamped) / 60
        let seconds = clamped.truncatingRemainder(dividingBy: 60)
        return String(format: "%d:%05.2f", minutes, seconds)
    }

    private func reextractNotes() {
        guard didInitializeNotes else { return }
        player.stop()
        let extracted = NoteExtractor.extract(from: correctedResult, parameters: parameters)
        notes = TransientRefiner.refine(notes: extracted, audioBuffer: audioBuffer)
        pianoRollFitRequestID += 1
    }

    private func initializeNotesIfNeeded() {
        guard !didInitializeNotes else { return }
        didInitializeNotes = true
        let extracted = NoteExtractor.extract(from: correctedResult, parameters: parameters)
        notes = TransientRefiner.refine(notes: extracted, audioBuffer: audioBuffer)
    }
}
