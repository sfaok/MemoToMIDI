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
    @State private var playbackMode: PlaybackMode = .midiOnly
    @State private var overlayMix: Double = 0.5
    @State private var mixDebounceTask: Task<Void, Never>?
    @State private var didInitializeNotes = false
    @State private var beatMap: BeatMap?
    @State private var manualBPMText: String
    @State private var isBeatTapPresented = false

    private let correctedResult: InferenceResult
    private enum PlaybackMode: String, CaseIterable {
        case midiOnly = "MIDI Only"
        case withRecording = "With Recording"
    }

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
            .onChange(of: playbackMode) { _, newMode in
                stopTransport()
                if newMode == .withRecording {
                    prepareOverlayResources()
                }
            }
            .onChange(of: overlayMix) { _, _ in
                scheduleOverlayMixUpdate()
            }
            .onChange(of: notePreparationSignature) { _, _ in
                stopTransport()
                if playbackMode == .withRecording {
                    prepareOverlayResources()
                }
            }
            .onChange(of: player.isPlaying) { wasPlaying, isPlaying in
                guard playbackMode == .withRecording else { return }
                guard wasPlaying, !isPlaying else { return }
                prepareOverlayResources()
            }
            .onAppear {
                initializeNotesIfNeeded()
                do {
                    try player.setup()
                } catch {
                    // Playback controls remain visible even if setup fails.
                }
                player.setPreset(selectedPreset)
                if playbackMode == .withRecording {
                    prepareOverlayResources()
                }
                pianoRollFitRequestID += 1
            }
            .onDisappear {
                mixDebounceTask?.cancel()
                stopTransport()
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
            seekTransport(to: time)
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

    private var notePreparationSignature: Int {
        var hasher = Hasher()
        hasher.combine(notes.count)
        for note in notes {
            hasher.combine(note.id)
            hasher.combine(note.pitch)
            hasher.combine(note.startTime.bitPattern)
            hasher.combine(note.duration.bitPattern)
            hasher.combine(note.velocity)
        }
        return hasher.finalize()
    }

    private var tuningLabel: String {
        let cents = Int(tuningResult.centsOffset.rounded())
        return "Tuning \(cents >= 0 ? "+" : "")\(cents)c"
    }

    private var playbackControls: some View {
        VStack(spacing: 10) {
            Picker("Playback Mode", selection: $playbackMode) {
                ForEach(PlaybackMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack(spacing: 12) {
                Button {
                    seekTransport(to: max(player.currentTime - 2.0, 0))
                } label: {
                    Image(systemName: "backward.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.bordered)
                .disabled(!canPlayCurrentMode || playbackMode == .withRecording)

                Button {
                    togglePlayback()
                } label: {
                    Image(systemName: isTransportPlaying ? "pause.fill" : "play.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canPlayCurrentMode)

                Button {
                    stopTransport()
                } label: {
                    Image(systemName: "stop.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.bordered)
                .disabled(!canPlayCurrentMode)

                Picker("Instrument", selection: $selectedPreset) {
                    ForEach(PlaybackPreset.allCases, id: \.self) { preset in
                        Text(preset.rawValue).tag(preset)
                    }
                }
                .pickerStyle(.segmented)
            }

            if playbackMode == .withRecording {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Original")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("MIDI")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $overlayMix, in: 0...1)
                }

                if player.isPreparingOverlay {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Preparing overlay...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                if let overlayError = player.overlayPreparationError, !overlayError.isEmpty {
                    Text(overlayError)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
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

    private var canPlayCurrentMode: Bool {
        guard !notes.isEmpty else { return false }
        if player.isPreparingOverlay { return false }
        if playbackMode == .withRecording {
            return audioRecorder.lastRecordingURL != nil
        }
        return true
    }

    private var isTransportPlaying: Bool {
        return player.isPlaying
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
                    stopTransport()
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
        stopTransport()
        let extracted = NoteExtractor.extract(from: correctedResult, parameters: parameters)
        let refined = TransientRefiner.refine(notes: extracted, audioBuffer: audioBuffer)
        notes = NoteExtractor.applyAudioVelocity(notes: refined, audioBuffer: audioBuffer)
        pianoRollFitRequestID += 1
    }

    private func initializeNotesIfNeeded() {
        guard !didInitializeNotes else { return }
        didInitializeNotes = true
        let extracted = NoteExtractor.extract(from: correctedResult, parameters: parameters)
        let refined = TransientRefiner.refine(notes: extracted, audioBuffer: audioBuffer)
        notes = NoteExtractor.applyAudioVelocity(notes: refined, audioBuffer: audioBuffer)
    }

    private func togglePlayback() {
        if playbackMode == .withRecording {
            if isTransportPlaying {
                pauseOverlayPlayback()
            } else {
                playOverlay()
            }
            return
        }

        if player.isPlaying {
            player.pause()
        } else {
            player.play(notes: notes)
        }
    }

    private func stopTransport() {
        player.stop()
        if playbackMode == .withRecording {
            player.prepare(notes: notes)
        }
    }

    private func seekTransport(to time: Double) {
        if playbackMode == .withRecording {
            return
        }

        let clampedTime = max(time, 0)

        player.seek(to: clampedTime)
    }

    private func playOverlay() {
        guard prepareOverlayResources() else {
            return
        }
        let gains = overlayGains
        player.playOverlay(
            notes: notes,
            recordingGain: gains.recordingGain,
            midiGain: gains.midiGain
        )
    }

    private func pauseOverlayPlayback() {
        player.pause()
    }

    private var overlayGains: (recordingGain: Float, midiGain: Float) {
        let clamped = min(max(overlayMix, 0), 1)
        return (Float(1 - clamped), Float(clamped))
    }

    private func scheduleOverlayMixUpdate() {
        guard playbackMode == .withRecording else { return }

        let gains = overlayGains
        mixDebounceTask?.cancel()
        mixDebounceTask = Task {
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }
            player.updateOverlayMix(
                recordingGain: gains.recordingGain,
                midiGain: gains.midiGain
            )
        }
    }

    @discardableResult
    private func prepareOverlayResources() -> Bool {
        guard let recordingURL = audioRecorder.lastRecordingURL else {
            audioRecorder.errorMessage = AudioRecorder.AudioRecorderError.missingRecording.localizedDescription
            return false
        }

        do {
            try player.loadRecordingForOverlay(url: recordingURL)
            player.prepare(notes: notes)
            return true
        } catch {
            audioRecorder.errorMessage = error.localizedDescription
            return false
        }
    }
}
