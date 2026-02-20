import SwiftUI

struct EditorView: View {
    let inferenceResult: InferenceResult
    let tuningResult: TuningResult

    @State private var parameters: ExtractionParameters
    @State private var notes: [NoteEvent]
    @State private var isCleanupExpanded = true
    @State private var pianoRollFitRequestID = 0

    private let correctedResult: InferenceResult

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
        VStack(spacing: 10) {
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

            PianoRollView(notes: $notes, fitRequestID: pianoRollFitRequestID)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .padding(.horizontal)

            CleanupSlidersView(parameters: $parameters, isExpanded: $isCleanupExpanded)
                .padding(.horizontal)
                .padding(.bottom, 8)
        }
        .navigationTitle("Editor")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                MIDIExportShareButton(notes: notes, bpm: MIDIConstants.defaultTempo) {
                    Label("Export MIDI", systemImage: "square.and.arrow.up")
                }
                .disabled(notes.isEmpty)
            }
        }
        .onChange(of: parameters.onsetThreshold) { _, _ in
            reextractNotes()
        }
        .onChange(of: parameters.minNoteLengthMs) { _, _ in
            reextractNotes()
        }
        .onChange(of: parameters.mergeGapMs) { _, _ in
            reextractNotes()
        }
        .onChange(of: parameters.frameThreshold) { _, _ in
            reextractNotes()
        }
        .onAppear {
            pianoRollFitRequestID += 1
        }
    }

    private var tuningLabel: String {
        let cents = Int(tuningResult.centsOffset.rounded())
        return "Tuning \(cents >= 0 ? "+" : "")\(cents)c"
    }

    private func reextractNotes() {
        notes = NoteExtractor.extract(from: correctedResult, parameters: parameters)
        pianoRollFitRequestID += 1
    }
}
