import SwiftUI

struct PianoRollView: View {
    @Binding var notes: [NoteEvent]
    let fitRequestID: Int
    let playbackTime: Double
    let onSeekRequested: ((Double) -> Void)?

    @State private var selectedNoteIDs: Set<UUID> = []
    @State private var pixelsPerSecond: CGFloat = 100
    @State private var zoomStartPixelsPerSecond: CGFloat?
    @State private var scrollOffset: CGPoint = .zero
    @State private var viewportSize: CGSize = .zero
    @State private var timelineStart: Double = 0
    @State private var timelineEnd: Double = 1
    @State private var pendingAutoFit = true

    private let labelColumnWidth: CGFloat = 44
    private let rulerHeight: CGFloat = 28
    private let minRowHeight: CGFloat = 12
    private let maxRowHeight: CGFloat = 60
    private let minPixelsPerSecond: CGFloat = 20
    private let maxPixelsPerSecond: CGFloat = 500
    private let scrollSpaceName = "PianoRollScroll"

    init(
        notes: Binding<[NoteEvent]>,
        fitRequestID: Int,
        playbackTime: Double,
        onSeekRequested: ((Double) -> Void)? = nil
    ) {
        self._notes = notes
        self.fitRequestID = fitRequestID
        self.playbackTime = playbackTime
        self.onSeekRequested = onSeekRequested
    }

    private var uniqueDetectedPitches: [Int] {
        Array(Set(notes.map { Int($0.pitch) })).sorted()
    }

    private var visiblePitches: [Int] {
        guard !uniqueDetectedPitches.isEmpty else { return [] }

        let minMIDIPitch = 0
        let maxMIDIPitch = 127
        var pitchSet = Set(uniqueDetectedPitches)

        if let highest = uniqueDetectedPitches.last, highest < maxMIDIPitch {
            pitchSet.insert(highest + 1)
        }
        if let lowest = uniqueDetectedPitches.first, lowest > minMIDIPitch {
            pitchSet.insert(lowest - 1)
        }

        while pitchSet.count < 3 {
            if let low = pitchSet.min(), low > minMIDIPitch {
                pitchSet.insert(low - 1)
                continue
            }
            if let high = pitchSet.max(), high < maxMIDIPitch {
                pitchSet.insert(high + 1)
                continue
            }
            break
        }

        return pitchSet.sorted(by: >)
    }

    private var gridHeight: CGFloat {
        CGFloat(max(visiblePitches.count, 1)) * rowHeight
    }

    private var availableGridWidth: CGFloat {
        max(viewportSize.width - labelColumnWidth, 1)
    }

    private var availableRowsHeight: CGFloat {
        max(viewportSize.height - rulerHeight, minRowHeight)
    }

    private var rowHeight: CGFloat {
        let rowCount = CGFloat(max(visiblePitches.count, 1))
        let fittedHeight = availableRowsHeight / rowCount
        return min(max(fittedHeight, minRowHeight), maxRowHeight)
    }

    private var centeredRowsHeight: CGFloat {
        max(gridHeight, availableRowsHeight)
    }

    private var rowVerticalInset: CGFloat {
        max((centeredRowsHeight - gridHeight) / 2, 0)
    }

    private var visibleDuration: Double {
        max(timelineEnd - timelineStart, 0.1)
    }

    private var gridWidth: CGFloat {
        max(CGFloat(visibleDuration) * pixelsPerSecond, availableGridWidth)
    }

    private var timelineSeconds: [Int] {
        let start = Int(floor(timelineStart))
        let end = Int(ceil(timelineEnd))
        guard end >= start else { return [] }
        return Array(start...end)
    }

    var body: some View {
        GeometryReader { geometry in
            Group {
                if notes.isEmpty {
                    emptyState
                } else {
                    ZStack(alignment: .topTrailing) {
                        ScrollView([.horizontal, .vertical]) {
                            VStack(spacing: 0) {
                                HStack(spacing: 0) {
                                    cornerCell
                                        .offset(x: scrollOffset.x)
                                        .zIndex(3)

                                    timeRuler
                                        .frame(width: gridWidth, height: rulerHeight)
                                        .zIndex(2)
                                }
                                .offset(y: scrollOffset.y)

                                HStack(spacing: 0) {
                                    VStack(spacing: 0) {
                                        Color.clear
                                            .frame(height: rowVerticalInset)

                                        pitchLabels
                                            .frame(width: labelColumnWidth, height: gridHeight)

                                        Color.clear
                                            .frame(height: rowVerticalInset)
                                    }
                                    .frame(width: labelColumnWidth, height: centeredRowsHeight, alignment: .top)
                                    .offset(x: scrollOffset.x)
                                        .zIndex(2)

                                    VStack(spacing: 0) {
                                        Color.clear
                                            .frame(height: rowVerticalInset)

                                        notesCanvas
                                            .frame(width: gridWidth, height: gridHeight)

                                        Color.clear
                                            .frame(height: rowVerticalInset)
                                    }
                                    .frame(width: gridWidth, height: centeredRowsHeight, alignment: .top)
                                }
                            }
                            .background(
                                GeometryReader { scrollGeometry in
                                    Color.clear.preference(
                                        key: PianoRollScrollOffsetKey.self,
                                        value: CGPoint(
                                            x: -scrollGeometry.frame(in: .named(scrollSpaceName)).minX,
                                            y: -scrollGeometry.frame(in: .named(scrollSpaceName)).minY
                                        )
                                    )
                                }
                            )
                        }
                        .coordinateSpace(name: scrollSpaceName)
                        .background(Color(white: 0.1))
                        .clipped()
                        .onPreferenceChange(PianoRollScrollOffsetKey.self) { value in
                            scrollOffset = value
                        }
                        .simultaneousGesture(magnificationGesture)

                        if !selectedNoteIDs.isEmpty {
                            Button(role: .destructive, action: deleteSelectedNotes) {
                                Label("Delete", systemImage: "trash")
                                    .font(.subheadline.weight(.semibold))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.red)
                            .padding(12)
                        }
                    }
                }
            }
            .onAppear {
                viewportSize = geometry.size
                attemptAutoFitIfNeeded()
            }
            .onChange(of: geometry.size) { _, newSize in
                viewportSize = newSize
                attemptAutoFitIfNeeded()
            }
            .onChange(of: fitRequestID) { _, _ in
                pendingAutoFit = true
                attemptAutoFitIfNeeded()
            }
            .onChange(of: notes.isEmpty) { _, isEmpty in
                if isEmpty {
                    selectedNoteIDs.removeAll()
                }
            }
        }
        .onChange(of: notes.map(\.id)) { _, newIDs in
            selectedNoteIDs.formIntersection(Set(newIDs))
        }
    }

    private var emptyState: some View {
        ZStack {
            Color(white: 0.1)

            VStack(spacing: 8) {
                Image(systemName: "waveform.path")
                    .font(.title3)
                    .foregroundStyle(.secondary)

                Text("No notes detected")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)

                Text("Adjust cleanup sliders to reveal notes.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 20)
        }
    }

    private var cornerCell: some View {
        Rectangle()
            .fill(Color(white: 0.14))
            .overlay(alignment: .center) {
                Text("Pitch")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .frame(width: labelColumnWidth, height: rulerHeight)
    }

    private var timeRuler: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color(white: 0.14))

            ForEach(timelineSeconds, id: \.self) { second in
                let x = xPosition(for: Double(second))

                Path { path in
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: rulerHeight))
                }
                .stroke(Color.white.opacity(0.18), lineWidth: 1)

                Text("\(second)s")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .offset(x: x + 3, y: 3)
            }

            Rectangle()
                .fill(Color.white.opacity(0.2))
                .frame(height: 1)
                .frame(maxHeight: .infinity, alignment: .bottom)

            Path { path in
                let x = xPosition(for: playbackTime)
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: rulerHeight))
            }
            .stroke(Color.yellow, lineWidth: 2)
        }
        .contentShape(Rectangle())
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onEnded { value in
                    let isTap = abs(value.translation.width) < 8 && abs(value.translation.height) < 8
                    guard isTap else { return }
                    seekPlayback(toXPosition: value.location.x)
                }
        )
    }

    private var pitchLabels: some View {
        VStack(spacing: 0) {
            ForEach(visiblePitches, id: \.self) { pitch in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(backgroundColor(forPitch: pitch))

                    Rectangle()
                        .fill(Color.white.opacity(0.08))
                        .frame(height: 0.5)
                        .frame(maxHeight: .infinity, alignment: .bottom)

                    Text(noteName(forMIDIPitch: pitch))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(Color.white.opacity(0.72))
                        .lineLimit(1)
                        .padding(.leading, 4)
                }
                .frame(height: rowHeight)
            }
        }
    }

    private var notesCanvas: some View {
        Canvas { context, size in
            let fullRect = CGRect(origin: .zero, size: size)
            context.fill(Path(fullRect), with: .color(Color(white: 0.1)))

            for (row, pitch) in visiblePitches.enumerated() {
                let y = CGFloat(row) * rowHeight
                let rowRect = CGRect(x: 0, y: y, width: size.width, height: rowHeight)

                context.fill(Path(rowRect), with: .color(backgroundColor(forPitch: pitch)))

                var horizontalLine = Path()
                horizontalLine.move(to: CGPoint(x: 0, y: y + rowHeight))
                horizontalLine.addLine(to: CGPoint(x: size.width, y: y + rowHeight))
                context.stroke(horizontalLine, with: .color(Color.white.opacity(0.08)), lineWidth: 0.5)
            }

            for second in timelineSeconds {
                let x = xPosition(for: Double(second))
                var verticalLine = Path()
                verticalLine.move(to: CGPoint(x: x, y: 0))
                verticalLine.addLine(to: CGPoint(x: x, y: size.height))
                context.stroke(verticalLine, with: .color(Color.white.opacity(0.12)), lineWidth: 1)
            }

            for note in notes {
                guard let rect = noteRect(for: note) else { continue }
                let roundedRect = RoundedRectangle(cornerRadius: 3, style: .continuous)

                // Velocity is read directly from the NoteEvent to drive visual intensity.
                let velocityRatio = Double(note.velocity) / 127.0
                let noteColor = Color(
                    hue: 0.62 - (0.16 * velocityRatio),
                    saturation: 0.82,
                    brightness: 0.35 + (0.55 * velocityRatio)
                )

                context.fill(roundedRect.path(in: rect), with: .color(noteColor))

                if selectedNoteIDs.contains(note.id) {
                    context.stroke(
                        roundedRect.path(in: rect.insetBy(dx: -1, dy: -1)),
                        with: .color(Color.white),
                        lineWidth: 2
                    )
                }
            }

            var scrubber = Path()
            let x = xPosition(for: playbackTime)
            scrubber.move(to: CGPoint(x: x, y: 0))
            scrubber.addLine(to: CGPoint(x: x, y: size.height))
            context.stroke(scrubber, with: .color(.yellow), lineWidth: 2)
        }
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onEnded { value in
                    let isTap = abs(value.translation.width) < 8 && abs(value.translation.height) < 8
                    guard isTap else { return }
                    handleTap(at: value.location)
                }
        )
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                if zoomStartPixelsPerSecond == nil {
                    zoomStartPixelsPerSecond = pixelsPerSecond
                }
                let base = zoomStartPixelsPerSecond ?? pixelsPerSecond
                pixelsPerSecond = min(max(base * value, minPixelsPerSecond), maxPixelsPerSecond)
            }
            .onEnded { _ in
                zoomStartPixelsPerSecond = nil
            }
    }

    private func handleTap(at point: CGPoint) {
        guard let hitNote = note(at: point) else {
            selectedNoteIDs.removeAll()
            seekPlayback(toXPosition: point.x)
            return
        }

        if selectedNoteIDs.contains(hitNote.id) {
            selectedNoteIDs.remove(hitNote.id)
        } else {
            selectedNoteIDs.insert(hitNote.id)
        }
    }

    private func note(at point: CGPoint) -> NoteEvent? {
        guard point.x >= 0, point.y >= 0 else { return nil }

        let rowFromTop = Int(point.y / rowHeight)
        guard rowFromTop >= 0, rowFromTop < visiblePitches.count else { return nil }

        let tappedPitch = visiblePitches[rowFromTop]
        let tappedTime = timelineStart + Double(point.x / pixelsPerSecond)
        let timeTolerance = Double(6.0 / pixelsPerSecond)

        let candidates = notes.filter { note in
            Int(note.pitch) == tappedPitch &&
            tappedTime >= (note.startTime - timeTolerance) &&
            tappedTime <= (note.startTime + note.duration + timeTolerance)
        }

        return candidates.min(by: { $0.duration < $1.duration })
    }

    private func seekPlayback(toXPosition x: CGFloat) {
        guard let onSeekRequested else { return }
        let unclamped = timelineStart + Double(x / pixelsPerSecond)
        let clamped = min(max(unclamped, 0), max(timelineEnd, 0))
        onSeekRequested(clamped)
    }

    private func deleteSelectedNotes() {
        notes.removeAll { selectedNoteIDs.contains($0.id) }
        selectedNoteIDs.removeAll()
    }

    private func noteRect(for note: NoteEvent) -> CGRect? {
        guard let rowFromTop = visiblePitches.firstIndex(of: Int(note.pitch)) else {
            return nil
        }

        let x = xPosition(for: note.startTime)
        let y = CGFloat(rowFromTop) * rowHeight + 1
        let width = max(CGFloat(note.duration) * pixelsPerSecond, 2)
        let height = max(rowHeight - 2, 2)

        return CGRect(x: x, y: y, width: width, height: height)
    }

    private func attemptAutoFitIfNeeded() {
        guard pendingAutoFit else { return }
        guard availableGridWidth > 1 else { return }

        if notes.isEmpty {
            pixelsPerSecond = 100
            timelineStart = 0
            timelineEnd = 1
            pendingAutoFit = false
            return
        }

        let firstNoteStart = notes.map(\.startTime).min() ?? 0
        let lastNoteEnd = notes.map { $0.startTime + $0.duration }.max() ?? firstNoteStart
        let noteSpan = max(lastNoteEnd - firstNoteStart, 0.01)
        let horizontalMargin = max(noteSpan * 0.05, 0.05)
        let paddedSpan = noteSpan + (horizontalMargin * 2)

        let fittedPixelsPerSecond = availableGridWidth / CGFloat(paddedSpan)
        pixelsPerSecond = min(max(fittedPixelsPerSecond, minPixelsPerSecond), maxPixelsPerSecond)
        timelineStart = max(0, firstNoteStart - horizontalMargin)
        timelineEnd = max(timelineStart + 0.1, lastNoteEnd + horizontalMargin)
        pendingAutoFit = false
    }

    private func xPosition(for time: Double) -> CGFloat {
        CGFloat(time - timelineStart) * pixelsPerSecond
    }

    private func backgroundColor(forPitch pitch: Int) -> Color {
        let pitchClass = pitch % 12
        return naturalPitchClasses.contains(pitchClass) ? Color(white: 0.16) : Color(white: 0.12)
    }

    private func noteName(forMIDIPitch pitch: Int) -> String {
        let name = noteNames[pitch % 12]
        let octave = (pitch / 12) - 1
        return "\(name)\(octave)"
    }

    private var naturalPitchClasses: Set<Int> {
        [0, 2, 4, 5, 7, 9, 11]
    }

    private var noteNames: [String] {
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    }
}

private struct PianoRollScrollOffsetKey: PreferenceKey {
    static var defaultValue: CGPoint = .zero

    static func reduce(value: inout CGPoint, nextValue: () -> CGPoint) {
        value = nextValue()
    }
}
