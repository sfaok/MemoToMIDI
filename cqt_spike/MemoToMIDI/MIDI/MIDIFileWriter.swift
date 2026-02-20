import Foundation

struct MIDIFileWriter {
    private struct MIDIEvent {
        let absoluteTick: Int
        let status: UInt8
        let pitch: UInt8
        let velocity: UInt8
        let isNoteOff: Bool
    }

    /// Generate a complete .mid file as raw bytes.
    /// - Parameters:
    ///   - notes: The note events to encode.
    ///   - bpm: Tempo in beats per minute.
    /// - Returns: Data containing a valid Standard MIDI File.
    static func write(notes: [NoteEvent], bpm: Double) -> Data {
        let safeBPM = bpm > 0 ? bpm : MIDIConstants.defaultTempo

        var trackData = Data()

        // Tempo meta event at tick 0.
        trackData.appendVLQ(0)
        trackData.append(bytes: [0xFF, 0x51, 0x03])
        let microsPerQuarter = UInt32((60_000_000.0 / safeBPM).rounded())
        trackData.appendTempoMicrosPerQuarter(microsPerQuarter)

        // Encode sorted note events.
        let events = buildEvents(from: notes, bpm: safeBPM)
        var previousTick = 0

        for event in events {
            let delta = max(0, event.absoluteTick - previousTick)
            trackData.appendVLQ(delta)
            trackData.append(bytes: [event.status, event.pitch, event.velocity])
            previousTick = event.absoluteTick
        }

        // End-of-track meta event.
        trackData.appendVLQ(0)
        trackData.append(bytes: [0xFF, 0x2F, 0x00])

        // SMF Type 0 file.
        var midiData = Data()
        midiData.appendASCII("MThd")
        midiData.appendUInt32BE(6)
        midiData.appendUInt16BE(0) // format 0
        midiData.appendUInt16BE(1) // one track
        midiData.appendUInt16BE(MIDIConstants.ppqn)

        midiData.appendASCII("MTrk")
        midiData.appendUInt32BE(UInt32(trackData.count))
        midiData.append(trackData)

        return midiData
    }

    #if DEBUG
    static func vlqSelfTest() -> [(value: Int, encoded: [UInt8], expected: [UInt8], isMatch: Bool)] {
        let vectors: [(Int, [UInt8])] = [
            (0, [0x00]),
            (127, [0x7F]),
            (128, [0x81, 0x00]),
            (480, [0x83, 0x60]),
            (960, [0x87, 0x40]),
            (16_383, [0xFF, 0x7F])
        ]

        return vectors.map { value, expected in
            let encoded = encodeVLQ(value)
            return (value: value, encoded: encoded, expected: expected, isMatch: encoded == expected)
        }
    }
    #endif

    private static func buildEvents(from notes: [NoteEvent], bpm: Double) -> [MIDIEvent] {
        var events: [MIDIEvent] = []
        events.reserveCapacity(notes.count * 2)

        for note in notes {
            guard note.duration > 0 else { continue }

            let startTick = secondsToTicks(note.startTime, bpm: bpm)
            let endTick = secondsToTicks(note.startTime + note.duration, bpm: bpm)
            guard endTick > startTick else { continue }

            let clampedVelocity = max(note.velocity, 1)

            events.append(
                MIDIEvent(
                    absoluteTick: startTick,
                    status: 0x90,
                    pitch: note.pitch,
                    velocity: clampedVelocity,
                    isNoteOff: false
                )
            )

            events.append(
                MIDIEvent(
                    absoluteTick: endTick,
                    status: 0x80,
                    pitch: note.pitch,
                    velocity: 0x40,
                    isNoteOff: true
                )
            )
        }

        events.sort { lhs, rhs in
            if lhs.absoluteTick != rhs.absoluteTick {
                return lhs.absoluteTick < rhs.absoluteTick
            }
            if lhs.isNoteOff != rhs.isNoteOff {
                return lhs.isNoteOff && !rhs.isNoteOff
            }
            return lhs.pitch < rhs.pitch
        }

        return events
    }

    private static func secondsToTicks(_ seconds: Double, bpm: Double) -> Int {
        guard seconds.isFinite else { return 0 }
        let clampedSeconds = max(0, seconds)
        let ticks = clampedSeconds * (bpm / 60.0) * Double(MIDIConstants.ppqn)
        guard ticks.isFinite else { return 0 }
        let rounded = Int(ticks.rounded())
        return min(max(0, rounded), 0x0FFF_FFFF)
    }

    fileprivate static func encodeVLQ(_ value: Int) -> [UInt8] {
        var workingValue = min(max(0, value), 0x0FFF_FFFF)
        var bytes: [UInt8] = [UInt8(workingValue & 0x7F)]
        workingValue >>= 7

        while workingValue > 0 {
            let next = UInt8(workingValue & 0x7F) | 0x80
            bytes.insert(next, at: 0)
            workingValue >>= 7
        }

        return bytes
    }
}

private extension Data {
    mutating func appendASCII(_ string: String) {
        append(string.data(using: .ascii) ?? Data())
    }

    mutating func appendUInt16BE(_ value: UInt16) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { rawBuffer in
            append(rawBuffer.bindMemory(to: UInt8.self))
        }
    }

    mutating func appendUInt32BE(_ value: UInt32) {
        var be = value.bigEndian
        Swift.withUnsafeBytes(of: &be) { rawBuffer in
            append(rawBuffer.bindMemory(to: UInt8.self))
        }
    }

    mutating func appendTempoMicrosPerQuarter(_ value: UInt32) {
        let clamped = Swift.min(value, 0x00FF_FFFF)
        let b1 = UInt8((clamped >> 16) & 0xFF)
        let b2 = UInt8((clamped >> 8) & 0xFF)
        let b3 = UInt8(clamped & 0xFF)
        append(bytes: [b1, b2, b3])
    }

    mutating func appendVLQ(_ value: Int) {
        append(bytes: MIDIFileWriter.encodeVLQ(value))
    }

    mutating func append(bytes: [UInt8]) {
        append(contentsOf: bytes)
    }
}
