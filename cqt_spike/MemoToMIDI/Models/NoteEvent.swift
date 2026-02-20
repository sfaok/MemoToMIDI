import Foundation

struct NoteEvent: Identifiable, Codable {
    let id: UUID
    var pitch: UInt8          // MIDI note number (0-127)
    var startTime: Double     // Seconds from recording start
    var duration: Double      // Seconds
    var velocity: UInt8       // 0-127

    init(id: UUID = UUID(), pitch: UInt8, startTime: Double, duration: Double, velocity: UInt8) {
        self.id = id
        self.pitch = pitch
        self.startTime = startTime
        self.duration = duration
        self.velocity = velocity
    }
}
