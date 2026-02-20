import AVFoundation
import Foundation

enum AudioConstants {
    static let sampleRate: Double = 22_050
    static let channelCount: AVAudioChannelCount = 1
    static let tapBufferSize: AVAudioFrameCount = 2_048
    static let maxWaveformSamples: Int = 220
}
