import AVFoundation
import Foundation

enum AudioConstants {
    static let sampleRate: Double = 22_050
    static let channelCount: AVAudioChannelCount = 1
    static let tapBufferSize: AVAudioFrameCount = 2_048
    static let maxWaveformSamples: Int = 220

    // Basic Pitch model windowing constants (validated in the CQT spike).
    static let windowSamples: Int = 43_844
    static let hopSamples: Int = 36_164
    static let overlapLen: Int = 7_680
    static let prependSamples: Int = 3_840
    static let modelHopSize: Int = 256
    static let modelFFTStepSamples: Int = modelHopSize
}

enum MIDIConstants {
    static let ppqn: UInt16 = 480
    static let defaultTempo: Double = 120.0
    static let defaultOnsetThreshold: Float = 0.45
    static let defaultFrameThreshold: Float = 0.3
    static let minNoteLengthMs: Double = 58.0
    static let guitarRangeLow: UInt8 = 40
    static let guitarRangeHigh: UInt8 = 84
}
