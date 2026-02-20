import AVFoundation
import Foundation

struct AudioFileDescription {
    let sampleRate: Double
    let channelCount: AVAudioChannelCount
    let duration: TimeInterval
    let frameCount: AVAudioFramePosition
}

enum AudioFileReader {
    static func describeFile(at url: URL) throws -> AudioFileDescription {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = file.length
        let duration = Double(frameCount) / format.sampleRate
        return AudioFileDescription(
            sampleRate: format.sampleRate,
            channelCount: format.channelCount,
            duration: duration,
            frameCount: frameCount
        )
    }

    static func readMonoFloat32(
        from url: URL,
        targetSampleRate: Double = AudioConstants.sampleRate
    ) throws -> [Float] {
        let inputFile = try AVAudioFile(forReading: url)
        let inputFormat = inputFile.processingFormat
        let frameCapacity = AVAudioFrameCount(inputFile.length)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCapacity) else {
            return []
        }
        try inputFile.read(into: inputBuffer)

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: AudioConstants.channelCount,
            interleaved: false
        ) else {
            return []
        }

        if inputFormat.sampleRate == outputFormat.sampleRate,
           inputFormat.channelCount == outputFormat.channelCount,
           inputFormat.commonFormat == .pcmFormatFloat32,
           let channelData = inputBuffer.floatChannelData?.pointee {
            return Array(UnsafeBufferPointer(start: channelData, count: Int(inputBuffer.frameLength)))
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            return []
        }

        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let estimatedOutputFrames = Int(Double(inputBuffer.frameLength) * ratio) + 64
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(max(estimatedOutputFrames, 1))
        ) else {
            return []
        }

        var conversionError: NSError?
        var providedInput = false
        _ = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if providedInput {
                outStatus.pointee = .endOfStream
                return nil
            }
            providedInput = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if conversionError != nil {
            return []
        }

        guard let outputData = outputBuffer.floatChannelData?.pointee else {
            return []
        }

        return Array(UnsafeBufferPointer(start: outputData, count: Int(outputBuffer.frameLength)))
    }
}
