import CoreML
import Foundation

/// Result of ML inference over an entire recording.
struct InferenceResult {
    let noteActivations: [[Float]]   // [totalFrames][88]
    let onsetActivations: [[Float]]  // [totalFrames][88]
    let frameDuration: Double        // seconds per frame (256.0 / 22050.0)
}

final class BasicPitchInference {
    private let inferenceQueue = DispatchQueue(label: "MemoToMIDI.BasicPitchInference", qos: .userInitiated)
    private let modelConfiguration: MLModelConfiguration

    private var model: BasicPitch_from_tf?
    private var didWarmUp = false

    init(computeUnits: MLComputeUnits = .cpuAndGPU) {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.modelConfiguration = configuration
    }

    func warmUp() async throws {
        try await runOnInferenceQueue {
            try self.warmUpSync()
        }
    }

    /// Process a full audio buffer through the model.
    /// - Parameter audioSamples: Float32 array at 22050 Hz mono
    /// - Parameter progressCallback: Called with 0.0–1.0 progress for UI updates
    /// - Returns: Aggregated note and onset matrices for the full recording
    func process(
        audioSamples: [Float],
        progressCallback: ((Double) -> Void)? = nil
    ) async throws -> InferenceResult {
        try await runOnInferenceQueue {
            try self.processSync(audioSamples: audioSamples, progressCallback: progressCallback)
        }
    }

    private func processSync(
        audioSamples: [Float],
        progressCallback: ((Double) -> Void)?
    ) throws -> InferenceResult {
        guard !audioSamples.isEmpty else {
            throw BasicPitchInferenceError.emptyAudio
        }

        reportProgress(0, callback: progressCallback)
        try warmUpSync()
        let loadedModel = try loadModelIfNeeded()

        var paddedSamples = [Float](repeating: 0, count: AudioConstants.prependSamples)
        paddedSamples.append(contentsOf: audioSamples)

        let totalWindows = max(1, Int(ceil(Double(paddedSamples.count) / Double(AudioConstants.hopSamples))))
        let overlapFrames = AudioConstants.overlapLen / AudioConstants.modelHopSize
        let halfOverlapFrames = overlapFrames / 2

        var noteActivations: [[Float]] = []
        var onsetActivations: [[Float]] = []

        for windowIndex in 0..<totalWindows {
            let startSample = windowIndex * AudioConstants.hopSamples
            var windowSamples = [Float](repeating: 0, count: AudioConstants.windowSamples)
            let availableSamples = max(0, paddedSamples.count - startSample)
            let copyCount = min(AudioConstants.windowSamples, availableSamples)

            if copyCount > 0 {
                for sampleIndex in 0..<copyCount {
                    windowSamples[sampleIndex] = paddedSamples[startSample + sampleIndex]
                }
            }

            let (windowNote, windowOnset) = try predictWindow(windowSamples: &windowSamples, model: loadedModel)
            guard windowNote.count == windowOnset.count else {
                throw BasicPitchInferenceError.invalidOutputShape("note/onset frame count mismatch")
            }

            let windowFrameCount = windowNote.count
            let keepStart = windowIndex == 0 ? 0 : halfOverlapFrames
            let keepEnd = windowIndex == totalWindows - 1
                ? windowFrameCount
                : max(keepStart, windowFrameCount - halfOverlapFrames)

            for frameIndex in keepStart..<keepEnd {
                noteActivations.append(windowNote[frameIndex])
                onsetActivations.append(windowOnset[frameIndex])
            }

            reportProgress(Double(windowIndex + 1) / Double(totalWindows), callback: progressCallback)
        }

        let expectedFrameCount = max(
            1,
            Int(Double(audioSamples.count) / Double(AudioConstants.modelHopSize))
        )

        if noteActivations.count > expectedFrameCount {
            noteActivations.removeSubrange(expectedFrameCount..<noteActivations.count)
        }
        if onsetActivations.count > expectedFrameCount {
            onsetActivations.removeSubrange(expectedFrameCount..<onsetActivations.count)
        }

        guard noteActivations.count == onsetActivations.count else {
            throw BasicPitchInferenceError.invalidOutputShape("final note/onset frame counts differ")
        }

        let frameDuration = Double(AudioConstants.modelHopSize) / AudioConstants.sampleRate
        return InferenceResult(
            noteActivations: noteActivations,
            onsetActivations: onsetActivations,
            frameDuration: frameDuration
        )
    }

    private func warmUpSync() throws {
        guard !didWarmUp else { return }
        let loadedModel = try loadModelIfNeeded()
        var zeros = [Float](repeating: 0, count: AudioConstants.windowSamples)
        _ = try predictWindow(windowSamples: &zeros, model: loadedModel)
        didWarmUp = true
    }

    private func loadModelIfNeeded() throws -> BasicPitch_from_tf {
        if let model {
            return model
        }

        do {
            let loadedModel = try BasicPitch_from_tf(configuration: modelConfiguration)
            self.model = loadedModel
            return loadedModel
        } catch {
            throw BasicPitchInferenceError.modelLoadFailed(error.localizedDescription)
        }
    }

    private func predictWindow(
        windowSamples: inout [Float],
        model: BasicPitch_from_tf
    ) throws -> (note: [[Float]], onset: [[Float]]) {
        // Generated CoreML interface confirms:
        // input name: `input_2`; note output: `Identity_1`; onset output: `Identity_2`.
        return try windowSamples.withUnsafeMutableBufferPointer { bufferPointer in
            guard let baseAddress = bufferPointer.baseAddress else {
                throw BasicPitchInferenceError.invalidInput("Window has no base address")
            }

            let inputArray = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(baseAddress),
                shape: [1, NSNumber(value: AudioConstants.windowSamples), 1],
                dataType: .float32,
                strides: [NSNumber(value: AudioConstants.windowSamples), 1, 1],
                deallocator: nil
            )

            do {
                let output = try model.prediction(input_2: inputArray)
                let note = try Self.unpackModelOutput(output.Identity_1, expectedFrequencyBins: 88)
                let onset = try Self.unpackModelOutput(output.Identity_2, expectedFrequencyBins: 88)
                return (note, onset)
            } catch {
                throw BasicPitchInferenceError.predictionFailed(error.localizedDescription)
            }
        }
    }

    private static func unpackModelOutput(
        _ output: MLMultiArray,
        expectedFrequencyBins: Int
    ) throws -> [[Float]] {
        let shape = output.shape.map { Int(truncating: $0) }
        let strides = output.strides.map { Int(truncating: $0) }

        guard shape.count == 3, strides.count == 3 else {
            throw BasicPitchInferenceError.invalidOutputShape("Expected rank-3 output, got shape \(shape)")
        }
        guard shape[0] >= 1 else {
            throw BasicPitchInferenceError.invalidOutputShape("Empty batch dimension in output")
        }
        guard shape[2] == expectedFrequencyBins else {
            throw BasicPitchInferenceError.invalidOutputShape(
                "Expected \(expectedFrequencyBins) bins, got \(shape[2])"
            )
        }

        let frameCount = shape[1]
        let frequencyBins = shape[2]
        var frames: [[Float]] = []
        frames.reserveCapacity(frameCount)

        switch output.dataType {
        case .float32:
            let values = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
            for frameIndex in 0..<frameCount {
                var frame = [Float](repeating: 0, count: frequencyBins)
                for frequencyIndex in 0..<frequencyBins {
                    let offset = frameIndex * strides[1] + frequencyIndex * strides[2]
                    frame[frequencyIndex] = values[offset]
                }
                frames.append(frame)
            }

        case .double:
            let values = output.dataPointer.bindMemory(to: Double.self, capacity: output.count)
            for frameIndex in 0..<frameCount {
                var frame = [Float](repeating: 0, count: frequencyBins)
                for frequencyIndex in 0..<frequencyBins {
                    let offset = frameIndex * strides[1] + frequencyIndex * strides[2]
                    frame[frequencyIndex] = Float(values[offset])
                }
                frames.append(frame)
            }

        default:
            throw BasicPitchInferenceError.invalidOutputShape(
                "Unsupported MLMultiArray dtype: \(output.dataType.rawValue)"
            )
        }

        return frames
    }

    private func reportProgress(_ progress: Double, callback: ((Double) -> Void)?) {
        guard let callback else { return }
        let clamped = min(max(progress, 0), 1)
        DispatchQueue.main.async {
            callback(clamped)
        }
    }

    private func runOnInferenceQueue<T>(_ block: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            inferenceQueue.async {
                do {
                    continuation.resume(returning: try block())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

enum BasicPitchInferenceError: LocalizedError {
    case emptyAudio
    case invalidInput(String)
    case modelLoadFailed(String)
    case predictionFailed(String)
    case invalidOutputShape(String)

    var errorDescription: String? {
        switch self {
        case .emptyAudio:
            return "Audio buffer is empty."
        case .invalidInput(let message):
            return "Invalid model input: \(message)"
        case .modelLoadFailed(let message):
            return "Failed to load BasicPitch model: \(message)"
        case .predictionFailed(let message):
            return "CoreML prediction failed: \(message)"
        case .invalidOutputShape(let message):
            return "Unexpected model output shape: \(message)"
        }
    }
}
