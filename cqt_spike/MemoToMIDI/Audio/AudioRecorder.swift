import AVFoundation
import Foundation

final class AudioRecorder: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var isRecording = false
    @Published private(set) var isPlaying = false
    @Published private(set) var lastRecordingURL: URL?
    @Published private(set) var recordingDuration: TimeInterval = 0
    @Published private(set) var waveformSamples: [Float] = []
    @Published var errorMessage: String?

    private let audioEngine = AVAudioEngine()
    private let stateQueue = DispatchQueue(label: "MemoToMIDI.AudioRecorder.StateQueue")
    private var outputFile: AVAudioFile?
    private var converter: AVAudioConverter?
    private var destinationFormat: AVAudioFormat?
    private var writtenFrameCount: Int64 = 0
    private var player: AVAudioPlayer?

    func requestMicrophoneAccess() async -> Bool {
        let session = AVAudioSession.sharedInstance()
        switch session.recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            return await withCheckedContinuation { continuation in
                session.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        @unknown default:
            return false
        }
    }

    func startRecording() throws {
        guard !isRecording else { return }
        guard AVAudioSession.sharedInstance().recordPermission == .granted else {
            throw AudioRecorderError.microphonePermissionDenied
        }

        try configureSession()
        stopPlayback()

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioConstants.sampleRate,
            channels: AudioConstants.channelCount,
            interleaved: false
        ) else {
            throw AudioRecorderError.unableToCreateOutputFormat
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioRecorderError.unableToCreateConverter
        }

        let outputURL = try Self.makeRecordingURL()
        let file = try AVAudioFile(
            forWriting: outputURL,
            settings: outputFormat.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        stateQueue.sync {
            self.outputFile = file
            self.converter = converter
            self.destinationFormat = outputFormat
            self.writtenFrameCount = 0
        }

        DispatchQueue.main.async {
            self.errorMessage = nil
            self.recordingDuration = 0
            self.waveformSamples = []
            self.lastRecordingURL = outputURL
        }

        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: AudioConstants.tapBufferSize, format: inputFormat) { [weak self] buffer, _ in
            self?.handleInputBuffer(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            DispatchQueue.main.async {
                self.isRecording = true
            }
        } catch {
            cleanupAfterStop()
            throw error
        }
    }

    func stopRecording() {
        guard isRecording else { return }
        cleanupAfterStop()
        DispatchQueue.main.async {
            self.isRecording = false
        }
    }

    func playLastRecording() throws {
        guard let url = lastRecordingURL else {
            throw AudioRecorderError.missingRecording
        }

        stopPlayback()
        try configureSession()

        let player = try AVAudioPlayer(contentsOf: url)
        player.delegate = self
        player.prepareToPlay()
        guard player.play() else {
            throw AudioRecorderError.playbackFailed
        }
        self.player = player
        DispatchQueue.main.async {
            self.isPlaying = true
        }
    }

    func stopPlayback() {
        guard let player else { return }
        player.stop()
        self.player = nil
        DispatchQueue.main.async {
            self.isPlaying = false
        }
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async {
            self.isPlaying = false
        }
    }

    private func cleanupAfterStop() {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()

        stateQueue.sync {
            self.outputFile = nil
            self.converter = nil
            self.destinationFormat = nil
        }

        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = error.localizedDescription
            }
        }
    }

    private func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setPreferredSampleRate(AudioConstants.sampleRate)
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    private func handleInputBuffer(_ buffer: AVAudioPCMBuffer) {
        stateQueue.sync {
            guard let converter, let outputFile, let destinationFormat else { return }

            let resampleRatio = destinationFormat.sampleRate / buffer.format.sampleRate
            let estimatedFrameCount = Int(Double(buffer.frameLength) * resampleRatio) + 64
            guard estimatedFrameCount > 0 else { return }
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: destinationFormat,
                frameCapacity: AVAudioFrameCount(estimatedFrameCount)
            ) else {
                return
            }

            var conversionError: NSError?
            var suppliedInput = false
            let status = converter.convert(to: convertedBuffer, error: &conversionError) { _, outStatus in
                if suppliedInput {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                suppliedInput = true
                outStatus.pointee = .haveData
                return buffer
            }

            if let conversionError {
                DispatchQueue.main.async {
                    self.errorMessage = conversionError.localizedDescription
                }
                return
            }

            if status == .error || convertedBuffer.frameLength == 0 {
                return
            }

            do {
                try outputFile.write(from: convertedBuffer)
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                }
                return
            }

            self.writtenFrameCount += Int64(convertedBuffer.frameLength)
            let durationSeconds = Double(self.writtenFrameCount) / AudioConstants.sampleRate
            let waveformLevel = Self.rmsLevel(from: convertedBuffer)

            DispatchQueue.main.async {
                self.recordingDuration = durationSeconds
                self.appendWaveformSample(waveformLevel)
            }
        }
    }

    private func appendWaveformSample(_ sample: Float) {
        waveformSamples.append(sample)
        if waveformSamples.count > AudioConstants.maxWaveformSamples {
            waveformSamples.removeFirst(waveformSamples.count - AudioConstants.maxWaveformSamples)
        }
    }

    private static func rmsLevel(from buffer: AVAudioPCMBuffer) -> Float {
        guard
            buffer.frameLength > 0,
            let channels = buffer.floatChannelData
        else {
            return 0
        }

        let samples = UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength))
        let sumSquares = samples.reduce(Float(0)) { $0 + ($1 * $1) }
        let rms = sqrt(sumSquares / Float(buffer.frameLength))
        return min(1, rms * 7)
    }

    private static func makeRecordingURL() throws -> URL {
        let docsURL = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let timestamp = recordingTimestampFormatter.string(from: Date())
        return docsURL.appendingPathComponent("MemoToMIDI_\(timestamp).wav")
    }

    private static let recordingTimestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
}

extension AudioRecorder {
    enum AudioRecorderError: LocalizedError {
        case microphonePermissionDenied
        case unableToCreateOutputFormat
        case unableToCreateConverter
        case missingRecording
        case playbackFailed

        var errorDescription: String? {
            switch self {
            case .microphonePermissionDenied:
                return "Microphone permission is required to record audio."
            case .unableToCreateOutputFormat:
                return "Unable to create the 22,050 Hz mono recording format."
            case .unableToCreateConverter:
                return "Unable to create an audio converter for the input format."
            case .missingRecording:
                return "No recording available yet."
            case .playbackFailed:
                return "Playback could not be started."
            }
        }
    }
}
