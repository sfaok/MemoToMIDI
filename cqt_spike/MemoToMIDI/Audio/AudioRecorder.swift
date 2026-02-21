import AVFoundation
import Foundation

final class AudioRecorder: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var isRecording = false
    @Published private(set) var isPlaying = false
    @Published private(set) var lastRecordingURL: URL?
    @Published private(set) var recordingDuration: TimeInterval = 0
    @Published private(set) var currentPlaybackTime: Double = 0
    @Published private(set) var playbackDuration: Double = 0
    @Published private(set) var waveformSamples: [Float] = []
    @Published var errorMessage: String?

    private let audioEngine = AVAudioEngine()
    private let stateQueue = DispatchQueue(label: "MemoToMIDI.AudioRecorder.StateQueue")
    private var outputFile: AVAudioFile?
    private var converter: AVAudioConverter?
    private var destinationFormat: AVAudioFormat?
    private var writtenFrameCount: Int64 = 0
    private var player: AVAudioPlayer?
    private var playerURL: URL?
    private var playbackTimer: Timer?
    private var playbackVolume: Float = 1.0

    func requestMicrophoneAccess() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            return await withCheckedContinuation { continuation in
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        @unknown default:
            return false
        }
    }

    func startRecording() throws {
        guard !isRecording else { return }
        guard AVAudioApplication.shared.recordPermission == .granted else {
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
        try playLastRecording(startTime: nil, startDelay: 0)
    }

    func playLastRecording(startTime: Double?, startDelay: TimeInterval = 0) throws {
        guard let url = lastRecordingURL else {
            throw AudioRecorderError.missingRecording
        }

        try configureSession()

        let player = try preparePlayer(for: url)
        let seekTime = startTime ?? player.currentTime
        let clampedStartTime = min(max(seekTime, 0), player.duration)

        if clampedStartTime >= player.duration {
            player.currentTime = 0
        } else {
            player.currentTime = clampedStartTime
        }

        let didStart: Bool
        if startDelay > 0 {
            didStart = player.play(atTime: player.deviceCurrentTime + startDelay)
        } else {
            didStart = player.play()
        }

        guard didStart else {
            throw AudioRecorderError.playbackFailed
        }

        startPlaybackTimer()
        DispatchQueue.main.async {
            self.isPlaying = true
            self.currentPlaybackTime = player.currentTime
        }
    }

    func pausePlayback() {
        guard let player, player.isPlaying else { return }
        player.pause()
        stopPlaybackTimer()
        DispatchQueue.main.async {
            self.isPlaying = false
            self.currentPlaybackTime = player.currentTime
        }
    }

    func seekPlayback(to time: Double) {
        guard let url = lastRecordingURL else { return }

        do {
            let player = try preparePlayer(for: url)
            player.currentTime = min(max(time, 0), player.duration)
            DispatchQueue.main.async {
                self.currentPlaybackTime = player.currentTime
                self.playbackDuration = player.duration
            }
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = error.localizedDescription
            }
        }
    }

    func setPlaybackVolume(_ volume: Float) {
        let clamped = min(max(volume, 0), 1)
        playbackVolume = clamped
        player?.volume = clamped
    }

    func stopPlayback() {
        stopPlaybackTimer()

        guard let player else {
            DispatchQueue.main.async {
                self.isPlaying = false
                self.currentPlaybackTime = 0
                self.playbackDuration = 0
            }
            return
        }

        player.stop()
        player.currentTime = 0
        self.player = nil
        self.playerURL = nil
        DispatchQueue.main.async {
            self.isPlaying = false
            self.currentPlaybackTime = 0
            self.playbackDuration = 0
        }
    }

    func playbackTimeSnapshot() -> Double {
        player?.currentTime ?? currentPlaybackTime
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        stopPlaybackTimer()
        DispatchQueue.main.async {
            self.isPlaying = false
            self.currentPlaybackTime = player.currentTime
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
        try session.setCategory(
            .playAndRecord,
            mode: .default,
            options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers]
        )
        try session.setPreferredSampleRate(AudioConstants.sampleRate)
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    private func preparePlayer(for url: URL) throws -> AVAudioPlayer {
        if let existingPlayer = player, playerURL == url {
            existingPlayer.volume = playbackVolume
            return existingPlayer
        }

        let createdPlayer = try AVAudioPlayer(contentsOf: url)
        createdPlayer.delegate = self
        createdPlayer.volume = playbackVolume
        createdPlayer.prepareToPlay()

        player = createdPlayer
        playerURL = url

        DispatchQueue.main.async {
            self.playbackDuration = createdPlayer.duration
        }

        return createdPlayer
    }

    private func startPlaybackTimer() {
        stopPlaybackTimer()
        let timer = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self, let player else { return }
            DispatchQueue.main.async {
                self.currentPlaybackTime = player.currentTime
            }
        }
        playbackTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    private func stopPlaybackTimer() {
        playbackTimer?.invalidate()
        playbackTimer = nil
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
