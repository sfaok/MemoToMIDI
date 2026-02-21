import AVFoundation
import Accelerate
import Combine
import Foundation

final class MIDIPlayer: ObservableObject {
    @Published var isPlaying: Bool = false
    @Published var currentTime: Double = 0.0
    @Published var isPreparingOverlay: Bool = false
    @Published var overlayPreparationError: String?

    private struct ScheduledNote {
        let id: UUID
        let pitch: UInt8
        let velocity: UInt8
        let startTime: Double
        let endTime: Double
    }

    private struct StopEvent {
        let pitch: UInt8
        let endTime: Double
    }

    private struct OverlayRenderEvent {
        let samplePosition: AVAudioFramePosition
        let pitch: UInt8
        let velocity: UInt8
        let isNoteOn: Bool
    }

    private enum TransportMode {
        case idle
        case midiOnlyPreview
        case overlayRendered
    }

    private enum MIDIPlayerError: LocalizedError {
        case engineNotRunning
        case unableToAllocateOverlayBuffer
        case unableToAllocateRenderBuffer
        case overlayRecordingMissing
        case unsupportedRenderFormat
        case manualRenderingFailed(String)

        var errorDescription: String? {
            switch self {
            case .engineNotRunning:
                return "Audio engine is not running."
            case .unableToAllocateOverlayBuffer:
                return "Unable to allocate overlay audio buffer."
            case .unableToAllocateRenderBuffer:
                return "Unable to allocate rendered MIDI buffer."
            case .overlayRecordingMissing:
                return "No recording is loaded for overlay playback."
            case .unsupportedRenderFormat:
                return "Unsupported audio format while rendering overlay audio."
            case let .manualRenderingFailed(status):
                return "Offline manual rendering failed: \(status)."
            }
        }
    }

    private let audioEngine = AVAudioEngine()
    private let sampler = AVAudioUnitSampler()
    private let audioPlayerNode = AVAudioPlayerNode()
    private let playbackQueue = DispatchQueue(label: "MemoToMIDI.MIDIPlayer.PlaybackQueue")

    private var timer: DispatchSourceTimer?
    private var isEngineConfigured = false
    private var soundFontURL: URL?
    private var currentPreset: PlaybackPreset = .sineWave
    private var isPrepared = false
    private var preparedNotesFingerprint: Int?

    private var scheduledNotes: [ScheduledNote] = []
    private var stopEvents: [StopEvent] = []
    private var nextStartIndex: Int = 0
    private var nextStopIndex: Int = 0
    private var activePitchCounts: [UInt8: Int] = [:]

    private var positionSeconds: Double = 0
    private var playbackBaseTime: Double = 0
    private var playbackStartAbsoluteTime: CFAbsoluteTime = 0
    private var playbackEndTime: Double = 0
    private var playing = false
    private var initialNotesActivated = false
    private var transportMode: TransportMode = .idle
    private var overlayPaused = false

    private var samplerVolume: Float = 1.0

    private var overlayBuffer: AVAudioPCMBuffer?
    private var overlaySourceURL: URL?

    private var cachedMIDIRender: AVAudioPCMBuffer?
    private var cachedMIDINotesFingerprint: Int?
    private var cachedMIDIDuration: Double = 0
    private var cachedMIDISampleRate: Double = 0
    private var cachedMIDIPreset: PlaybackPreset?
    private var cachedOverlayMix: AVAudioPCMBuffer?
    private var cachedOverlayRecordingGain: Float = 0.7
    private var cachedOverlayMIDIGain: Float = 0.7

    private let tickInterval: TimeInterval = 1.0 / 240.0
    private let autoStopBuffer: Double = 0.06
    private let manualRenderChunkSize: AVAudioFrameCount = 256

    deinit {
        playbackQueue.sync {
            cancelTimerLocked()
            stopAllActiveNotesLocked()
            if audioEngine.isRunning {
                audioEngine.stop()
            }
        }
    }

    func setup() throws {
        try playbackQueue.sync {
            try ensureEngineReadyLocked()
            applyPresetLocked(currentPreset)
        }
    }

    /// Loads a recording WAV file for overlay playback through the same engine.
    func loadRecordingForOverlay(url: URL) throws {
        try playbackQueue.sync {
            if overlaySourceURL == url, overlayBuffer != nil {
                return
            }

            try ensureEngineReadyLocked()

            let audioFile = try AVAudioFile(forReading: url)
            let frameCount = AVAudioFrameCount(audioFile.length)
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: audioFile.processingFormat,
                frameCapacity: frameCount
            ) else {
                throw MIDIPlayerError.unableToAllocateOverlayBuffer
            }
            try audioFile.read(into: buffer)

            overlayBuffer = buffer
            overlaySourceURL = url
            invalidateRenderedMIDICacheLocked()
            cachedOverlayMix = nil

            audioPlayerNode.stop()
            audioEngine.disconnectNodeOutput(audioPlayerNode)
            audioEngine.connect(audioPlayerNode, to: audioEngine.mainMixerNode, format: audioFile.processingFormat)
        }
    }

    /// Call before play() to eliminate startup latency.
    /// Prepares engine and pre-sorts the note schedule for timer-driven playback.
    func prepare(notes: [NoteEvent]) {
        let notesFingerprint = scheduleFingerprint(for: notes)
        playbackQueue.sync {
            do {
                try ensureEngineReadyLocked()
                prepareScheduleLocked(notes: notes)
                isPrepared = true
                preparedNotesFingerprint = notesFingerprint
            } catch {
                isPrepared = false
                preparedNotesFingerprint = nil
            }
        }
    }

    func play(notes: [NoteEvent], from startTime: Double? = nil, startDelay: TimeInterval = 0) {
        let clampedDelay = max(startDelay, 0)
        let intendedStartAbsoluteTime = CFAbsoluteTimeGetCurrent() + clampedDelay
        let notesFingerprint = scheduleFingerprint(for: notes)

        playbackQueue.async { [weak self] in
            guard let self else { return }
            if playing {
                return
            }

            if !isPrepared || preparedNotesFingerprint != notesFingerprint || !audioEngine.isRunning {
                do {
                    try ensureEngineReadyLocked()
                    prepareScheduleLocked(notes: notes)
                    isPrepared = true
                    preparedNotesFingerprint = notesFingerprint
                } catch {
                    publishPlayingOnMain(false)
                    return
                }
            }

            guard !scheduledNotes.isEmpty else {
                positionSeconds = 0
                publishCurrentTimeOnMain(positionSeconds)
                publishPlayingOnMain(false)
                return
            }

            if let startTime {
                positionSeconds = min(max(startTime, 0), playbackEndTime)
            } else {
                positionSeconds = min(positionSeconds, playbackEndTime)
            }

            stopAllActiveNotesLocked()
            rebuildCursorsLocked(at: positionSeconds)
            if clampedDelay == 0 {
                activateNotesThatShouldBeSoundingLocked(at: positionSeconds)
                initialNotesActivated = true
            } else {
                initialNotesActivated = false
            }

            transportMode = .midiOnlyPreview
            playbackBaseTime = positionSeconds
            playbackStartAbsoluteTime = intendedStartAbsoluteTime
            playing = true
            overlayPaused = false
            startTimerLocked()
            handleTimerTickLocked()

            publishCurrentTimeOnMain(positionSeconds)
            publishPlayingOnMain(true)
        }
    }

    func playOverlay(
        notes: [NoteEvent],
        recordingGain: Float = 0.7,
        midiGain: Float = 0.7,
        from startTime: Double? = nil
    ) {
        playbackQueue.async { [weak self] in
            guard let self else { return }

            if playing {
                return
            }

            do {
                try ensureEngineReadyLocked()
                guard let recordingBuffer = overlayBuffer else {
                    throw MIDIPlayerError.overlayRecordingMissing
                }

                // Resume from pause without re-rendering.
                if transportMode == .overlayRendered, overlayPaused {
                    resumeRenderedOverlayLocked()
                    return
                }

                if let startTime, startTime > 0 {
                    // v1 overlay playback intentionally starts from beginning only.
                    positionSeconds = 0
                }

                let duration = Double(recordingBuffer.frameLength) / recordingBuffer.format.sampleRate
                let sampleRate = recordingBuffer.format.sampleRate

                publishOverlayPreparingOnMain(true)
                publishOverlayPreparationErrorOnMain(nil)

                let clampedRecordingGain = min(max(recordingGain, 0), 1)
                let clampedMIDIGain = min(max(midiGain, 0), 1)

                let midiBuffer: AVAudioPCMBuffer
                var renderElapsed: CFAbsoluteTime = 0
                if canReuseRenderedMIDIBufferLocked(for: notes, duration: duration, sampleRate: sampleRate),
                   let cachedBuffer = cachedMIDIRender {
                    midiBuffer = cachedBuffer
                } else {
                    let renderStart = CFAbsoluteTimeGetCurrent()
                    midiBuffer = try renderMIDIToBuffer(notes: notes, duration: duration, sampleRate: sampleRate)
                    cacheRenderedMIDIBufferLocked(midiBuffer, notes: notes, duration: duration, sampleRate: sampleRate)
                    renderElapsed = CFAbsoluteTimeGetCurrent() - renderStart
                }

                print("OVERLAY: rec format=\(recordingBuffer.format) frames=\(recordingBuffer.frameLength)")
                print("OVERLAY: midi format=\(midiBuffer.format) frames=\(midiBuffer.frameLength)")

                let mixedBuffer: AVAudioPCMBuffer
                let mixElapsed: CFAbsoluteTime
                if let cachedMix = cachedOverlayMix,
                   abs(cachedOverlayRecordingGain - clampedRecordingGain) < 0.0001,
                   abs(cachedOverlayMIDIGain - clampedMIDIGain) < 0.0001,
                   formatsMatch(cachedMix.format, recordingBuffer.format),
                   cachedMix.frameLength == min(recordingBuffer.frameLength, midiBuffer.frameLength) {
                    mixedBuffer = cachedMix
                    mixElapsed = 0
                } else {
                    let mixStart = CFAbsoluteTimeGetCurrent()
                    guard let remixedBuffer = mixOverlayBuffers(
                        recording: recordingBuffer,
                        midi: midiBuffer,
                        recordingGain: clampedRecordingGain,
                        midiGain: clampedMIDIGain
                    ) else {
                        throw MIDIPlayerError.unsupportedRenderFormat
                    }
                    mixedBuffer = remixedBuffer
                    mixElapsed = CFAbsoluteTimeGetCurrent() - mixStart

                    cacheOverlayMixBufferLocked(
                        mixedBuffer,
                        recordingGain: clampedRecordingGain,
                        midiGain: clampedMIDIGain
                    )
                }

                print(String(format: "Overlay render: %.2fs, mix: %.4fs", renderElapsed, mixElapsed))

                publishOverlayPreparingOnMain(false)

                stopAllActiveNotesLocked()
                audioPlayerNode.stop()
                audioPlayerNode.scheduleBuffer(mixedBuffer, at: nil, options: [], completionHandler: nil)

                let nowAbsolute = CFAbsoluteTimeGetCurrent()
                audioPlayerNode.play()
                playbackStartAbsoluteTime = nowAbsolute

                positionSeconds = 0
                playbackBaseTime = 0
                playbackEndTime = duration + autoStopBuffer
                playing = true
                initialNotesActivated = false
                overlayPaused = false
                transportMode = .overlayRendered
                startTimerLocked()

                publishCurrentTimeOnMain(0)
                publishPlayingOnMain(true)
            } catch {
                publishOverlayPreparingOnMain(false)
                publishOverlayPreparationErrorOnMain(error.localizedDescription)
                stopLocked(resetToStart: false)
            }
        }
    }

    func pause() {
        playbackQueue.async { [weak self] in
            guard let self else { return }
            guard playing else { return }

            updatePositionFromClockLocked()
            cancelTimerLocked()

            switch transportMode {
            case .overlayRendered:
                audioPlayerNode.pause()
                overlayPaused = true
            case .midiOnlyPreview:
                stopAllActiveNotesLocked()
                initialNotesActivated = false
                overlayPaused = false
            case .idle:
                break
            }

            playing = false
            publishCurrentTimeOnMain(positionSeconds)
            publishPlayingOnMain(false)
        }
    }

    func stop() {
        playbackQueue.async { [weak self] in
            guard let self else { return }
            stopLocked(resetToStart: true)
        }
    }

    func seek(to time: Double) {
        playbackQueue.async { [weak self] in
            guard let self else { return }

            // v1 overlay mode intentionally does not support seeking/scrubbing.
            if transportMode == .overlayRendered || overlayPaused {
                publishCurrentTimeOnMain(positionSeconds)
                return
            }

            let maxSeekTime = playbackEndTime > 0 ? playbackEndTime : max(time, 0)
            let clampedTime = min(max(time, 0), maxSeekTime)
            positionSeconds = clampedTime

            if playing {
                stopAllActiveNotesLocked()
                rebuildCursorsLocked(at: positionSeconds)
                activateNotesThatShouldBeSoundingLocked(at: positionSeconds)
                initialNotesActivated = true
                playbackBaseTime = positionSeconds
                playbackStartAbsoluteTime = CFAbsoluteTimeGetCurrent()
            } else {
                rebuildCursorsLocked(at: positionSeconds)
            }

            publishCurrentTimeOnMain(positionSeconds)
        }
    }

    func setPreset(_ preset: PlaybackPreset) {
        playbackQueue.async { [weak self] in
            guard let self else { return }
            currentPreset = preset
            let wasPlaying = playing
            let priorTransportMode = transportMode

            do {
                try ensureEngineReadyLocked()
            } catch {
                return
            }

            stopAllActiveNotesLocked()
            applyPresetLocked(preset)
            invalidateRenderedMIDICacheLocked()

            if wasPlaying, priorTransportMode == .midiOnlyPreview {
                transportMode = priorTransportMode
                activateNotesThatShouldBeSoundingLocked(at: positionSeconds)
            } else if wasPlaying {
                cancelTimerLocked()
                playing = false
                publishPlayingOnMain(false)
            }
        }
    }

    func setVolume(_ volume: Float) {
        setSamplerVolume(volume)
    }

    func setSamplerVolume(_ volume: Float) {
        playbackQueue.async { [weak self] in
            guard let self else { return }
            let clamped = min(max(volume, 0), 1)
            samplerVolume = clamped
            sampler.volume = clamped
        }
    }

    func updateOverlayMix(recordingGain: Float, midiGain: Float) {
        playbackQueue.async { [weak self] in
            guard let self else { return }
            guard transportMode == .overlayRendered || overlayPaused else { return }
            guard let recordingBuffer = overlayBuffer,
                  let midiBuffer = cachedMIDIRender else { return }

            let clampedRecordingGain = min(max(recordingGain, 0), 1)
            let clampedMIDIGain = min(max(midiGain, 0), 1)

            let mixStart = CFAbsoluteTimeGetCurrent()
            guard let mixedBuffer = mixOverlayBuffers(
                recording: recordingBuffer,
                midi: midiBuffer,
                recordingGain: clampedRecordingGain,
                midiGain: clampedMIDIGain
            ) else { return }

            cacheOverlayMixBufferLocked(
                mixedBuffer,
                recordingGain: clampedRecordingGain,
                midiGain: clampedMIDIGain
            )

            let wasPlaying = playing
            let currentPos = max(positionSeconds, 0)

            audioPlayerNode.stop()

            guard let scheduledBuffer = trimmedOverlayBuffer(mixedBuffer, startingAt: currentPos) else {
                if wasPlaying {
                    stopLocked(resetToStart: true)
                }
                return
            }

            audioPlayerNode.scheduleBuffer(scheduledBuffer, at: nil, options: [], completionHandler: nil)

            if wasPlaying {
                audioPlayerNode.play()
                playbackBaseTime = currentPos
                playbackStartAbsoluteTime = CFAbsoluteTimeGetCurrent()
            } else {
                playbackBaseTime = currentPos
            }

            let mixElapsed = CFAbsoluteTimeGetCurrent() - mixStart
            print(String(format: "Overlay remix: %.4fs", mixElapsed))
        }
    }

    private func ensureEngineReadyLocked() throws {
        try configureAudioSession()

        if !isEngineConfigured {
            audioEngine.attach(sampler)
            audioEngine.attach(audioPlayerNode)

            audioEngine.connect(sampler, to: audioEngine.mainMixerNode, format: nil)
            audioEngine.connect(audioPlayerNode, to: audioEngine.mainMixerNode, format: nil)

            soundFontURL = Self.findBundledSoundFont()
            isEngineConfigured = true
        }

        sampler.volume = samplerVolume

        if !audioEngine.isRunning {
            audioEngine.prepare()
            try audioEngine.start()
        }

        guard audioEngine.isRunning else {
            throw MIDIPlayerError.engineNotRunning
        }
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: .default,
            options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers]
        )
        try session.setActive(true)
    }

    private func resumeRenderedOverlayLocked() {
        guard transportMode == .overlayRendered, overlayPaused else {
            return
        }

        let nowAbsolute = CFAbsoluteTimeGetCurrent()
        audioPlayerNode.play()
        playbackStartAbsoluteTime = nowAbsolute

        playbackBaseTime = positionSeconds
        playing = true
        overlayPaused = false
        startTimerLocked()

        publishCurrentTimeOnMain(positionSeconds)
        publishPlayingOnMain(true)
    }

    private func prepareScheduleLocked(notes: [NoteEvent]) {
        scheduledNotes = notes
            .compactMap { note in
                let start = max(note.startTime, 0)
                let duration = max(note.duration, 0)
                guard duration > 0 else { return nil }
                return ScheduledNote(
                    id: note.id,
                    pitch: note.pitch,
                    velocity: max(note.velocity, 1),
                    startTime: start,
                    endTime: start + duration
                )
            }
            .sorted {
                if $0.startTime == $1.startTime {
                    return $0.pitch < $1.pitch
                }
                return $0.startTime < $1.startTime
            }

        stopEvents = scheduledNotes
            .map { StopEvent(pitch: $0.pitch, endTime: $0.endTime) }
            .sorted {
                if $0.endTime == $1.endTime {
                    return $0.pitch < $1.pitch
                }
                return $0.endTime < $1.endTime
            }

        playbackEndTime = (scheduledNotes.map(\.endTime).max() ?? 0) + autoStopBuffer
        nextStartIndex = 0
        nextStopIndex = 0
    }

    private func rebuildCursorsLocked(at time: Double) {
        nextStartIndex = scheduledNotes.firstIndex(where: { $0.startTime > time }) ?? scheduledNotes.count
        nextStopIndex = stopEvents.firstIndex(where: { $0.endTime > time }) ?? stopEvents.count
    }

    private func startTimerLocked() {
        cancelTimerLocked()

        let timer = DispatchSource.makeTimerSource(queue: playbackQueue)
        timer.schedule(deadline: .now(), repeating: tickInterval, leeway: .milliseconds(1))
        timer.setEventHandler { [weak self] in
            self?.handleTimerTickLocked()
        }
        self.timer = timer
        timer.resume()
    }

    private func cancelTimerLocked() {
        timer?.setEventHandler {}
        timer?.cancel()
        timer = nil
    }

    private func handleTimerTickLocked() {
        guard playing else { return }
        if CFAbsoluteTimeGetCurrent() < playbackStartAbsoluteTime {
            publishCurrentTimeOnMain(positionSeconds)
            return
        }

        switch transportMode {
        case .midiOnlyPreview:
            if !initialNotesActivated {
                activateNotesThatShouldBeSoundingLocked(at: playbackBaseTime)
                initialNotesActivated = true
            }
            updatePositionFromClockLocked()
            processScheduledEventsLocked(at: positionSeconds)
        case .overlayRendered:
            updatePositionFromClockLocked()
        case .idle:
            break
        }

        publishCurrentTimeOnMain(positionSeconds)

        if positionSeconds >= playbackEndTime {
            stopLocked(resetToStart: true)
        }
    }

    private func updatePositionFromClockLocked() {
        let elapsed = max(CFAbsoluteTimeGetCurrent() - playbackStartAbsoluteTime, 0)
        positionSeconds = min(playbackBaseTime + elapsed, playbackEndTime)
    }

    private func processScheduledEventsLocked(at time: Double) {
        while nextStartIndex < scheduledNotes.count, scheduledNotes[nextStartIndex].startTime <= time {
            start(note: scheduledNotes[nextStartIndex])
            nextStartIndex += 1
        }

        while nextStopIndex < stopEvents.count, stopEvents[nextStopIndex].endTime <= time {
            stop(noteEvent: stopEvents[nextStopIndex])
            nextStopIndex += 1
        }
    }

    private func activateNotesThatShouldBeSoundingLocked(at time: Double) {
        for note in scheduledNotes {
            if note.startTime > time {
                break
            }
            if note.endTime > time {
                start(note: note)
            }
        }
    }

    private func start(note: ScheduledNote) {
        sampler.startNote(note.pitch, withVelocity: note.velocity, onChannel: 0)
        activePitchCounts[note.pitch, default: 0] += 1
    }

    private func stop(noteEvent: StopEvent) {
        guard let count = activePitchCounts[noteEvent.pitch] else { return }

        if count <= 1 {
            activePitchCounts.removeValue(forKey: noteEvent.pitch)
            sampler.stopNote(noteEvent.pitch, onChannel: 0)
        } else {
            activePitchCounts[noteEvent.pitch] = count - 1
        }
    }

    private func stopAllActiveNotesLocked() {
        for pitch in activePitchCounts.keys {
            sampler.stopNote(pitch, onChannel: 0)
        }
        activePitchCounts.removeAll()

        audioPlayerNode.stop()
        overlayPaused = false
        transportMode = .idle
    }

    private func stopLocked(resetToStart: Bool) {
        cancelTimerLocked()
        stopAllActiveNotesLocked()
        playing = false
        initialNotesActivated = false

        if resetToStart {
            positionSeconds = 0
            nextStartIndex = 0
            nextStopIndex = 0
            isPrepared = false
            preparedNotesFingerprint = nil
        }

        publishCurrentTimeOnMain(positionSeconds)
        publishPlayingOnMain(false)
    }

    /// Renders all notes to an audio buffer using the sampler offline.
    /// Returns a buffer matching the recording's duration and sample rate.
    private func renderMIDIToBuffer(
        notes: [NoteEvent],
        duration: Double,
        sampleRate: Double
    ) throws -> AVAudioPCMBuffer {
        guard let renderFormat = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        ) else {
            throw MIDIPlayerError.unsupportedRenderFormat
        }

        let totalFramesPosition = AVAudioFramePosition(max((duration * sampleRate).rounded(.up), 0))
        let safeFrameCapacity = AVAudioFrameCount(max(totalFramesPosition, 1))
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: renderFormat, frameCapacity: safeFrameCapacity) else {
            throw MIDIPlayerError.unableToAllocateRenderBuffer
        }
        outputBuffer.frameLength = AVAudioFrameCount(totalFramesPosition)

        guard totalFramesPosition > 0 else {
            return outputBuffer
        }

        let offlineEngine = AVAudioEngine()
        let offlineSampler = AVAudioUnitSampler()
        offlineEngine.attach(offlineSampler)
        offlineEngine.connect(offlineSampler, to: offlineEngine.mainMixerNode, format: renderFormat)
        offlineEngine.connect(offlineEngine.mainMixerNode, to: offlineEngine.outputNode, format: renderFormat)

        try offlineEngine.enableManualRenderingMode(
            .offline,
            format: renderFormat,
            maximumFrameCount: manualRenderChunkSize
        )

        applyPreset(currentPreset, to: offlineSampler)
        offlineSampler.volume = 1.0

        offlineEngine.prepare()
        try offlineEngine.start()
        defer {
            offlineEngine.stop()
        }

        guard let renderChunkBuffer = AVAudioPCMBuffer(pcmFormat: renderFormat, frameCapacity: manualRenderChunkSize) else {
            throw MIDIPlayerError.unableToAllocateRenderBuffer
        }

        var events: [OverlayRenderEvent] = []
        events.reserveCapacity(notes.count * 2)

        for note in notes {
            let startSeconds = max(note.startTime, 0)
            let durationSeconds = max(note.duration, 0)
            guard durationSeconds > 0 else { continue }

            let rawStart = AVAudioFramePosition((startSeconds * sampleRate).rounded())
            let rawEnd = AVAudioFramePosition(((startSeconds + durationSeconds) * sampleRate).rounded())

            if rawEnd <= 0 || rawStart >= totalFramesPosition {
                continue
            }

            let startSample = min(max(rawStart, 0), totalFramesPosition)
            let unclampedEnd = max(rawEnd, startSample + 1)
            let endSample = min(max(unclampedEnd, 0), totalFramesPosition)

            events.append(OverlayRenderEvent(
                samplePosition: startSample,
                pitch: note.pitch,
                velocity: max(note.velocity, 1),
                isNoteOn: true
            ))
            events.append(OverlayRenderEvent(
                samplePosition: endSample,
                pitch: note.pitch,
                velocity: 0,
                isNoteOn: false
            ))
        }

        events.sort { lhs, rhs in
            if lhs.samplePosition != rhs.samplePosition {
                return lhs.samplePosition < rhs.samplePosition
            }
            if lhs.isNoteOn != rhs.isNoteOn {
                // Note-off first so a same-sample retrigger gets a fresh attack.
                return !lhs.isNoteOn
            }
            return lhs.pitch < rhs.pitch
        }

        var activePitchCounts: [UInt8: Int] = [:]
        var currentFrame: AVAudioFramePosition = 0
        var nextEventIndex = 0

        while currentFrame < totalFramesPosition {
            // Fire all note events due at this sample position before rendering.
            while nextEventIndex < events.count,
                  events[nextEventIndex].samplePosition <= currentFrame {
                let event = events[nextEventIndex]
                if event.isNoteOn {
                    offlineSampler.startNote(event.pitch, withVelocity: event.velocity, onChannel: 0)
                    activePitchCounts[event.pitch, default: 0] += 1
                } else {
                    let currentCount = activePitchCounts[event.pitch] ?? 0
                    if currentCount <= 1 {
                        activePitchCounts.removeValue(forKey: event.pitch)
                        offlineSampler.stopNote(event.pitch, onChannel: 0)
                    } else {
                        activePitchCounts[event.pitch] = currentCount - 1
                    }
                }
                nextEventIndex += 1
            }

            let nextEventFrame = nextEventIndex < events.count ? events[nextEventIndex].samplePosition : totalFramesPosition
            if nextEventFrame <= currentFrame {
                continue
            }

            let framesUntilEvent = AVAudioFrameCount(nextEventFrame - currentFrame)
            let framesUntilEnd = AVAudioFrameCount(totalFramesPosition - currentFrame)
            let framesToRender = min(framesUntilEvent, framesUntilEnd, manualRenderChunkSize)

            var renderStatus: AVAudioEngineManualRenderingStatus = .success
            repeat {
                renderStatus = try offlineEngine.renderOffline(framesToRender, to: renderChunkBuffer)
            } while renderStatus == .cannotDoInCurrentContext

            switch renderStatus {
            case .success:
                try copyChunk(
                    renderChunkBuffer,
                    to: outputBuffer,
                    destinationFrame: currentFrame,
                    frameCount: framesToRender
                )
            case .insufficientDataFromInputNode:
                break
            case .error:
                throw MIDIPlayerError.manualRenderingFailed("error")
            case .cannotDoInCurrentContext:
                throw MIDIPlayerError.manualRenderingFailed("cannotDoInCurrentContext")
            @unknown default:
                throw MIDIPlayerError.manualRenderingFailed("unknown")
            }

            currentFrame += AVAudioFramePosition(framesToRender)
        }

        return outputBuffer
    }

    private func copyChunk(
        _ sourceChunk: AVAudioPCMBuffer,
        to destinationBuffer: AVAudioPCMBuffer,
        destinationFrame: AVAudioFramePosition,
        frameCount: AVAudioFrameCount
    ) throws {
        guard sourceChunk.format.commonFormat == .pcmFormatFloat32,
              destinationBuffer.format.commonFormat == .pcmFormatFloat32 else {
            throw MIDIPlayerError.unsupportedRenderFormat
        }

        let channelCount = Int(destinationBuffer.format.channelCount)
        let framesToCopy = Int(frameCount)

        if destinationBuffer.format.isInterleaved {
            guard let sourcePointer = sourceChunk.floatChannelData?.pointee,
                  let destinationPointer = destinationBuffer.floatChannelData?.pointee else {
                throw MIDIPlayerError.unsupportedRenderFormat
            }
            let destinationOffset = Int(destinationFrame) * channelCount
            memcpy(
                destinationPointer.advanced(by: destinationOffset),
                sourcePointer,
                framesToCopy * channelCount * MemoryLayout<Float>.size
            )
        } else {
            guard let sourceChannels = sourceChunk.floatChannelData,
                  let destinationChannels = destinationBuffer.floatChannelData else {
                throw MIDIPlayerError.unsupportedRenderFormat
            }

            let destinationOffset = Int(destinationFrame)
            for channel in 0 ..< channelCount {
                memcpy(
                    destinationChannels[channel].advanced(by: destinationOffset),
                    sourceChannels[channel],
                    framesToCopy * MemoryLayout<Float>.size
                )
            }
        }
    }

    private func cacheRenderedMIDIBufferLocked(
        _ buffer: AVAudioPCMBuffer,
        notes: [NoteEvent],
        duration: Double,
        sampleRate: Double
    ) {
        cachedMIDIRender = buffer
        cachedMIDINotesFingerprint = scheduleFingerprint(for: notes)
        cachedMIDIDuration = duration
        cachedMIDISampleRate = sampleRate
        cachedMIDIPreset = currentPreset
        cachedOverlayMix = nil
    }

    private func invalidateRenderedMIDICacheLocked() {
        cachedMIDIRender = nil
        cachedMIDINotesFingerprint = nil
        cachedMIDIDuration = 0
        cachedMIDISampleRate = 0
        cachedMIDIPreset = nil
        cachedOverlayMix = nil
    }

    private func canReuseRenderedMIDIBufferLocked(
        for notes: [NoteEvent],
        duration: Double,
        sampleRate: Double
    ) -> Bool {
        guard cachedMIDIRender != nil else { return false }
        guard let cachedNotesFingerprint = cachedMIDINotesFingerprint else { return false }
        guard cachedMIDIPreset == currentPreset else { return false }
        guard abs(cachedMIDIDuration - duration) < 0.0001 else { return false }
        guard abs(cachedMIDISampleRate - sampleRate) < 0.0001 else { return false }
        return cachedNotesFingerprint == scheduleFingerprint(for: notes)
    }

    private func cacheOverlayMixBufferLocked(
        _ buffer: AVAudioPCMBuffer,
        recordingGain: Float,
        midiGain: Float
    ) {
        cachedOverlayMix = buffer
        cachedOverlayRecordingGain = recordingGain
        cachedOverlayMIDIGain = midiGain
    }

    /// Mixes recording and rendered MIDI into a single playback buffer.
    private func mixOverlayBuffers(
        recording: AVAudioPCMBuffer,
        midi: AVAudioPCMBuffer,
        recordingGain: Float,
        midiGain: Float
    ) -> AVAudioPCMBuffer? {
        guard formatsMatch(recording.format, midi.format) else {
            print("OVERLAY MIX ERROR: format mismatch - recording: \(recording.format), midi: \(midi.format)")
            return nil
        }

        guard recording.format.commonFormat == .pcmFormatFloat32,
              midi.format.commonFormat == .pcmFormatFloat32,
              !recording.format.isInterleaved,
              !midi.format.isInterleaved else {
            print("OVERLAY MIX ERROR: unsupported format for mixing (requires non-interleaved Float32)")
            return nil
        }

        let frameCount = min(recording.frameLength, midi.frameLength)
        if recording.frameLength != midi.frameLength {
            print("OVERLAY MIX WARN: frame mismatch - recording: \(recording.frameLength), midi: \(midi.frameLength)")
        }

        guard let output = AVAudioPCMBuffer(pcmFormat: recording.format, frameCapacity: frameCount) else {
            return nil
        }
        output.frameLength = frameCount

        guard let recordingChannels = recording.floatChannelData,
              let midiChannels = midi.floatChannelData,
              let outputChannels = output.floatChannelData else {
            return nil
        }

        let channels = Int(recording.format.channelCount)
        let frameLength = vDSP_Length(frameCount)
        var recordingScale = recordingGain
        var midiScale = midiGain

        for channel in 0 ..< channels {
            let recordingData = recordingChannels[channel]
            let midiData = midiChannels[channel]
            let outputData = outputChannels[channel]

            vDSP_vsmul(recordingData, 1, &recordingScale, outputData, 1, frameLength)
            vDSP_vsma(midiData, 1, &midiScale, outputData, 1, outputData, 1, frameLength)
        }

        return output
    }

    private func formatsMatch(_ lhs: AVAudioFormat, _ rhs: AVAudioFormat) -> Bool {
        lhs.commonFormat == rhs.commonFormat
            && lhs.channelCount == rhs.channelCount
            && lhs.isInterleaved == rhs.isInterleaved
            && abs(lhs.sampleRate - rhs.sampleRate) < 0.0001
    }

    private func trimmedOverlayBuffer(_ source: AVAudioPCMBuffer, startingAt time: Double) -> AVAudioPCMBuffer? {
        let sampleRate = source.format.sampleRate
        let rawStartFrame = AVAudioFramePosition((max(time, 0) * sampleRate).rounded())
        let startFrame = min(max(rawStartFrame, 0), AVAudioFramePosition(source.frameLength))

        if startFrame == 0 {
            return source
        }

        let remainingFramesPosition = AVAudioFramePosition(source.frameLength) - startFrame
        guard remainingFramesPosition > 0 else {
            return nil
        }
        let remainingFrames = AVAudioFrameCount(remainingFramesPosition)

        guard source.format.commonFormat == .pcmFormatFloat32,
              !source.format.isInterleaved,
              let sourceChannels = source.floatChannelData,
              let output = AVAudioPCMBuffer(pcmFormat: source.format, frameCapacity: remainingFrames),
              let outputChannels = output.floatChannelData else {
            return nil
        }

        output.frameLength = remainingFrames
        let channels = Int(source.format.channelCount)
        let startIndex = Int(startFrame)
        let bytesToCopy = Int(remainingFrames) * MemoryLayout<Float>.size

        for channel in 0 ..< channels {
            memcpy(outputChannels[channel], sourceChannels[channel].advanced(by: startIndex), bytesToCopy)
        }

        return output
    }

    private func scheduleFingerprint(for notes: [NoteEvent]) -> Int {
        var hasher = Hasher()
        hasher.combine(notes.count)
        for note in notes {
            hasher.combine(note.id)
            hasher.combine(note.pitch)
            hasher.combine(note.startTime.bitPattern)
            hasher.combine(note.duration.bitPattern)
            hasher.combine(note.velocity)
        }
        return hasher.finalize()
    }

    private func applyPresetLocked(_ preset: PlaybackPreset) {
        applyPreset(preset, to: sampler)
    }

    private func applyPreset(_ preset: PlaybackPreset, to sampler: AVAudioUnitSampler) {
        let melodicBankMSB: UInt8 = 0x79
        let bankLSB: UInt8 = 0

        if let soundFontURL {
            do {
                switch preset {
                case .piano:
                    try sampler.loadSoundBankInstrument(
                        at: soundFontURL,
                        program: 0,
                        bankMSB: melodicBankMSB,
                        bankLSB: bankLSB
                    )
                case .cleanGuitar:
                    try sampler.loadSoundBankInstrument(
                        at: soundFontURL,
                        program: 25,
                        bankMSB: melodicBankMSB,
                        bankLSB: bankLSB
                    )
                case .sineWave:
                    sampler.sendProgramChange(80, bankMSB: melodicBankMSB, bankLSB: bankLSB, onChannel: 0)
                }
                return
            } catch {
                // If loading the bundled soundfont fails, fall through to program-change fallback.
            }
        }

        // Fallback for v1: rely on sampler's default tone.
        switch preset {
        case .piano:
            sampler.sendProgramChange(0, bankMSB: melodicBankMSB, bankLSB: bankLSB, onChannel: 0)
        case .cleanGuitar:
            sampler.sendProgramChange(25, bankMSB: melodicBankMSB, bankLSB: bankLSB, onChannel: 0)
        case .sineWave:
            sampler.sendProgramChange(80, bankMSB: melodicBankMSB, bankLSB: bankLSB, onChannel: 0)
        }
    }

    private func publishPlayingOnMain(_ playing: Bool) {
        DispatchQueue.main.async {
            self.isPlaying = playing
        }
    }

    private func publishCurrentTimeOnMain(_ time: Double) {
        DispatchQueue.main.async {
            self.currentTime = time
        }
    }

    private func publishOverlayPreparingOnMain(_ preparing: Bool) {
        DispatchQueue.main.async {
            self.isPreparingOverlay = preparing
        }
    }

    private func publishOverlayPreparationErrorOnMain(_ message: String?) {
        DispatchQueue.main.async {
            self.overlayPreparationError = message
        }
    }

    private static func findBundledSoundFont() -> URL? {
        if let soundInFolder = Bundle.main.urls(forResourcesWithExtension: "sf2", subdirectory: "Sounds")?.first {
            return soundInFolder
        }
        if let soundInResourcesFolder = Bundle.main.urls(forResourcesWithExtension: "sf2", subdirectory: "Resources/Sounds")?.first {
            return soundInResourcesFolder
        }
        return Bundle.main.urls(forResourcesWithExtension: "sf2", subdirectory: nil)?.first
    }
}

enum PlaybackPreset: String, CaseIterable {
    case piano = "Piano"
    case cleanGuitar = "Clean Guitar"
    case sineWave = "Sine"
}
