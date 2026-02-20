import AVFoundation
import Combine
import Foundation

final class MIDIPlayer: ObservableObject {
    @Published var isPlaying: Bool = false
    @Published var currentTime: Double = 0.0

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

    private enum MIDIPlayerError: LocalizedError {
        case engineNotRunning

        var errorDescription: String? {
            switch self {
            case .engineNotRunning:
                return "Audio engine is not running."
            }
        }
    }

    private let audioEngine = AVAudioEngine()
    private let sampler = AVAudioUnitSampler()
    private let playbackQueue = DispatchQueue(label: "MemoToMIDI.MIDIPlayer.PlaybackQueue")

    private var timer: DispatchSourceTimer?
    private var isEngineConfigured = false
    private var soundFontURL: URL?
    private var currentPreset: PlaybackPreset = .sineWave

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

    private let tickInterval: TimeInterval = 1.0 / 60.0
    private let autoStopBuffer: Double = 0.06

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

    func play(notes: [NoteEvent]) {
        playbackQueue.async { [weak self] in
            guard let self else { return }

            do {
                try ensureEngineReadyLocked()
            } catch {
                publishPlayingOnMain(false)
                return
            }

            if playing {
                return
            }

            prepareScheduleLocked(notes: notes)
            guard !scheduledNotes.isEmpty else {
                positionSeconds = 0
                publishCurrentTimeOnMain(positionSeconds)
                publishPlayingOnMain(false)
                return
            }
            positionSeconds = min(positionSeconds, playbackEndTime)

            stopAllActiveNotesLocked()
            rebuildCursorsLocked(at: positionSeconds)
            activateNotesThatShouldBeSoundingLocked(at: positionSeconds)

            playbackBaseTime = positionSeconds
            playbackStartAbsoluteTime = CFAbsoluteTimeGetCurrent()
            playing = true
            startTimerLocked()

            publishCurrentTimeOnMain(positionSeconds)
            publishPlayingOnMain(true)
        }
    }

    func pause() {
        playbackQueue.async { [weak self] in
            guard let self else { return }
            guard playing else { return }

            updatePositionFromClockLocked()
            cancelTimerLocked()
            stopAllActiveNotesLocked()
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

            let maxSeekTime = playbackEndTime > 0 ? playbackEndTime : max(time, 0)
            let clampedTime = min(max(time, 0), maxSeekTime)
            positionSeconds = clampedTime

            if playing {
                stopAllActiveNotesLocked()
                rebuildCursorsLocked(at: positionSeconds)
                activateNotesThatShouldBeSoundingLocked(at: positionSeconds)
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

            do {
                try ensureEngineReadyLocked()
            } catch {
                return
            }

            stopAllActiveNotesLocked()
            applyPresetLocked(preset)

            if playing {
                activateNotesThatShouldBeSoundingLocked(at: positionSeconds)
            }
        }
    }

    private func ensureEngineReadyLocked() throws {
        try configureAudioSession()

        if !isEngineConfigured {
            audioEngine.attach(sampler)
            audioEngine.connect(sampler, to: audioEngine.mainMixerNode, format: nil)
            soundFontURL = Self.findBundledSoundFont()
            isEngineConfigured = true
        }

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
        try session.setCategory(.playback, mode: .default, options: [])
        try session.setActive(true)
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
        timer.schedule(deadline: .now(), repeating: tickInterval, leeway: .milliseconds(2))
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
        updatePositionFromClockLocked()
        processScheduledEventsLocked(at: positionSeconds)
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
    }

    private func stopLocked(resetToStart: Bool) {
        cancelTimerLocked()
        stopAllActiveNotesLocked()
        playing = false

        if resetToStart {
            positionSeconds = 0
            nextStartIndex = 0
            nextStopIndex = 0
        }

        publishCurrentTimeOnMain(positionSeconds)
        publishPlayingOnMain(false)
    }

    private func applyPresetLocked(_ preset: PlaybackPreset) {
        let melodicBankMSB: UInt8 = 0x79
        let bankLSB: UInt8 = 0

        if let soundFontURL {
            switch preset {
            case .piano:
                try? sampler.loadSoundBankInstrument(
                    at: soundFontURL,
                    program: 0,
                    bankMSB: melodicBankMSB,
                    bankLSB: bankLSB
                )
            case .cleanGuitar:
                try? sampler.loadSoundBankInstrument(
                    at: soundFontURL,
                    program: 25,
                    bankMSB: melodicBankMSB,
                    bankLSB: bankLSB
                )
            case .sineWave:
                sampler.sendProgramChange(80, bankMSB: melodicBankMSB, bankLSB: bankLSB, onChannel: 0)
            }
            return
        }

        // Fallback for v1: no bundled SoundFont, rely on sampler's default tone.
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
