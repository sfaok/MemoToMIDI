import Foundation

struct ExtractionParameters {
    var onsetThreshold: Float
    var frameThreshold: Float
    var minNoteLengthMs: Double
    var mergeGapMs: Double

    static let `default` = ExtractionParameters(
        onsetThreshold: MIDIConstants.defaultOnsetThreshold,
        frameThreshold: MIDIConstants.defaultFrameThreshold,
        minNoteLengthMs: MIDIConstants.minNoteLengthMs,
        mergeGapMs: 30.0
    )
}

struct NoteExtractor {
    private static let midiBasePitch = 21
    private static let expectedPitchBins = 88

    static func extract(
        from result: InferenceResult,
        parameters: ExtractionParameters = .default
    ) -> [NoteEvent] {
        let frameCount = min(result.noteActivations.count, result.onsetActivations.count)
        guard frameCount > 0 else { return [] }

        let pitchCount = min(
            expectedPitchBins,
            result.noteActivations[0].count,
            result.onsetActivations[0].count
        )
        guard pitchCount > 0 else { return [] }

        let onsetThreshold = parameters.onsetThreshold.clamped(to: 0...1)
        let frameThreshold = parameters.frameThreshold.clamped(to: 0...1)
        let minNoteLengthMs = max(0, parameters.minNoteLengthMs)
        let mergeGapMs = max(0, parameters.mergeGapMs)
        let frameDurationMs = result.frameDuration * 1000.0

        var events: [NoteEvent] = []
        events.reserveCapacity(64)

        for pitchIndex in 0..<pitchCount {
            var rawSegments: [RawSegment] = []
            rawSegments.reserveCapacity(16)

            var activeStartFrame = -1
            var activePeak: Float = 0

            for frameIndex in 0..<frameCount {
                let frameActivation = result.noteActivations[frameIndex][pitchIndex]
                let onsetActivation = result.onsetActivations[frameIndex][pitchIndex]
                let frameIsActive = frameActivation >= frameThreshold
                let onsetTriggered = onsetActivation >= onsetThreshold && frameIsActive

                if activeStartFrame < 0 {
                    if onsetTriggered {
                        activeStartFrame = frameIndex
                        activePeak = frameActivation
                    }
                    continue
                }

                if !frameIsActive {
                    rawSegments.append(
                        RawSegment(
                            startFrame: activeStartFrame,
                            endFrame: frameIndex,
                            peakActivation: activePeak
                        )
                    )
                    activeStartFrame = -1
                    activePeak = 0
                    continue
                }

                if onsetTriggered {
                    rawSegments.append(
                        RawSegment(
                            startFrame: activeStartFrame,
                            endFrame: frameIndex,
                            peakActivation: activePeak
                        )
                    )
                    activeStartFrame = frameIndex
                    activePeak = frameActivation
                    continue
                }

                if frameActivation > activePeak {
                    activePeak = frameActivation
                }
            }

            if activeStartFrame >= 0 {
                rawSegments.append(
                    RawSegment(
                        startFrame: activeStartFrame,
                        endFrame: frameCount,
                        peakActivation: activePeak
                    )
                )
            }

            var filteredSegments: [RawSegment] = []
            filteredSegments.reserveCapacity(rawSegments.count)
            for segment in rawSegments {
                let durationMs = Double(segment.endFrame - segment.startFrame) * frameDurationMs
                if durationMs >= minNoteLengthMs {
                    filteredSegments.append(segment)
                }
            }

            guard !filteredSegments.isEmpty else { continue }

            var mergedSegments: [RawSegment] = []
            mergedSegments.reserveCapacity(filteredSegments.count)
            var current = filteredSegments[0]

            for index in 1..<filteredSegments.count {
                let next = filteredSegments[index]
                let gapMs = Double(next.startFrame - current.endFrame) * frameDurationMs
                if gapMs < mergeGapMs {
                    current.endFrame = next.endFrame
                    if next.peakActivation > current.peakActivation {
                        current.peakActivation = next.peakActivation
                    }
                } else {
                    mergedSegments.append(current)
                    current = next
                }
            }
            mergedSegments.append(current)

            let midiPitch = UInt8(pitchIndex + midiBasePitch)
            for segment in mergedSegments {
                let startTime = Double(segment.startFrame) * result.frameDuration
                let duration = Double(segment.endFrame - segment.startFrame) * result.frameDuration
                let velocity = velocityFromPeakActivation(segment.peakActivation)

                events.append(
                    NoteEvent(
                        pitch: midiPitch,
                        startTime: startTime,
                        duration: duration,
                        velocity: velocity
                    )
                )
            }
        }

        events.sort {
            if $0.startTime == $1.startTime {
                return $0.pitch < $1.pitch
            }
            return $0.startTime < $1.startTime
        }

        return events
    }

    private static func velocityFromPeakActivation(_ peakActivation: Float) -> UInt8 {
        let scaled = Int(peakActivation.clamped(to: 0...1) * 127.0)
        return UInt8(max(1, min(127, scaled)))
    }
}

private struct RawSegment {
    var startFrame: Int
    var endFrame: Int
    var peakActivation: Float
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
