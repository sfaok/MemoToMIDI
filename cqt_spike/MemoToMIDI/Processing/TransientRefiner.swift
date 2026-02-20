import Accelerate
import Foundation

struct TransientRefiner {
    /// Refines note start times by snapping to the nearest audio transient.
    /// Returns a new array with adjusted startTime values. Notes are not
    /// added, removed, or reordered — only startTime is modified.
    static func refine(
        notes: [NoteEvent],
        audioBuffer: [Float],
        sampleRate: Double = AudioConstants.sampleRate
    ) -> [NoteEvent] {
        guard !notes.isEmpty, !audioBuffer.isEmpty, sampleRate > 0 else { return notes }

        let hopSamples = 256 // Basic Pitch frame hop at 22050 Hz
        let frameWindowSec = Double(hopSamples) / sampleRate
        let detection = transientDetection(audioBuffer: audioBuffer, windowSamples: hopSamples, hopSamples: hopSamples)
        let peaks = pickPeaks(detection: detection, audioBuffer: audioBuffer, hopSamples: hopSamples, sampleRate: sampleRate)
        guard !peaks.isEmpty else { return notes }

        var refined = notes
        let sortedIndices = notes.indices.sorted { notes[$0].startTime < notes[$1].startTime }
        var cursor = 0

        while cursor < sortedIndices.count {
            var cluster: [Int] = []
            let firstIndex = sortedIndices[cursor]
            let clusterStart = notes[firstIndex].startTime
            cluster.append(firstIndex)
            cursor += 1

            while cursor < sortedIndices.count {
                let nextIndex = sortedIndices[cursor]
                if notes[nextIndex].startTime - clusterStart > frameWindowSec { break }
                cluster.append(nextIndex)
                cursor += 1
            }

            let center = cluster.reduce(0.0) { $0 + notes[$1].startTime } / Double(cluster.count)
            let candidates = peaks.filter { abs($0.time - center) <= frameWindowSec }
            guard let bestPeak = candidates.max(by: { $0.value < $1.value }) else { continue }

            for index in cluster {
                let endTime = notes[index].startTime + notes[index].duration
                refined[index].startTime = bestPeak.time
                refined[index].duration = max(0.01, endTime - bestPeak.time)
            }
        }

#if DEBUG
        let refinedNotes = refined
        for (original, refined) in zip(notes, refinedNotes) {
            let deltaMs = (refined.startTime - original.startTime) * 1000.0
            if abs(deltaMs) > 0.1 {
                let noteName = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][Int(refined.pitch) % 12]
                let octave = Int(refined.pitch) / 12 - 1
                print("TR: \(noteName)\(octave) shifted \(String(format: "%+.1f", deltaMs))ms")
            }
        }
        let shifted = zip(notes, refinedNotes).filter { abs($0.startTime - $1.startTime) > 0.0001 }.count
        print("TR: \(shifted)/\(notes.count) notes refined")
#endif

        return refined
    }

    private static func transientDetection(audioBuffer: [Float], windowSamples: Int, hopSamples: Int) -> [Float] {
        guard audioBuffer.count >= windowSamples else { return [] }
        let frameCount = (audioBuffer.count - windowSamples) / hopSamples + 1
        guard frameCount > 1 else { return [] }

        var energy = [Float](repeating: 0, count: frameCount)
        audioBuffer.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            for frame in 0..<frameCount {
                var meanSquare: Float = 0
                vDSP_measqv(base + frame * hopSamples, 1, &meanSquare, vDSP_Length(windowSamples))
                energy[frame] = meanSquare
            }
        }

        var detection = [Float](repeating: 0, count: frameCount)
        for i in 1..<frameCount { detection[i] = max(0, energy[i] - energy[i - 1]) }
        return detection
    }

    private static func pickPeaks(
        detection: [Float],
        audioBuffer: [Float],
        hopSamples: Int,
        sampleRate: Double
    ) -> [(time: Double, value: Float)] {
        guard detection.count > 2 else { return [] }
        let threshold = max(median(detection) * 1.5, 1e-8)
        let minPeakFrames = max(1, Int(ceil((0.03 * sampleRate) / Double(hopSamples))))
        let searchRadius = max(1, hopSamples / 2)
        var peaks: [(frame: Int, value: Float)] = []

        for i in 1..<(detection.count - 1) where detection[i] >= threshold && detection[i] >= detection[i - 1] && detection[i] > detection[i + 1] {
            if let last = peaks.last, i - last.frame < minPeakFrames {
                if detection[i] > last.value { peaks[peaks.count - 1] = (i, detection[i]) }
            } else {
                peaks.append((i, detection[i]))
            }
        }

        return peaks.map { peak in
            let approxSample = peak.frame * hopSamples
            let lo = max(0, approxSample - searchRadius)
            let hi = min(audioBuffer.count - 1, approxSample + searchRadius)
            var bestSample = lo
            var bestMagnitude = abs(audioBuffer[lo])
            if hi > lo {
                for s in (lo + 1)...hi where abs(audioBuffer[s]) > bestMagnitude {
                    bestMagnitude = abs(audioBuffer[s])
                    bestSample = s
                }
            }
            return (Double(bestSample) / sampleRate, peak.value)
        }
    }

    private static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[mid - 1] + sorted[mid]) * 0.5 : sorted[mid]
    }
}
