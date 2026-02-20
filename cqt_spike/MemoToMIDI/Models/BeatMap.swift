import Foundation

struct BeatMap: Codable {
    var taps: [Double]
    var timeSignature: Int = 4

    /// BPM derived from median inter-tap interval. Returns nil if fewer than 2 taps.
    var bpm: Double? {
        guard let medianInterval else { return nil }
        guard medianInterval > 0 else { return nil }
        return 60.0 / medianInterval
    }

    /// Number of complete bars based on tap count and time signature.
    var barCount: Int {
        guard timeSignature > 0 else { return 0 }
        return taps.count / timeSignature
    }

    /// Returns the nearest beat position (from taps) to a given time.
    func nearestBeat(to time: Double) -> Double? {
        guard !taps.isEmpty else { return nil }
        return taps.min { abs($0 - time) < abs($1 - time) }
    }

    /// Returns all beat positions, optionally extended/interpolated to cover
    /// the full recording duration. If the user stopped tapping before the
    /// recording ended, extrapolate using the median interval.
    func allBeats(throughDuration duration: Double) -> [Double] {
        guard duration.isFinite else {
            return taps.sorted()
        }

        let safeDuration = max(0, duration)
        var beats = taps.sorted()
        guard !beats.isEmpty else { return beats }

        if let medianInterval, medianInterval > 0 {
            var nextBeat = beats[beats.count - 1] + medianInterval
            while nextBeat <= safeDuration {
                beats.append(nextBeat)
                nextBeat += medianInterval
            }
        }

        return beats.filter { $0 <= safeDuration }
    }

    private var medianInterval: Double? {
        let sortedTaps = taps.sorted()
        guard sortedTaps.count >= 2 else { return nil }

        var intervals: [Double] = []
        intervals.reserveCapacity(sortedTaps.count - 1)
        for index in 1..<sortedTaps.count {
            let delta = sortedTaps[index] - sortedTaps[index - 1]
            if delta > 0 {
                intervals.append(delta)
            }
        }

        guard !intervals.isEmpty else { return nil }
        intervals.sort()

        let middle = intervals.count / 2
        if intervals.count.isMultiple(of: 2) {
            return (intervals[middle - 1] + intervals[middle]) / 2
        }
        return intervals[middle]
    }
}
