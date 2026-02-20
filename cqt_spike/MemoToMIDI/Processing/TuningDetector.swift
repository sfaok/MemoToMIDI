import Foundation

struct TuningResult {
    let centsOffset: Double
    let referenceFrequency: Double
    let confidence: Double
}

struct TuningDetector {
    private static let strongActivationThreshold: Float = 0.5
    private static let interpolationStabilityThreshold: Double = 0.01

    static func detect(from result: InferenceResult) -> TuningResult {
        let frameCount = result.noteActivations.count
        guard frameCount > 0 else {
            return TuningResult(centsOffset: 0, referenceFrequency: 440.0, confidence: 0)
        }

        var samples: [Double] = []
        samples.reserveCapacity(frameCount)

        for frame in result.noteActivations {
            let pitchCount = frame.count
            guard pitchCount >= 3 else { continue }

            for pitchIndex in 1..<(pitchCount - 1) {
                let center = frame[pitchIndex]
                guard center > strongActivationThreshold else { continue }

                let left = frame[pitchIndex - 1]
                let right = frame[pitchIndex + 1]

                let denominator = Double(2.0 * center - left - right)
                guard denominator.isFinite, abs(denominator) >= interpolationStabilityThreshold else {
                    continue
                }

                let numerator = Double(0.5 * (right - left))
                let rawShift = numerator / denominator
                guard rawShift.isFinite, abs(rawShift) <= 0.5 else { continue }

                let clampedShift = min(max(rawShift, -0.5), 0.5)
                let cents = clampedShift * 100.0
                guard cents.isFinite else { continue }

                samples.append(cents)
            }
        }

        guard !samples.isEmpty else {
            return TuningResult(centsOffset: 0, referenceFrequency: 440.0, confidence: 0)
        }

        samples.sort()

        let medianCents = median(ofSorted: samples)
        let clampedCents = min(max(medianCents, -50.0), 50.0)

        let confidence: Double
        if samples.count < 10 {
            confidence = 0
        } else {
            let q1 = percentile(ofSorted: samples, p: 0.25)
            let q3 = percentile(ofSorted: samples, p: 0.75)
            let iqr = q3 - q1
            confidence = min(1.0, max(0.0, 1.0 - (iqr / 30.0)))
        }

        let referenceFrequency = 440.0 * pow(2.0, clampedCents / 1200.0)

        return TuningResult(
            centsOffset: clampedCents,
            referenceFrequency: referenceFrequency,
            confidence: confidence
        )
    }

    private static func median(ofSorted values: [Double]) -> Double {
        let count = values.count
        let midpoint = count / 2

        if count % 2 == 0 {
            return (values[midpoint - 1] + values[midpoint]) / 2.0
        }
        return values[midpoint]
    }

    private static func percentile(ofSorted values: [Double], p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        if values.count == 1 { return values[0] }

        let clampedP = min(max(p, 0.0), 1.0)
        let position = clampedP * Double(values.count - 1)
        let lowerIndex = Int(floor(position))
        let upperIndex = min(values.count - 1, lowerIndex + 1)

        if lowerIndex == upperIndex {
            return values[lowerIndex]
        }

        let fraction = position - Double(lowerIndex)
        return values[lowerIndex] * (1.0 - fraction) + values[upperIndex] * fraction
    }
}
