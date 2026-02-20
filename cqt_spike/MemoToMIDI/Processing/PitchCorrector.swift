import Foundation

struct PitchCorrector {
    static func correct(
        result: InferenceResult,
        centsOffset: Double
    ) -> InferenceResult {
        guard centsOffset != 0, centsOffset.isFinite else {
            return result
        }

        let shiftBins = -centsOffset / 100.0

        return InferenceResult(
            noteActivations: shift(matrix: result.noteActivations, shiftBins: shiftBins),
            onsetActivations: shift(matrix: result.onsetActivations, shiftBins: shiftBins),
            frameDuration: result.frameDuration
        )
    }

    private static func shift(matrix: [[Float]], shiftBins: Double) -> [[Float]] {
        guard !matrix.isEmpty else { return matrix }

        var shiftedMatrix: [[Float]] = []
        shiftedMatrix.reserveCapacity(matrix.count)

        for row in matrix {
            let pitchCount = row.count
            guard pitchCount > 0 else {
                shiftedMatrix.append([])
                continue
            }

            var shiftedRow = [Float](repeating: 0, count: pitchCount)

            for outputPitch in 0..<pitchCount {
                let sourcePosition = Double(outputPitch) - shiftBins
                let sourceLow = Int(floor(sourcePosition))
                let sourceHigh = sourceLow + 1
                let fraction = Float(sourcePosition - Double(sourceLow))

                let lowValue: Float
                if sourceLow >= 0 && sourceLow < pitchCount {
                    lowValue = row[sourceLow]
                } else {
                    lowValue = 0
                }

                let highValue: Float
                if sourceHigh >= 0 && sourceHigh < pitchCount {
                    highValue = row[sourceHigh]
                } else {
                    highValue = 0
                }

                shiftedRow[outputPitch] = lowValue * (1 - fraction) + highValue * fraction
            }

            shiftedMatrix.append(shiftedRow)
        }

        return shiftedMatrix
    }
}
