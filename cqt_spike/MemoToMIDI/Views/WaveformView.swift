import SwiftUI

struct WaveformView: View {
    let samples: [Float]

    var body: some View {
        GeometryReader { geometry in
            if samples.isEmpty {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.secondary.opacity(0.15))
                    .overlay(
                        Text("Waveform")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    )
            } else {
                HStack(alignment: .center, spacing: 2) {
                    ForEach(Array(downsampledSamples(maxBars: 120).enumerated()), id: \.offset) { _, sample in
                        Capsule(style: .continuous)
                            .fill(Color.accentColor.opacity(0.85))
                            .frame(width: 2, height: max(4, CGFloat(sample) * geometry.size.height))
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .padding(.horizontal, 8)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.secondary.opacity(0.15))
                )
            }
        }
    }

    private func downsampledSamples(maxBars: Int) -> [Float] {
        let clampedMaxBars = max(1, maxBars)
        guard samples.count > clampedMaxBars else { return samples }
        let stride = max(1, samples.count / clampedMaxBars)
        var result: [Float] = []
        result.reserveCapacity(clampedMaxBars)

        var index = 0
        while index < samples.count {
            let sliceEnd = min(samples.count, index + stride)
            let chunk = samples[index..<sliceEnd]
            let average = chunk.reduce(Float(0), +) / Float(chunk.count)
            result.append(average)
            index += stride
        }
        return result
    }
}
