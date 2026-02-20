import SwiftUI

struct CleanupSlidersView: View {
    @Binding var parameters: ExtractionParameters
    @Binding var isExpanded: Bool

    var body: some View {
        VStack(spacing: 12) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)

                    Text("Cleanup Sliders")
                        .font(.headline)
                        .foregroundStyle(.primary)

                    Spacer()

                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.secondary.opacity(0.4))
                        .frame(width: 36, height: 4)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(spacing: 14) {
                    sliderRow(
                        title: "Sensitivity",
                        valueText: String(format: "%.2f", parameters.onsetThreshold)
                    ) {
                        Slider(
                            value: Binding(
                                get: { Double(parameters.onsetThreshold) },
                                set: { parameters.onsetThreshold = Float($0) }
                            ),
                            in: 0.2...0.8
                        )
                    } footer: {
                        HStack {
                            Text("More Notes")
                            Spacer()
                            Text("Fewer Notes")
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }

                    sliderRow(
                        title: "Min Length",
                        valueText: "\(Int(parameters.minNoteLengthMs.rounded())) ms"
                    ) {
                        Slider(
                            value: $parameters.minNoteLengthMs,
                            in: 20...500,
                            step: 1
                        )
                    }

                    sliderRow(
                        title: "Merge Gap",
                        valueText: "\(Int(parameters.mergeGapMs.rounded())) ms"
                    ) {
                        Slider(
                            value: $parameters.mergeGapMs,
                            in: 0...200,
                            step: 1
                        )
                    }

                    sliderRow(
                        title: "Note Sustain",
                        valueText: String(format: "%.2f", parameters.frameThreshold)
                    ) {
                        Slider(
                            value: Binding(
                                get: { Double(parameters.frameThreshold) },
                                set: { parameters.frameThreshold = Float($0) }
                            ),
                            in: 0.1...0.7
                        )
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    @ViewBuilder
    private func sliderRow<Control: View, Footer: View>(
        title: String,
        valueText: String,
        @ViewBuilder control: () -> Control,
        @ViewBuilder footer: () -> Footer
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text(valueText)
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            control()
            footer()
        }
    }

    @ViewBuilder
    private func sliderRow<Control: View>(
        title: String,
        valueText: String,
        @ViewBuilder control: () -> Control
    ) -> some View {
        sliderRow(title: title, valueText: valueText, control: control) {
            EmptyView()
        }
    }
}
