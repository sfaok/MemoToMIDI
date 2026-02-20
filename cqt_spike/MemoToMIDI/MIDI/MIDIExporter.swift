import SwiftUI
import UIKit

enum MIDIExporterError: LocalizedError {
    case failedToWriteFile(underlying: Error)
    case fileUnavailable(path: String)

    var errorDescription: String? {
        switch self {
        case .failedToWriteFile(let underlying):
            return "Failed to write MIDI file: \(underlying.localizedDescription)"
        case .fileUnavailable(let path):
            return "MIDI file was not available at expected path: \(path)"
        }
    }
}

struct MIDIExporter {
    /// Write MIDI data to a temporary file and return the URL.
    /// Filename format: MemoToMIDI_YYYYMMDD_HHmmss.mid
    static func exportToFile(notes: [NoteEvent], bpm: Double) throws -> URL {
        let midiData = MIDIFileWriter.write(notes: notes, bpm: bpm)
        let timestamp = timestampFormatter.string(from: Date())
        let filename = "MemoToMIDI_\(timestamp).mid"
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

        do {
            if FileManager.default.fileExists(atPath: fileURL.path) {
                try FileManager.default.removeItem(at: fileURL)
            }
            try midiData.write(to: fileURL, options: .atomic)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                throw MIDIExporterError.fileUnavailable(path: fileURL.path)
            }
            return fileURL
        } catch let exporterError as MIDIExporterError {
            throw exporterError
        } catch {
            throw MIDIExporterError.failedToWriteFile(underlying: error)
        }
    }

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
}

struct MIDIShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
        // No-op. UIActivityViewController does not require live updates.
    }
}

struct MIDIExportShareButton<Label: View>: View {
    private struct ShareItem: Identifiable {
        let url: URL
        var id: URL { url }
    }

    let notes: [NoteEvent]
    let bpm: Double
    let label: () -> Label

    @State private var shareItem: ShareItem?
    @State private var exportError: MIDIExporterError?

    init(
        notes: [NoteEvent],
        bpm: Double = MIDIConstants.defaultTempo,
        @ViewBuilder label: @escaping () -> Label
    ) {
        self.notes = notes
        self.bpm = bpm
        self.label = label
    }

    var body: some View {
        Button(action: exportAndShare) {
            label()
        }
        .sheet(item: $shareItem, onDismiss: {
            shareItem = nil
        }) { item in
            MIDIShareSheet(items: [item.url])
        }
        .alert(
            "MIDI Export Failed",
            isPresented: Binding(
                get: { exportError != nil },
                set: { shouldShow in
                    if !shouldShow {
                        exportError = nil
                    }
                }
            ),
            actions: {
                Button("OK", role: .cancel) {
                    exportError = nil
                }
            },
            message: {
                Text(exportError?.errorDescription ?? "Unknown error")
            }
        )
    }

    private func exportAndShare() {
        do {
            let exportedFileURL = try MIDIExporter.exportToFile(notes: notes, bpm: bpm)
            guard FileManager.default.fileExists(atPath: exportedFileURL.path) else {
                throw MIDIExporterError.fileUnavailable(path: exportedFileURL.path)
            }
            shareItem = ShareItem(url: exportedFileURL)
        } catch let exporterError as MIDIExporterError {
            exportError = exporterError
        } catch {
            exportError = .failedToWriteFile(underlying: error)
        }
    }
}

extension MIDIExportShareButton where Label == Text {
    init(notes: [NoteEvent], bpm: Double = MIDIConstants.defaultTempo) {
        self.init(notes: notes, bpm: bpm) {
            Text("Export MIDI")
        }
    }
}
