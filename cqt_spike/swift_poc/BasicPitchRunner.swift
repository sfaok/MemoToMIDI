import AVFoundation
import CoreML
import Foundation

enum RunnerError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case audioReadFailed(String)
    case modelIOFailed(String)
    case npyParseFailed(String)

    var description: String {
        switch self {
        case .invalidArguments(let message): return "Argument error: \(message)"
        case .audioReadFailed(let message): return "Audio error: \(message)"
        case .modelIOFailed(let message): return "Model I/O error: \(message)"
        case .npyParseFailed(let message): return "NPY parse error: \(message)"
        }
    }
}

struct Config {
    var audioPath: String = "cqt_spike/step3_reference_audio_22050_float32.wav"
    var modelPath: String = "cqt_spike/BasicPitch_from_tf.mlpackage"
    var refNotePath: String = "cqt_spike/step3_coreml_cpu_and_gpu_full_note.npy"
    var refOnsetPath: String = "cqt_spike/step3_coreml_cpu_and_gpu_full_onset.npy"
    var topKFrames: Int = 10
}

struct Matrix {
    let rows: Int
    let cols: Int
    let values: [Float]

    init(rows: Int, cols: Int, values: [Float]) {
        self.rows = rows
        self.cols = cols
        self.values = values
    }

    func row(_ index: Int) -> ArraySlice<Float> {
        let start = index * cols
        let end = start + cols
        return values[start..<end]
    }
}

struct NPYArray {
    let shape: [Int]
    let values: [Float]
}

struct DiffSummary {
    let alignedRows: Int
    let alignedCols: Int
    let maxAbsDiff: Float
    let meanAbsDiff: Float
    let topFrames: [(frame: Int, meanAbsDiff: Float)]
}

func printUsage() {
    print(
        """
        Usage:
          swift cqt_spike/swift_poc/BasicPitchRunner.swift [options]

        Options:
          --audio <path>       Input WAV path (must be 22050 Hz mono for now)
          --model <path>       CoreML model path (.mlpackage)
          --ref-note <path>    Reference stitched NOTE .npy
          --ref-onset <path>   Reference stitched ONSET .npy
          --top-k <int>        Top divergent frames to print if diff > 1e-4 (default: 10)
        """
    )
}

func parseArgs(_ args: [String]) throws -> Config {
    var config = Config()
    var i = 1
    while i < args.count {
        let arg = args[i]
        if arg == "--help" || arg == "-h" {
            printUsage()
            exit(0)
        }
        guard i + 1 < args.count else {
            throw RunnerError.invalidArguments("Missing value after \(arg)")
        }
        let value = args[i + 1]
        switch arg {
        case "--audio":
            config.audioPath = value
        case "--model":
            config.modelPath = value
        case "--ref-note":
            config.refNotePath = value
        case "--ref-onset":
            config.refOnsetPath = value
        case "--top-k":
            guard let parsed = Int(value), parsed > 0 else {
                throw RunnerError.invalidArguments("--top-k must be a positive integer")
            }
            config.topKFrames = parsed
        default:
            throw RunnerError.invalidArguments("Unknown argument: \(arg)")
        }
        i += 2
    }
    return config
}

func loadMonoWaveform(path: String, expectedSampleRate: Double) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat

    guard format.channelCount == 1 else {
        throw RunnerError.audioReadFailed("Expected mono WAV, got \(format.channelCount) channels")
    }
    guard abs(format.sampleRate - expectedSampleRate) < 0.01 else {
        throw RunnerError.audioReadFailed("Expected \(Int(expectedSampleRate)) Hz WAV, got \(format.sampleRate) Hz")
    }
    guard format.commonFormat == .pcmFormatFloat32 else {
        throw RunnerError.audioReadFailed(
            "Expected float32 PCM WAV for this PoC. Got format commonFormat=\(format.commonFormat.rawValue)"
        )
    }

    var samples: [Float] = []
    samples.reserveCapacity(Int(audioFile.length))

    while true {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audioFile.length)) else {
            throw RunnerError.audioReadFailed("Unable to allocate AVAudioPCMBuffer")
        }
        do {
            try audioFile.read(into: buffer)
        } catch {
            let nsError = error as NSError
            // AVAudioFile can throw this generic ObjC error at EOF after all frames are consumed.
            if nsError.domain == "Foundation._GenericObjCError", nsError.code == 0 {
                break
            }
            throw RunnerError.audioReadFailed("Failed while reading WAV: \(nsError)")
        }
        let frameCount = Int(buffer.frameLength)
        if frameCount == 0 { break }

        guard let channels = buffer.floatChannelData else {
            throw RunnerError.audioReadFailed("No float channel data available")
        }
        let channel = channels[0]
        samples.append(contentsOf: UnsafeBufferPointer(start: channel, count: frameCount))
    }
    return samples
}

func makeWindows(audio: [Float], prependZeros: Int, windowSize: Int, hopSize: Int) -> [[Float]] {
    var padded = [Float](repeating: 0, count: prependZeros)
    padded.append(contentsOf: audio)

    var windows: [[Float]] = []
    var start = 0
    while start < padded.count {
        let end = min(start + windowSize, padded.count)
        var window = Array(padded[start..<end])
        if window.count < windowSize {
            window.append(contentsOf: [Float](repeating: 0, count: windowSize - window.count))
        }
        windows.append(window)
        start += hopSize
    }
    return windows
}

func makeInput(window: [Float]) throws -> MLMultiArray {
    let input = try MLMultiArray(shape: [1, NSNumber(value: window.count), 1], dataType: .float32)
    let shape = input.shape.map { Int(truncating: $0) }
    guard shape == [1, window.count, 1] else {
        throw RunnerError.modelIOFailed("Unexpected MLMultiArray input shape allocation: \(shape)")
    }

    let strides = input.strides.map { Int(truncating: $0) }
    guard strides.count == 3 else {
        throw RunnerError.modelIOFailed("Unexpected input strides count: \(strides.count)")
    }

    let ptr = input.dataPointer.bindMemory(to: Float32.self, capacity: input.count)
    for i in 0..<input.count {
        ptr[i] = 0
    }

    for i in 0..<window.count {
        let offset = i * strides[1]
        ptr[offset] = window[i]
    }
    return input
}

func multiArrayToBatch0Matrix(_ array: MLMultiArray) throws -> Matrix {
    let shape = array.shape.map { Int(truncating: $0) }
    let strides = array.strides.map { Int(truncating: $0) }

    guard shape.count == 3, strides.count == 3 else {
        throw RunnerError.modelIOFailed("Expected 3D MLMultiArray, got shape=\(shape) strides=\(strides)")
    }
    guard shape[0] >= 1 else {
        throw RunnerError.modelIOFailed("Expected non-empty batch dimension, got shape=\(shape)")
    }

    let time = shape[1]
    let freq = shape[2]
    var values = [Float](repeating: 0, count: time * freq)

    switch array.dataType {
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        for t in 0..<time {
            for f in 0..<freq {
                let srcOffset = t * strides[1] + f * strides[2]
                values[t * freq + f] = ptr[srcOffset]
            }
        }
    case .double:
        let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        for t in 0..<time {
            for f in 0..<freq {
                let srcOffset = t * strides[1] + f * strides[2]
                values[t * freq + f] = Float(ptr[srcOffset])
            }
        }
    default:
        throw RunnerError.modelIOFailed("Unsupported MLMultiArray data type: \(array.dataType.rawValue)")
    }

    return Matrix(rows: time, cols: freq, values: values)
}

func mapNoteAndOnset(from prediction: MLFeatureProvider) throws -> (note: Matrix, onset: Matrix) {
    var note: Matrix?
    var onset: Matrix?
    var fallback88: [Matrix] = []

    for key in prediction.featureNames.sorted() {
        guard let feature = prediction.featureValue(for: key), let array = feature.multiArrayValue else {
            continue
        }
        let matrix = try multiArrayToBatch0Matrix(array)
        let lower = key.lowercased()

        if lower.contains("note") || key == "Identity_1" {
            note = matrix
            continue
        }
        if lower.contains("onset") || key == "Identity_2" {
            onset = matrix
            continue
        }
        if matrix.cols == 88 {
            fallback88.append(matrix)
        }
    }

    if note == nil || onset == nil {
        if fallback88.count == 2 {
            note = note ?? fallback88[0]
            onset = onset ?? fallback88[1]
        }
    }

    guard let noteOut = note, let onsetOut = onset else {
        throw RunnerError.modelIOFailed("Failed to map note/onset outputs from prediction keys: \(prediction.featureNames)")
    }
    return (noteOut, onsetOut)
}

func unwrapOutput(
    windows: [Matrix],
    audioOriginalLengthSamples: Int,
    overlapFrames: Int,
    hopSizeSamples: Int,
    annotationsFPS: Int,
    audioWindowLengthSeconds: Int
) throws -> Matrix {
    guard let first = windows.first else {
        throw RunnerError.modelIOFailed("No model outputs to unwrap")
    }

    let nOverlapHalf = overlapFrames / 2
    let trimmedRowsPerWindow = first.rows - (2 * nOverlapHalf)
    guard trimmedRowsPerWindow > 0 else {
        throw RunnerError.modelIOFailed(
            "Invalid overlap trim. rows=\(first.rows), overlapFrames=\(overlapFrames), half=\(nOverlapHalf)"
        )
    }

    let cols = first.cols
    var concatenated = [Float]()
    concatenated.reserveCapacity(windows.count * trimmedRowsPerWindow * cols)

    for window in windows {
        guard window.rows == first.rows, window.cols == cols else {
            throw RunnerError.modelIOFailed("Inconsistent output window shape while unwrapping")
        }
        for r in nOverlapHalf..<(window.rows - nOverlapHalf) {
            concatenated.append(contentsOf: window.row(r))
        }
    }

    let totalRows = concatenated.count / cols
    let expectedWindows = Float(audioOriginalLengthSamples) / Float(hopSizeSamples)
    let framesPerWindow = (audioWindowLengthSeconds * annotationsFPS) - overlapFrames
    let expectedRows = Int(expectedWindows * Float(framesPerWindow))
    let finalRows = min(totalRows, max(0, expectedRows))

    if finalRows == totalRows {
        return Matrix(rows: totalRows, cols: cols, values: concatenated)
    }
    let trimmedCount = finalRows * cols
    return Matrix(rows: finalRows, cols: cols, values: Array(concatenated.prefix(trimmedCount)))
}

func matrixStats(_ matrix: Matrix) -> (max: Float, mean: Float) {
    if matrix.values.isEmpty { return (0, 0) }
    var maxValue = -Float.greatestFiniteMagnitude
    var sum: Float = 0
    for value in matrix.values {
        if value > maxValue { maxValue = value }
        sum += value
    }
    return (maxValue, sum / Float(matrix.values.count))
}

func regexCapture(_ pattern: String, in text: String) -> String? {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return nil }
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    guard let match = regex.firstMatch(in: text, options: [], range: range), match.numberOfRanges > 1 else {
        return nil
    }
    guard let captureRange = Range(match.range(at: 1), in: text) else { return nil }
    return String(text[captureRange])
}

func loadNpyFloat32(path: String) throws -> NPYArray {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    let bytes = [UInt8](data)

    guard bytes.count >= 10 else {
        throw RunnerError.npyParseFailed("File too small to be a valid NPY: \(path)")
    }

    let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]
    guard Array(bytes[0..<6]) == magic else {
        throw RunnerError.npyParseFailed("Missing NPY magic bytes in \(path)")
    }

    let major = Int(bytes[6])
    let minor = Int(bytes[7])
    var cursor = 8
    let headerLength: Int
    if major == 1 {
        guard bytes.count >= 10 else {
            throw RunnerError.npyParseFailed("NPY v1 header truncated in \(path)")
        }
        headerLength = Int(bytes[cursor]) | (Int(bytes[cursor + 1]) << 8)
        cursor += 2
    } else if major == 2 || major == 3 {
        guard bytes.count >= 12 else {
            throw RunnerError.npyParseFailed("NPY v2/v3 header truncated in \(path)")
        }
        headerLength = Int(bytes[cursor])
            | (Int(bytes[cursor + 1]) << 8)
            | (Int(bytes[cursor + 2]) << 16)
            | (Int(bytes[cursor + 3]) << 24)
        cursor += 4
    } else {
        throw RunnerError.npyParseFailed("Unsupported NPY version \(major).\(minor)")
    }

    guard bytes.count >= cursor + headerLength else {
        throw RunnerError.npyParseFailed("Header length exceeds file bounds in \(path)")
    }
    let headerBytes = bytes[cursor..<(cursor + headerLength)]
    cursor += headerLength

    guard let header = String(bytes: headerBytes, encoding: .ascii) else {
        throw RunnerError.npyParseFailed("Failed to parse NPY header as ASCII in \(path)")
    }

    guard let descr = regexCapture("'descr'\\s*:\\s*'([^']+)'", in: header) else {
        throw RunnerError.npyParseFailed("Missing 'descr' in NPY header for \(path)")
    }
    guard descr == "<f4" || descr == "|f4" else {
        throw RunnerError.npyParseFailed("Only float32 NPY files are supported. Found descr=\(descr) in \(path)")
    }

    guard let fortranOrder = regexCapture("'fortran_order'\\s*:\\s*(True|False)", in: header) else {
        throw RunnerError.npyParseFailed("Missing 'fortran_order' in NPY header for \(path)")
    }
    guard fortranOrder == "False" else {
        throw RunnerError.npyParseFailed("Fortran-order arrays are not supported for \(path)")
    }

    guard let shapeText = regexCapture("'shape'\\s*:\\s*\\(([^\\)]*)\\)", in: header) else {
        throw RunnerError.npyParseFailed("Missing 'shape' in NPY header for \(path)")
    }
    let dims = shapeText
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    let shape = try dims.map {
        guard let v = Int($0), v >= 0 else {
            throw RunnerError.npyParseFailed("Invalid shape value '\($0)' in \(path)")
        }
        return v
    }
    guard !shape.isEmpty else {
        throw RunnerError.npyParseFailed("Parsed empty shape in \(path)")
    }

    let elementCount = shape.reduce(1, *)
    let expectedBytes = elementCount * MemoryLayout<Float32>.size
    guard bytes.count >= cursor + expectedBytes else {
        throw RunnerError.npyParseFailed(
            "NPY payload truncated in \(path). expected=\(expectedBytes) bytes, found=\(bytes.count - cursor)"
        )
    }

    var values = [Float]()
    values.reserveCapacity(elementCount)
    for i in 0..<elementCount {
        let o = cursor + (i * 4)
        let bits = UInt32(bytes[o])
            | (UInt32(bytes[o + 1]) << 8)
            | (UInt32(bytes[o + 2]) << 16)
            | (UInt32(bytes[o + 3]) << 24)
        values.append(Float(bitPattern: bits))
    }

    return NPYArray(shape: shape, values: values)
}

func matrixFromNpy(_ npy: NPYArray, path: String) throws -> Matrix {
    guard npy.shape.count == 2 else {
        throw RunnerError.npyParseFailed("Expected 2D array in \(path), got shape=\(npy.shape)")
    }
    return Matrix(rows: npy.shape[0], cols: npy.shape[1], values: npy.values)
}

func diffSummary(actual: Matrix, reference: Matrix, topK: Int) -> DiffSummary {
    let rows = min(actual.rows, reference.rows)
    let cols = min(actual.cols, reference.cols)
    if rows == 0 || cols == 0 {
        return DiffSummary(alignedRows: rows, alignedCols: cols, maxAbsDiff: 0, meanAbsDiff: 0, topFrames: [])
    }

    var maxAbs: Float = 0
    var sumAbs: Float = 0
    var frameMeans = [Float](repeating: 0, count: rows)

    for r in 0..<rows {
        var rowSum: Float = 0
        let aStart = r * actual.cols
        let bStart = r * reference.cols
        for c in 0..<cols {
            let diff = abs(actual.values[aStart + c] - reference.values[bStart + c])
            if diff > maxAbs { maxAbs = diff }
            rowSum += diff
            sumAbs += diff
        }
        frameMeans[r] = rowSum / Float(cols)
    }

    let meanAbs = sumAbs / Float(rows * cols)
    let top = frameMeans.enumerated()
        .sorted { $0.element > $1.element }
        .prefix(topK)
        .map { (frame: $0.offset, meanAbsDiff: $0.element) }

    return DiffSummary(alignedRows: rows, alignedCols: cols, maxAbsDiff: maxAbs, meanAbsDiff: meanAbs, topFrames: top)
}

func run() throws {
    let config = try parseArgs(CommandLine.arguments)

    let audioSampleRate = 22050
    let windowSize = 43844
    let fftHop = 256
    let annotationsFPS = audioSampleRate / fftHop
    let overlapFrames = 30
    let overlapLenSamples = overlapFrames * fftHop
    let prependSamples = overlapLenSamples / 2
    let hopSizeSamples = windowSize - overlapLenSamples

    print("Window constants (from basic_pitch inference.py):")
    print("  overlap_frames=\(overlapFrames), fft_hop=\(fftHop), overlap_len_samples=\(overlapLenSamples)")
    print("  prepend_zeros=\(prependSamples), window_size=\(windowSize), hop_size=\(hopSizeSamples)")

    let waveform = try loadMonoWaveform(path: config.audioPath, expectedSampleRate: Double(audioSampleRate))
    let windows = makeWindows(audio: waveform, prependZeros: prependSamples, windowSize: windowSize, hopSize: hopSizeSamples)

    let modelURL = URL(fileURLWithPath: config.modelPath)
    let modelConfig = MLModelConfiguration()
    modelConfig.computeUnits = .cpuAndGPU
    let compiledModelURL = try MLModel.compileModel(at: modelURL)
    let model = try MLModel(contentsOf: compiledModelURL, configuration: modelConfig)

    guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first else {
        throw RunnerError.modelIOFailed("Model has no input descriptions")
    }

    let startTime = Date()
    var noteWindows: [Matrix] = []
    var onsetWindows: [Matrix] = []
    noteWindows.reserveCapacity(windows.count)
    onsetWindows.reserveCapacity(windows.count)

    for window in windows {
        let inputArray = try makeInput(window: window)
        let features = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: inputArray),
        ])
        let prediction = try model.prediction(from: features)
        let mapped = try mapNoteAndOnset(from: prediction)
        noteWindows.append(mapped.note)
        onsetWindows.append(mapped.onset)
    }

    let stitchedNote = try unwrapOutput(
        windows: noteWindows,
        audioOriginalLengthSamples: waveform.count,
        overlapFrames: overlapFrames,
        hopSizeSamples: hopSizeSamples,
        annotationsFPS: annotationsFPS,
        audioWindowLengthSeconds: 2
    )
    let stitchedOnset = try unwrapOutput(
        windows: onsetWindows,
        audioOriginalLengthSamples: waveform.count,
        overlapFrames: overlapFrames,
        hopSizeSamples: hopSizeSamples,
        annotationsFPS: annotationsFPS,
        audioWindowLengthSeconds: 2
    )

    let elapsed = Date().timeIntervalSince(startTime)
    let noteStats = matrixStats(stitchedNote)
    let onsetStats = matrixStats(stitchedOnset)

    print("")
    print("Inference summary:")
    print("  windows_processed=\(windows.count)")
    print("  note_output_shape=\(stitchedNote.rows)x\(stitchedNote.cols)")
    print(String(format: "  note_max=%.9f note_mean=%.9f", noteStats.max, noteStats.mean))
    print(String(format: "  onset_max=%.9f onset_mean=%.9f", onsetStats.max, onsetStats.mean))
    print(String(format: "  processing_time_sec=%.4f", elapsed))

    let noteRef = try matrixFromNpy(try loadNpyFloat32(path: config.refNotePath), path: config.refNotePath)
    let onsetRef = try matrixFromNpy(try loadNpyFloat32(path: config.refOnsetPath), path: config.refOnsetPath)
    let noteDiff = diffSummary(actual: stitchedNote, reference: noteRef, topK: config.topKFrames)
    let onsetDiff = diffSummary(actual: stitchedOnset, reference: onsetRef, topK: config.topKFrames)

    print("")
    print("Reference diff:")
    print(
        String(
            format: "  note max_abs_diff=%.9f mean_abs_diff=%.9f aligned_shape=%dx%d",
            noteDiff.maxAbsDiff,
            noteDiff.meanAbsDiff,
            noteDiff.alignedRows,
            noteDiff.alignedCols
        )
    )
    print(
        String(
            format: "  onset max_abs_diff=%.9f mean_abs_diff=%.9f aligned_shape=%dx%d",
            onsetDiff.maxAbsDiff,
            onsetDiff.meanAbsDiff,
            onsetDiff.alignedRows,
            onsetDiff.alignedCols
        )
    )

    let threshold: Float = 1e-4
    if noteDiff.maxAbsDiff > threshold {
        print("  NOTE diverges above \(threshold). Top frames by mean abs diff:")
        for item in noteDiff.topFrames {
            print(String(format: "    frame=%d mean_abs_diff=%.9f", item.frame, item.meanAbsDiff))
        }
    }
    if onsetDiff.maxAbsDiff > threshold {
        print("  ONSET diverges above \(threshold). Top frames by mean abs diff:")
        for item in onsetDiff.topFrames {
            print(String(format: "    frame=%d mean_abs_diff=%.9f", item.frame, item.meanAbsDiff))
        }
    }
}

do {
    try run()
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
