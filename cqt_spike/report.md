# CQT Spike Report

## Inputs
- Audio source: `C:\Users\email\MemoToMIDI\cqt_spike\synthetic_g_major.wav`

## Saved Artifacts
- `reference_audio`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_audio.npy`
- `reference_input`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_input.npy`
- `reference_note_output`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_note_output.npy`
- `reference_onset_output`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_onset_output.npy`
- `reference_contour_output`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_contour_output.npy`
- `reference_note_output_batched`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_note_output_batched.npy`
- `reference_onset_output_batched`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_onset_output_batched.npy`
- `reference_contour_output_batched`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_contour_output_batched.npy`
- `reference_midi`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_output.mid`
- `reference_note_events`: `C:\Users\email\MemoToMIDI\cqt_spike\reference_note_events.json`
- `metadata`: `C:\Users\email\MemoToMIDI\cqt_spike\metadata.json`
- `conversion_report`: `C:\Users\email\MemoToMIDI\cqt_spike\conversion_report.json`

## Tensor Shapes/Dtypes
- `reference_audio`: shape=[66150], dtype=float32, min=-0.900002, max=0.898824
- `reference_input`: shape=[2, 43844, 1], dtype=float32, min=-0.900002, max=0.898824
- `reference_note_output`: shape=[259, 88], dtype=float32, min=0.069983, max=0.776506
- `reference_onset_output`: shape=[259, 88], dtype=float32, min=0.031994, max=0.690317
- `reference_contour_output`: shape=[259, 264], dtype=float32, min=0.037747, max=0.582807
- `model_output_note`: shape=[259, 88], dtype=float32, min=0.069983, max=0.776506
- `model_output_onset`: shape=[259, 88], dtype=float32, min=0.031994, max=0.690317
- `model_output_contour`: shape=[259, 264], dtype=float32, min=0.037747, max=0.582807

## CoreML Conversion
- Status: `done`
- Attempt `tf`: `failed`
  - Error: `RuntimeError('BlobWriter not loaded')`
- Attempt `tflite`: `failed`
  - Error: `ValueError('Unable to determine the type of the model, i.e. the source framework. Please provide the value of argument "source", from one of ["tensorflow", "pytorch", "milinternal"]. Note that model conversion requires the source package that generates the model. Please make sure you have the appropriate version of source package installed. E.g., if you\'re converting model originally trained with TensorFlow 1.14, make sure you have `tensorflow==1.14` installed.')`
