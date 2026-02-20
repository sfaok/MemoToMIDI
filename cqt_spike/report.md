# CQT Spike Report

## Inputs
- Audio source: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/synthetic_g_major.wav`

## Saved Artifacts
- `reference_audio`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_audio.npy`
- `reference_input`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_input.npy`
- `reference_note_output`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_note_output.npy`
- `reference_onset_output`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_onset_output.npy`
- `reference_contour_output`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_contour_output.npy`
- `reference_note_output_batched`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_note_output_batched.npy`
- `reference_onset_output_batched`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_onset_output_batched.npy`
- `reference_contour_output_batched`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_contour_output_batched.npy`
- `reference_midi`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_output.mid`
- `reference_note_events`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/reference_note_events.json`
- `metadata`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/metadata.json`
- `conversion_report`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/conversion_report.json`

## Tensor Shapes/Dtypes
- `reference_audio`: shape=[66150], dtype=float32, min=-0.900002, max=0.898824
- `reference_input`: shape=[2, 43844, 1], dtype=float32, min=-0.900002, max=0.898824
- `reference_note_output`: shape=[259, 88], dtype=float32, min=0.070002, max=0.776989
- `reference_onset_output`: shape=[259, 88], dtype=float32, min=0.032078, max=0.690241
- `reference_contour_output`: shape=[259, 264], dtype=float32, min=0.037773, max=0.583383
- `model_output_note`: shape=[259, 88], dtype=float32, min=0.070002, max=0.776989
- `model_output_onset`: shape=[259, 88], dtype=float32, min=0.032078, max=0.690241
- `model_output_contour`: shape=[259, 264], dtype=float32, min=0.037773, max=0.583383

## CoreML Conversion
- Status: `done`
- Attempt `tf`: `success`
  - Saved model: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/BasicPitch_from_tf.mlpackage`
  - Compare: `{'status': 'ok', 'max_abs_diff': {'note': 0.12656021118164062, 'contour': 0.2006065547466278, 'onset': 0.13552778959274292}}`
- Attempt `tflite`: `failed`
  - Error: `ValueError('Unable to determine the type of the model, i.e. the source framework. Please provide the value of argument "source", from one of ["tensorflow", "pytorch", "milinternal"]. Note that model conversion requires the source package that generates the model. Please make sure you have the appropriate version of source package installed. E.g., if you\'re converting model originally trained with TensorFlow 1.14, make sure you have `tensorflow==1.14` installed.')`
