# Step 3 Diagnostics

## Inputs
- Audio: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/synthetic_g_major.wav`
- CoreML model: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/BasicPitch_from_tf.mlpackage`
- Overlap frames: `30`

## Check 1 - Identical Input Tensor Per Window
- Saved exact per-window model inputs:
- TensorFlow windows: `['/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_diag_tf_input_window_000.npy', '/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_diag_tf_input_window_001.npy']`
- TensorFlow stacked: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_diag_tf_input_all_windows.npy` shape=[2, 43844, 1]
- CoreML windows: `['/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_diag_coreml_input_window_000.npy', '/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_diag_coreml_input_window_001.npy']`
- CoreML stacked: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_diag_coreml_input_all_windows.npy` shape=[2, 43844, 1]

- Per-window input diffs (TF vs CoreML):
- window `0`: max_abs_diff=`0.000000000000`, mean_abs_diff=`0.000000000000`
- window `1`: max_abs_diff=`0.000000000000`, mean_abs_diff=`0.000000000000`
- Global max abs diff: `0.000000000000`
- Global mean abs diff: `0.000000000000`
- Check 1 verdict: `IDENTICAL (<1e-6)`

## Check 2 - Single Window, No Stitching
- Compared first raw output window only (`note` head), no overlap averaging/unwrap.
- Max abs diff: `0.126560211`
- Mean abs diff: `0.003850002`
- Frame-flip rates (% of active TF frames):
- threshold `0.3`: active_frames_tf=`154`, flipped_active_frames=`2`, frame_flip_rate_pct=`1.298701`
- threshold `0.5`: active_frames_tf=`150`, flipped_active_frames=`101`, frame_flip_rate_pct=`67.333333`
- threshold `0.7`: active_frames_tf=`9`, flipped_active_frames=`4`, frame_flip_rate_pct=`44.444444`

## Check 3 - TF Note Activation Histogram
- Source tensor: full aggregated TF note output (after normal Step 3 unwrapping).
- Histogram bins: `[0.0, 0.1), ..., [0.9, 1.0]`
| Bin | Count |
| --- | ---: |
| 0.0-0.1 | 15954 |
| 0.1-0.2 | 6052 |
| 0.2-0.3 | 16 |
| 0.3-0.4 | 22 |
| 0.4-0.5 | 180 |
| 0.5-0.6 | 145 |
| 0.6-0.7 | 379 |
| 0.7-0.8 | 44 |
| 0.8-0.9 | 0 |
| 0.9-1.0 | 0 |

- Total activations: `22792`
- Activations in 0.4-0.6: `325` (`1.425939%` of all activations)
- Active activations (>=0.5): `568`
- Active activations in 0.5-0.6: `145` (`25.528169%` of active activations)
