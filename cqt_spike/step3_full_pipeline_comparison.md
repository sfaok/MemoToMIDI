# Step 3 Full Pipeline Comparison

## Inputs
- Audio: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/synthetic_g_major.wav`
- Overlap frames: `30`
- Total windows: `2`

## Saved Aggregated Tensors
- Tensor layout: `(time_frames, frequency_bins)`
- TensorFlow `note`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_tf_full_note.npy`
- TensorFlow `onset`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_tf_full_onset.npy`
- TensorFlow `contour`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_tf_full_contour.npy`
- CoreML CPU+GPU `note`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_coreml_cpu_and_gpu_full_note.npy`
- CoreML CPU+GPU `onset`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_coreml_cpu_and_gpu_full_onset.npy`
- CoreML CPU+GPU `contour`: `/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/cqt_spike/step3_coreml_cpu_and_gpu_full_contour.npy`

## CPU+GPU Aggregated Diff Metrics
- `note`: max_abs_diff=0.161053330, mean_abs_diff=0.004453291, aligned_shape=[259, 88]
- `onset`: max_abs_diff=0.204722404, mean_abs_diff=0.009774516, aligned_shape=[259, 88]
- `contour`: max_abs_diff=0.222591609, mean_abs_diff=0.005596353, aligned_shape=[259, 264]

## Note Threshold Flip Metrics (0.5)
- Active frames (TF): `255`
- Flipped active frames: `153`
- Frame-flip rate (% of active frames): `60.000000`
- Active activations (TF): `568`
- Flipped activations: `156`
- Activation-flip rate (% of active activations): `27.464789`

## Gate
- Status: `RED`
- Reason: Frame-flip rate is >5%; conversion parity issue needs debugging.

## RED Follow-Up (CPU_ONLY)
- CPU_ONLY run status: `BLOCKED`
- CPU_ONLY error: `CPU_ONLY prediction probe crashed.`
- CPU_ONLY details: terminated by signal 8. scikit-learn version 1.6.1 is not supported. Minimum required version: 0.17. Maximum required version: 1.5.1. Disabling scikit-learn conversion API.
/Users/jamesrobertson/MemoToMIDI/MemoToMIDI/.venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(

## Divergence Localization (CPU+GPU)
- Worst diverging tensor: `contour` (max_abs_diff=0.222591609)
- Most divergent frame indices (by mean abs diff):
- frame `165`: mean_abs_diff=`0.006980894`
- frame `163`: mean_abs_diff=`0.006958646`
- frame `164`: mean_abs_diff=`0.006920549`
- frame `161`: mean_abs_diff=`0.006830894`
- frame `209`: mean_abs_diff=`0.006789192`
- frame `162`: mean_abs_diff=`0.006724764`
- frame `166`: mean_abs_diff=`0.006724692`
- frame `160`: mean_abs_diff=`0.006715830`
- frame `84`: mean_abs_diff=`0.006647557`
- frame `159`: mean_abs_diff=`0.006603451`
- frame `203`: mean_abs_diff=`0.006587166`
- frame `158`: mean_abs_diff=`0.006566737`
