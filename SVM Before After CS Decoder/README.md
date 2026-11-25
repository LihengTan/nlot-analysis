# SVM Decoder for Before/After CS Responses

`decoder.py` uses a linear support vector machine (SVM) to decode learning stage from population calcium responses to the conditioned stimulus (CS) within each mouse. This implementation was used to analyze NLOT recordings.

The analysis uses 25 balanced leave-pair-out folds. It compares the two held-out decision scores within each fold to obtain a pairwise AUROC. Train-fold baseline scaling is shared by the full-population decoder and the two-component PCA control. PCA is fitted on the training trials of each fold. Both decoders are evaluated against 50 matched whole-recording circular shifts.

## Tested environment

Tested on Windows with Python 3.13.13, NumPy 2.4.3, scikit-learn 1.9.1, and SciPy 1.17.1. SciPy is a dependency of scikit-learn.

```bash
python -m pip install numpy==2.4.3 scikit-learn==1.9.1 scipy==1.17.1
```

## Usage

Use one CSV per mouse. Include one header row and one row per frame. The columns are in this order.

1. `time_seconds` gives the frame timestamp in seconds. Timestamps must increase throughout the file, including across segment boundaries.
2. `cs_before` marks the five selected CS onsets before learning. Enter `1` on the first frame of each selected CS and `0` or leave the field blank on other frames.
3. `cs_after` marks the five selected CS onsets after learning, using the same convention.
4. `segment_id` is optional. Place it here if the recording contains interruptions or joins between separate recording segments.
5. The remaining columns contain numeric fluorescence values, one column per cell. Each column follows the same cell throughout the file.

Run the decoder from a terminal.

```bash
python decoder.py recording.csv --recording-id session-01 --output decoder_results.json
```

The recording ID seeds the circular-shift draws and defaults to the CSV filename. Choose a new path for `--output`, or omit it to print JSON to the terminal. Python users can also import `decode_cs()` and pass frame times, fluorescence traces, and the two lists of CS onset times directly.

## Recording boundaries

For a recording made of separate segments, enter a distinct `segment_id` on the first frame of each segment, such as `1`, `2` and `3`. Repeat the ID on the other frames of that segment or leave them blank. Blank cells inherit the previous ID, and any initial unmarked frames form the first segment. An omitted or entirely blank column uses the continuous-recording workflow without boundary checks. The Python interface accepts the same per-frame markers via `segment_ids`.

With segment markers, interpolation is done separately within each recorded segment. The 6 s baseline and 6 s response window of every observed CS must fit together in one segment. If any observed window crosses a boundary or is incomplete, the decoder reports that recording as invalid. For the circular-shift control, an offset is eligible only if every shifted window fits in a single segment without wrapping across the recording endpoints. The 50 draws are sampled with replacement from the eligible offsets. The decoder reports an error if none are available.

## Outputs

`result["full"]` contains the full-population decoder results. `result["pca2"]` contains the PCA control results, using two components when at least two usable cell features are available in the fold. Both entries return these fields for one recording:

| Field | Meaning |
| --- | --- |
| `auroc` | Observed pairwise AUROC. Averaged across folds, After scores above Before count as 1, ties as 0.5, and After scores below Before as 0. |
| `fold_scores` | A 25 × 2 array of held-out SVM decision scores. Each row is one test pair, with Before in column 1 and After in column 2. Rows cycle through all After trials for each Before trial, in input order. |
| `shift_aurocs` | The 50 pairwise AUROCs from the circular-shift control, in draw order. |
| `mean_shift_auroc` | Mean of those 50 AUROCs. |

`result["shift_frames"]` contains the 50 offsets, shared by both decoders, in frames on the 30 Hz grid. Divide by 30 to obtain seconds of recorded time. In segmented mode, `result["n_valid_shift_offsets"]` gives the number of eligible offsets before sampling. The command writes these fields to JSON, and the Python function returns the equivalent dictionary. Group-level statistics and plotting are separate analyses.

## Analysis settings

Use recordings sampled at approximately 30 Hz. Within the recording or each marked segment, the median interval between frames must fall within approximately 5% of 1/30 s. The decoder interpolates the traces onto a uniform 30 Hz grid and uses a 6 s baseline and a 6 s response window for each CS onset.

Allow more than 600 s of recorded data for the circular-shift control. Shift offsets are constrained to at least 300 s from alignment in either circular direction. The 50 offsets are drawn with replacement, and near the minimum recording length very few distinct offsets are available.

The sampling rate, window length, trials per phase, minimum shift and number of shifts are constants at the top of `decoder.py`. These settings were used to decode population calcium activity recorded from NLOT neurons in this study.
