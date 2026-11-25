"""Before-versus-after CS population decoding for one recording."""

import argparse
import csv
import json
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC


SAMPLE_RATE = 30
WINDOW_SECONDS = 6.0
MIN_SHIFT_SECONDS = 300.0
N_TRIALS = 5
N_SHIFTS = 50


def _resample(time, signal):
    """Interpolate each trace on the 30 Hz analysis grid."""
    interval = 1 / SAMPLE_RATE
    if not np.isclose(np.median(np.diff(time)), interval, atol=interval * 0.05):
        raise ValueError("Expected fluorescence recorded at approximately 30 Hz.")

    n_frames = int(round((time[-1] - time[0]) * SAMPLE_RATE)) + 1
    grid = time[0] + np.arange(n_frames, dtype=float) / SAMPLE_RATE
    gridded = np.full((n_frames, signal.shape[1]), np.nan, dtype=np.float32)
    for cell, trace in enumerate(signal.T):
        finite = np.isfinite(trace)
        if np.count_nonzero(finite) < 2:
            continue
        measured_time = time[finite]
        inside = (grid >= measured_time[0] - interval / 10) & (
            grid <= measured_time[-1] + interval / 10
        )
        gridded[inside, cell] = np.interp(grid[inside], measured_time, trace[finite])
    return grid, gridded


def _event_windows(time, onsets):
    windows = []
    for onset in onsets:
        onset = time[np.argmin(np.abs(time - onset))]
        if onset - WINDOW_SECONDS < time[0] or onset + WINDOW_SECONDS > time[-1]:
            raise ValueError("Each CS onset needs a complete 6 s baseline and 6 s response window.")
        baseline = np.flatnonzero((time >= onset - WINDOW_SECONDS) & (time < onset))
        response = np.flatnonzero((time >= onset) & (time < onset + WINDOW_SECONDS))
        windows.append((baseline, response))
    return windows


def _segment_starts(segment_ids, n_frames):
    if segment_ids is None:
        return None
    labels = np.asarray(segment_ids, dtype=object)
    if labels.shape != (n_frames,):
        raise ValueError("Expected one segment ID per frame. Blank IDs are allowed.")

    starts = [0]
    previous = None
    for index, value in enumerate(labels):
        label = "" if value is None else str(value).strip()
        if label:
            if index and label != previous:
                starts.append(index)
            previous = label
    return starts if previous is not None else None


def _prepare_segmented_recording(time, signal, onsets, starts):
    if np.any(onsets < time[0]) or np.any(onsets > time[-1]):
        raise ValueError("Each CS onset must fall within the recording.")

    onset_segments = np.searchsorted(time[starts], onsets, side="right") - 1
    windows = [None] * len(onsets)
    traces = []
    ends = []
    offset = 0
    for segment, (start, stop) in enumerate(zip(starts, starts[1:] + [len(time)])):
        segment_time = time[start:stop]
        trace = signal[start:stop]
        if len(segment_time) > 1:
            grid, trace = _resample(segment_time, trace)
            # Keep only recorded time before joining the resampled segments.
            in_recording = grid <= segment_time[-1] + 1e-9
            grid, trace = grid[in_recording], trace[in_recording]
        else:
            grid = segment_time
            trace = trace.astype(np.float32)

        for index in np.flatnonzero(onset_segments == segment):
            try:
                (baseline, response), = _event_windows(grid, [onsets[index]])
            except ValueError as error:
                phase = "Before" if index < N_TRIALS else "After"
                trial = index % N_TRIALS + 1
                raise ValueError(
                    f"Invalid recording: {phase} CS {trial} at {onsets[index]:g} s needs "
                    "a complete baseline and response window in one recording segment."
                ) from error
            windows[index] = (baseline + offset, response + offset)

        traces.append(trace)
        offset += len(grid)
        ends.append(offset)
    return np.concatenate(traces, axis=0), windows, np.asarray(ends)


def _eligible_shifts(windows, segment_ends, minimum_shift):
    n_frames = segment_ends[-1]
    candidates = np.arange(minimum_shift, n_frames - minimum_shift)
    for baseline, response in windows:
        start = (baseline[0] - candidates) % n_frames
        end = (response[-1] - candidates) % n_frames
        start_segment = np.searchsorted(segment_ends, start, side="right")
        end_segment = np.searchsorted(segment_ends, end, side="right")
        candidates = candidates[(start <= end) & (start_segment == end_segment)]
    return candidates


def _trial_statistics(signal, windows, shift=0):
    statistics = []
    for baseline_indices, response_indices in windows:
        # Applying one shift to all columns preserves the population pattern.
        baseline = signal[(baseline_indices - shift) % len(signal)].astype(float)
        response = signal[(response_indices - shift) % len(signal)].astype(float)
        baseline_mean = np.nanmean(baseline, axis=0)
        statistics.append(
            (
                np.nanmean(response, axis=0) - baseline_mean,
                baseline_mean,
                np.nanstd(baseline, axis=0, ddof=1),
                len(baseline_indices),
            )
        )
    deltas, means, sds, counts = zip(*statistics)
    return np.stack(deltas), np.stack(means), np.stack(sds), np.asarray(counts)


def _pooled_sd(means, sds, counts):
    counts = counts[:, None].astype(float)
    total_n = counts.sum(axis=0)
    total_sum = np.sum(counts * means, axis=0)
    total_sumsq = np.sum((counts - 1) * sds**2 + counts * means**2, axis=0)
    variance = (total_sumsq - total_sum**2 / total_n) / (total_n - 1)
    return np.sqrt(np.maximum(variance, 0))


def _pairwise_auroc(scores):
    before, after = scores.T
    return float(np.mean((after > before) + 0.5 * (after == before)))


def _decode(deltas, means, sds, counts):
    labels = np.repeat([0, 1], N_TRIALS)
    fold_scores = {"full": [], "pca2": []}
    # Both held-out scores in a pair come from the same fitted classifier.
    for before, after in product(range(N_TRIALS), range(N_TRIALS, 2 * N_TRIALS)):
        held = [before, after]
        train = np.ones(labels.size, dtype=bool)
        train[held] = False
        scale = _pooled_sd(means[train], sds[train], counts[train])
        valid = np.isfinite(scale) & (scale > 1e-12) & np.all(np.isfinite(deltas), axis=0)
        if not np.any(valid):
            raise ValueError("No cells with usable fluorescence remained in a held-out fold.")
        features = deltas[:, valid] / scale[valid]
        x_train, x_test = features[train], features[held]
        y_train = labels[train]

        full = SVC(kernel="linear", C=1.0).fit(x_train, y_train)
        fold_scores["full"].append(full.decision_function(x_test))

        pca = PCA(n_components=min(2, x_train.shape[1]), svd_solver="full").fit(x_train)
        projected = SVC(kernel="linear", C=1.0).fit(pca.transform(x_train), y_train)
        fold_scores["pca2"].append(projected.decision_function(pca.transform(x_test)))

    return {name: np.asarray(scores) for name, scores in fold_scores.items()}


def decode_cs(
    time_seconds, fluorescence, before_cs_seconds, after_cs_seconds, *, recording_id, segment_ids=None
):
    """Decode one recording and return held-out scores and matched shift AUROCs.

    Specify five CS onset times per phase in seconds. Fluorescence has one row
    per timestamp and one column per cell. The recording ID seeds the shift
    draws (the original analysis used each input file's basename as its ID).
    Optional segment IDs mark the first frame of each recorded segment.
    Blank entries carry the preceding ID forward. All-blank IDs use the
    continuous-recording workflow.
    """
    time = np.asarray(time_seconds, dtype=float)
    signal = np.asarray(fluorescence, dtype=float)
    before = np.asarray(before_cs_seconds, dtype=float)
    after = np.asarray(after_cs_seconds, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Expected strictly increasing finite frame times in seconds.")
    if signal.ndim != 2 or signal.shape[0] != time.size or signal.shape[1] == 0:
        raise ValueError("Expected fluorescence with shape (number of frames, number of cells).")
    if before.shape != (N_TRIALS,) or after.shape != (N_TRIALS,):
        raise ValueError("Expected five CS onsets before learning and five after learning.")
    onsets = np.concatenate([before, after])
    if not np.all(np.isfinite(onsets)) or np.unique(onsets).size != 2 * N_TRIALS:
        raise ValueError("Expected distinct, finite CS onset times.")

    starts = _segment_starts(segment_ids, len(time))
    if starts is None:
        grid, signal = _resample(time, signal)
        windows = _event_windows(grid, onsets)
        frame_interval = np.median(np.diff(grid))
    else:
        signal, windows, segment_ends = _prepare_segmented_recording(time, signal, onsets, starts)
        frame_interval = 1 / SAMPLE_RATE

    minimum_shift = int(round(MIN_SHIFT_SECONDS / frame_interval))
    if len(signal) <= 2 * minimum_shift:
        raise ValueError("The recording must span more than 600 s for the circular-shift control.")
    digest = sha256(f"{recording_id}|52000".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    result = {}
    if starts is None:
        shifts = rng.integers(minimum_shift, len(signal) - minimum_shift, size=N_SHIFTS)
    else:
        eligible = _eligible_shifts(windows, segment_ends, minimum_shift)
        if eligible.size == 0:
            raise ValueError("No circular shift keeps all CS windows within single recording segments.")
        shifts = rng.choice(eligible, size=N_SHIFTS, replace=True)
        result["n_valid_shift_offsets"] = int(eligible.size)
    result["shift_frames"] = shifts

    observed = _decode(*_trial_statistics(signal, windows))
    shift_aurocs = {name: [] for name in observed}
    for shift in shifts:
        shifted = _decode(*_trial_statistics(signal, windows, shift))
        for name, scores in shifted.items():
            shift_aurocs[name].append(_pairwise_auroc(scores))

    for name, scores in observed.items():
        null = np.asarray(shift_aurocs[name])
        result[name] = {
            "auroc": _pairwise_auroc(scores),
            "fold_scores": scores,
            "shift_aurocs": null,
            "mean_shift_auroc": float(null.mean()),
        }
    return result


def _read_recording(path):
    def parse_onset_flag(value):
        return float(value.strip() or "0")

    with path.open(encoding="utf-8-sig", newline="") as file:
        header = [name.strip().lower() for name in next(csv.reader(file), [])]

    numeric_columns = None
    segment_ids = None
    if "segment_id" in header:
        segment_column = header.index("segment_id")
        if segment_column != 3 or header.count("segment_id") != 1:
            raise ValueError("Put one segment_id column immediately after the two CS onset columns.")
        numeric_columns = [index for index in range(len(header)) if index != segment_column]
        segment_ids = np.loadtxt(
            path, delimiter=",", skiprows=1, usecols=segment_column, dtype=str, ndmin=1,
            encoding="utf-8-sig", comments=None, quotechar='"',
        )

    recording = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        ndmin=2,
        encoding="utf-8-sig",
        converters={1: parse_onset_flag, 2: parse_onset_flag},
        usecols=numeric_columns,
        comments=None,
        quotechar='"',
    )
    if recording.shape[1] < 4:
        raise ValueError("The recording CSV needs time, two CS onset columns, and at least one cell column.")

    flags = recording[:, 1:3]
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("CS onset columns must contain 1 on an onset frame and 0 or blank otherwise.")
    before, after = flags.T == 1
    if np.any(before & after):
        raise ValueError("A frame can mark either a Before or an After CS onset.")

    time = recording[:, 0]
    return time, recording[:, 3:], time[before], time[after], segment_ids


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="CSV with time, Before and After onset flags, and cell traces")
    parser.add_argument("--recording-id", help="ID used for the shift seed (default: recording filename)")
    parser.add_argument("--output", type=Path, help="New JSON output file (default: print JSON)")
    args = parser.parse_args()

    try:
        time, traces, before, after, segment_ids = _read_recording(args.recording)
        result = decode_cs(
            time, traces, before, after,
            recording_id=args.recording_id or args.recording.name,
            segment_ids=segment_ids,
        )
        output = json.dumps(result, default=np.ndarray.tolist, indent=2, allow_nan=False)
        if args.output is None:
            print(output)
        else:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(output + "\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    _main()
