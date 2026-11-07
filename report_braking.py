"""Compare braking events in a single-model or paired ACC shadow CSV."""
import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class Event:
    start: int
    stop: int  # First sample outside the event.


def read_log(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        models = [key for key in ("predicted_brake", "comparison_brake") if key in fields]
        required = {"elapsed_s", "actual_brake", "predicted_brake"}
        if missing := required - fields:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        columns = {key: [] for key in ("elapsed_s", "actual_brake", *models)}
        for row in reader:
            for key, values in columns.items():
                try:
                    value = float(row[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Line {reader.line_num}: invalid {key}") from exc
                if not math.isfinite(value) or (key != "elapsed_s" and not 0 <= value <= 1):
                    raise ValueError(f"Line {reader.line_num}: out-of-range {key}")
                values.append(value)
    times = columns.pop("elapsed_s")
    if len(times) < 2 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Need at least two strictly increasing timestamps.")
    return times, columns.pop("actual_brake"), columns


def find_events(values, breaks, threshold):
    events = []
    start = None
    for index, value in enumerate([*values, 0.0]):
        active = value >= threshold
        if start is not None and (not active or index in breaks):
            events.append(Event(start, index))
            start = None
        if active and start is None:
            start = index
    return events


def overlap(left, right):
    return max(0, min(left.stop, right.stop) - max(left.start, right.start))


def separation(left, right, times):
    if left.stop <= right.start:
        return times[right.start] - times[left.stop - 1]
    if right.stop <= left.start:
        return times[left.start] - times[right.stop - 1]
    return 0.0


def timing(value):
    return "n/a" if value is None else f"{value * 1000:+.0f} ms"


def report_model(name, times, actual_events, scored_events, actual, predictions, breaks, args):
    predicted = find_events(predictions, breaks, args.threshold)
    onset_errors = []
    release_errors = []
    detected = 0
    print(f"\n{name}")
    print("  Actual start   Onset      Release    Actual peak   Predicted peak")
    for event in scored_events:
        matches = [candidate for candidate in predicted if overlap(event, candidate)]
        actual_peak = max(actual[event.start:event.stop])
        if not matches:
            print(f"  {times[event.start]:8.2f} s   missed     -          {actual_peak:10.0%}   -")
            continue
        detected += 1
        onset = None
        release = None
        first, last = matches[0], matches[-1]
        held_before = any(other.start < event.start and overlap(other, first)
                          for other in actual_events)
        held_after = any(other.start > event.start and overlap(other, last)
                         for other in actual_events)
        if not held_before and event.start not in breaks and first.start not in breaks:
            onset = times[first.start] - times[event.start]
            onset_errors.append(onset)
        if not held_after and event.stop not in breaks and last.stop not in breaks:
            release = times[last.stop] - times[event.stop]
            release_errors.append(release)
        onset_text = "held" if held_before else timing(onset)
        release_text = "held" if held_after else timing(release)
        predicted_peak = max(predictions[event.start:event.stop])
        print(f"  {times[event.start]:8.2f} s   {onset_text:9}  {release_text:9}  "
              f"{actual_peak:10.0%}   {predicted_peak:14.0%}")

    nearby = []
    isolated = []
    for event in predicted:
        distance = min((separation(event, other, times) for other in actual_events), default=math.inf)
        if 0 < distance <= args.near_ms / 1000:
            nearby.append(event)
        elif distance > args.near_ms / 1000:
            isolated.append(event)
    held_events = sum(sum(bool(overlap(event, other)) for other in actual_events) > 1
                      for event in predicted)
    print(f"  Detected {detected}/{len(scored_events)}; missed {len(scored_events) - detected}.")
    print(f"  Predicted presses spanning multiple actual presses: {held_events}.")
    print(f"  Median onset {timing(median(onset_errors) if onset_errors else None)}; "
          f"median release {timing(median(release_errors) if release_errors else None)}.")
    print(f"  Non-overlapping predictions: {len(nearby)} near actual braking, {len(isolated)} isolated.")
    for event in isolated:
        peak = max(predictions[event.start:event.stop])
        print(f"    {times[event.start]:.2f} s: {event.stop - event.start} sample(s), peak {peak:.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--threshold", type=float, default=0.2, help="Brake pedal threshold (0–1).")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum samples per actual event.")
    parser.add_argument("--near-ms", type=float, default=450, help="Proximity for unmatched predictions.")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or args.min_samples < 1 or not 0 <= args.near_ms < math.inf:
        parser.error("Use a threshold in (0, 1], positive min-samples and finite nonnegative near-ms.")
    try:
        times, actual, models = read_log(args.log)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    gaps = [right - left for left, right in zip(times, times[1:])]
    typical_gap = median(gaps)
    breaks = {0, len(times), *(i for i, gap in enumerate(gaps, 1) if gap > 1.75 * typical_gap)}
    actual_events = find_events(actual, breaks, args.threshold)
    scored_events = [event for event in actual_events if event.stop - event.start >= args.min_samples]
    print(f"{args.log.name} | {len(times)} frames | median sample gap {typical_gap * 1000:.0f} ms")
    print(f"Threshold {args.threshold:.0%}; {len(scored_events)} actual events "
          f"with at least {args.min_samples} sample(s) ({len(actual_events)} total).")
    print("A detection overlaps an actual event. Timing uses the first and last overlapping predictions.")
    print("Held means the prediction stayed on across separate actual presses. Held boundaries are excluded from timing medians.")
    print("Positive timing means late. Timing at a log edge or sample gap is unavailable.")
    print(f"An isolated prediction stays more than {args.near_ms:g} ms from any actual brake sample.")
    for key, predictions in models.items():
        report_model(key, times, actual_events, scored_events, actual, predictions, breaks, args)


if __name__ == "__main__":
    main()
