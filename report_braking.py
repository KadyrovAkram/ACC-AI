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
        models = [key for key in ("predicted_brake", "comparison_brake", "classifier_brake") if key in fields]
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


def sample_breaks(times):
    gaps = [right - left for left, right in zip(times, times[1:])]
    typical_gap = median(gaps)
    return typical_gap, {0, len(times), *(i for i, gap in enumerate(gaps, 1)
                                         if gap > 1.75 * typical_gap)}


def event_matches(times, actual_events, predicted, breaks, near_s):
    matches = {event: [guess for guess in predicted if overlap(event, guess)]
               for event in actual_events}
    free = [guess for guess in predicted if not any(overlap(event, guess)
                                                     for event in actual_events)]
    late = {}
    for event in reversed(actual_events):
        if matches[event] or event.stop in breaks:
            continue
        for guess in free:
            if (event.stop <= guess.start
                    and times[guess.start] - times[event.stop - 1] <= near_s
                    and not any(event.stop <= gap <= guess.start for gap in breaks)):
                late[event] = guess
                free.remove(guess)
                break
    isolated = [guess for guess in free
                if all(separation(guess, event, times) > near_s for event in actual_events)]
    return matches, late, isolated, len(free) - len(isolated)


def brake_zones(events, times, breaks, gap_s):
    zones = []
    for event in events:
        if zones and times[event.start] - times[zones[-1][-1].stop - 1] <= gap_s and not any(
                zones[-1][-1].stop <= gap <= event.start for gap in breaks):
            zones[-1].append(event)
        else:
            zones.append([event])
    return zones


def zone_stats(times, actual, predicted, breaks, threshold=0.2, gap_s=0.6):
    actual_events = find_events(actual, breaks, threshold)
    guesses = find_events(predicted, breaks, threshold)
    zones = [zone for zone in brake_zones(actual_events, times, breaks, gap_s)
             if sum(event.stop - event.start for event in zone) >= 2]
    onsets = []
    releases = []
    missed = []
    for zone in zones:
        matching = [guess for guess in guesses if any(overlap(event, guess) for event in zone)]
        if not matching:
            missed.append(times[zone[0].start])
            continue
        if zone[0].start not in breaks and matching[0].start not in breaks:
            onsets.append(times[matching[0].start] - times[zone[0].start])
        if zone[-1].stop not in breaks and matching[-1].stop not in breaks:
            releases.append(times[matching[-1].stop] - times[zone[-1].stop])
    isolated = sum(all(separation(guess, event, times) > 0.45 for event in actual_events)
                   for guess in guesses)
    period = median(b - a for a, b in zip(times, times[1:]))
    # Count logged active samples at the typical cadence, without bridging gaps.
    actual_seconds = sum(value >= threshold for value in actual) * period
    predicted_seconds = sum(value >= threshold for value in predicted) * period
    return len(zones), missed, onsets, releases, isolated, actual_seconds, predicted_seconds


def report_model(name, times, actual_events, scored_events, actual, predictions, breaks, args):
    predicted = find_events(predictions, breaks, args.threshold)
    matches_by_event, late, isolated, nearby = event_matches(
        times, actual_events, predicted, breaks, args.near_ms / 1000
    )
    onset_errors = []
    release_errors = []
    detected = 0
    print(f"\n{name}")
    print("  Actual start   Onset      Release    Actual peak   Predicted peak")
    for event in scored_events:
        matches = matches_by_event[event]
        actual_peak = max(actual[event.start:event.stop])
        if not matches:
            if event in late:
                delay = times[late[event].start] - times[event.start]
                print(f"  {times[event.start]:8.2f} s   after release ({timing(delay)} from onset)")
            else:
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

    held_events = sum(sum(bool(overlap(event, other)) for other in actual_events) > 1
                      for event in predicted)
    late_count = sum(event in late for event in scored_events)
    print(f"  Detected {detected}/{len(scored_events)}; after release {late_count}; "
          f"missed without a nearby late response {len(scored_events) - detected - late_count}.")
    print(f"  Predicted presses spanning multiple actual presses: {held_events}.")
    print(f"  Median onset {timing(median(onset_errors) if onset_errors else None)}; "
          f"median release {timing(median(release_errors) if release_errors else None)}.")
    print(f"  Non-overlapping predictions: {nearby} near actual braking, {len(isolated)} isolated.")
    for event in isolated:
        peak = max(predictions[event.start:event.stop])
        print(f"    {times[event.start]:.2f} s: {event.stop - event.start} sample(s), peak {peak:.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--threshold", type=float, default=0.2, help="Brake pedal threshold (0–1).")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum samples per actual event.")
    parser.add_argument("--near-ms", type=float, default=450, help="Proximity for unmatched predictions.")
    parser.add_argument("--startup-seconds", type=float, default=0,
                        help="Report samples before this elapsed time separately.")
    parser.add_argument("--zone-gap-ms", type=float, default=600,
                        help="Merge actual brake presses within this gap for the zone summary.")
    parser.add_argument("--prediction-cutoff", type=float,
                        help="Apply this cutoff to the primary model, keeping the actual threshold fixed.")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or args.min_samples < 1 or not 0 <= args.near_ms < math.inf or not 0 <= args.startup_seconds < math.inf or not 0 <= args.zone_gap_ms < math.inf:
        parser.error("Use a threshold in (0, 1], positive min-samples, and finite nonnegative timing options.")
    try:
        times, actual, models = read_log(args.log)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    if args.prediction_cutoff is not None:
        if not 0 < args.prediction_cutoff <= 1:
            parser.error("--prediction-cutoff must be in (0, 1].")
        models["predicted_brake"] = [float(value >= args.prediction_cutoff)
                                     for value in models["predicted_brake"]]
    if args.startup_seconds:
        end = next((i for i, t in enumerate(times) if t >= args.startup_seconds), len(times))
        print(f"Startup before {args.startup_seconds:g} s ({end} frames)")
        for key, predictions in models.items():
            calls = sum(p >= args.threshold and a < args.threshold
                        for p, a in zip(predictions[:end], actual[:end]))
            print(f"  {key}: {calls} extra braking frames")
        times, actual = times[end:], actual[end:]
        models = {key: predictions[end:] for key, predictions in models.items()}
        if len(times) < 2:
            parser.error("Need at least two samples after startup.")
    typical_gap, breaks = sample_breaks(times)
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
        total, missed, onset, release, isolated, actual_s, predicted_s = zone_stats(
            times, actual, predictions, breaks, args.threshold, args.zone_gap_ms / 1000
        )
        print(f"  Zones ({args.zone_gap_ms:g} ms merge gap): {total - len(missed)}/{total} detected; "
              f"median onset {timing(median(onset) if onset else None)}, "
              f"release {timing(median(release) if release else None)}.")
        print(f"  Brake time: {predicted_s:.1f} s predicted vs {actual_s:.1f} s actual; "
              f"{isolated} isolated calls.")
        if missed:
            print("  Missed zones (s): " + ", ".join(f"{time:.2f}" for time in missed))


if __name__ == "__main__":
    main()
