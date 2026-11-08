"""Compare steering and throttle around human braking zones in a shadow log."""
import argparse
import csv
import math
from pathlib import Path
from statistics import median

from report_braking import brake_zones, find_events, sample_breaks, timing


STEER = 0.15
THROTTLE = 0.5


def read_controls(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        keys = ["elapsed_s", "actual_brake", "actual_steering", "actual_throttle",
                "predicted_steering", "predicted_throttle"]
        fields = set(reader.fieldnames or [])
        prefixes = ["predicted"]
        if {"comparison_steering", "comparison_throttle"} <= fields:
            keys.extend(["comparison_steering", "comparison_throttle"])
            prefixes.append("comparison")
        if missing := set(keys) - fields:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        rows = [{key: float(row[key]) for key in keys} for row in reader]
    times = [row["elapsed_s"] for row in rows]
    if (len(rows) < 2 or any(right <= left for left, right in zip(times, times[1:]))
            or not all(math.isfinite(value) for row in rows for value in row.values())):
        raise ValueError("Need finite controls and at least two increasing timestamps.")
    return times, rows, prefixes


def first_sustained(rows, key, start, stop, threshold, direction=1):
    return next((i for i in range(start, stop - 1)
                 if direction * rows[i][key] >= threshold
                 and direction * rows[i + 1][key] >= threshold), None)


def report(path):
    times, rows, prefixes = read_controls(path)
    _, breaks = sample_breaks(times)
    brake = [row["actual_brake"] for row in rows]
    zones = brake_zones(find_events(brake, breaks, .2), times, breaks, .6)
    zones = [zone for zone in zones if sum(e.stop - e.start for e in zone) >= 3]
    print(f"{path.name} | {len(zones)} sustained brake zones")
    print("Phase estimates from driver inputs, without track-position labels.")
    print("Turn-in: two steering samples past +/-0.15. Exit: two throttle samples at or above 50%.")
    print("Positive timing means the model acted later. Missing phase candidates show as n/a.")
    for prefix in prefixes:
        steer_offsets, throttle_offsets = [], []
        steer_opportunities = throttle_opportunities = 0
        print(f"\n{prefix}\n  Zone start   Turn-in     Throttle exit")
        for n, zone in enumerate(zones):
            start, release = zone[0].start, zone[-1].stop
            left = max(gap for gap in breaks if gap <= start)
            right = min(gap for gap in breaks if gap > start)
            if n + 1 < len(zones):
                right = min(right, zones[n + 1][0].start)
            steer_end = next((i for i in range(start, right)
                              if times[i] > times[release - 1] + .7), right)
            steer_start = next((i for i in range(left, start + 1)
                                if times[i] >= times[start] - .25), start)
            actual_turns = [(i, sign) for sign in (-1, 1)
                            if (i := first_sustained(rows, "actual_steering", steer_start,
                                                     steer_end, STEER, sign)) is not None]
            steer_delta = None
            if actual_turns:
                steer_opportunities += 1
                turn, sign = min(actual_turns)
                before = next((i for i in range(steer_start, steer_end)
                               if times[i] >= times[turn] - .5), turn)
                after = next((i for i in range(turn, steer_end)
                              if times[i] > times[turn] + .5), steer_end)
                predicted = first_sustained(rows, f"{prefix}_steering", before, after, STEER, sign)
                if predicted is not None:
                    steer_delta = times[predicted] - times[turn]
                    steer_offsets.append(steer_delta)
            throttle_delta = None
            if release < right:
                exit_end = next((i for i in range(release, right)
                                 if times[i] > times[release] + 3), right)
                exit_start = next((i for i in range(start, release + 1)
                                   if times[i] >= times[release] - .5), release)
                actual_exit = first_sustained(rows, "actual_throttle", release, exit_end, THROTTLE)
                if actual_exit is not None:
                    throttle_opportunities += 1
                    predicted = first_sustained(rows, f"{prefix}_throttle", exit_start, exit_end, THROTTLE)
                    if predicted is not None:
                        throttle_delta = times[predicted] - times[actual_exit]
                        throttle_offsets.append(throttle_delta)
            print(f"  {times[start]:8.2f} s   {timing(steer_delta):10}  {timing(throttle_delta)}")
        print(f"  Matched same-direction turn-in: {len(steer_offsets)}/{steer_opportunities}; "
              f"median offset {timing(median(steer_offsets) if steer_offsets else None)}.")
        print(f"  Matched throttle exit: {len(throttle_offsets)}/{throttle_opportunities}; "
              f"median offset {timing(median(throttle_offsets) if throttle_offsets else None)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        report(args.log)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
