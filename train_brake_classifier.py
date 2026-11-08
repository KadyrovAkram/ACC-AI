"""Train a brake classifier and evaluate frozen models on a separate ACC stint."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from report_braking import sample_breaks, zone_stats
from shadow_drive import load_model
from train_imitation import BrakeClassifier, IMAGE_SIZE, ROAD_CROP, check_device, load_image, read_session


BRAKE_ON = 0.2
THRESHOLDS = [step / 100 for step in range(5, 100, 5)] + [0.96, 0.97, 0.98, 0.99, 0.995]


def skip_idle_start(samples):
    """Drop leading frames before the driver's first throttle, brake or steering input."""
    for index, sample in enumerate(samples):
        steering, throttle, brake = sample.controls
        if throttle > 0 or brake > 0 or abs(steering) > 0.01:
            return samples[index:]
    raise ValueError("No driver input found in stint.")


def future_brake(samples, shift_s):
    """Brake pedal shift_s after each frame, or None when no sample lands close enough."""
    if shift_s == 0:
        return [sample.controls[2] for sample in samples]
    times = [sample.time for sample in samples]
    tolerance = 0.5 * median(b - a for a, b in zip(times, times[1:]))
    targets = []
    later = 0
    for sample in samples:
        goal = sample.time + shift_s
        while later + 1 < len(samples) and abs(times[later + 1] - goal) <= abs(times[later] - goal):
            later += 1
        targets.append(samples[later].controls[2] if abs(times[later] - goal) <= tolerance else None)
    return targets


def load_stint(path, shift_s=0.0, exclusions=(), trim_start=True):
    """Return samples, images, current brake pedal and the (possibly shifted) training target."""
    every = read_session(path)
    origin = every[0].time
    started = skip_idle_start(every)
    startup_seconds = started[0].time - origin
    samples = started if trim_start else every
    samples = [sample for sample in samples
               if not any(start <= sample.time - origin < stop
                          for name, start, stop in exclusions if name == path.name)]
    if len(samples) < 2:
        raise ValueError(f"{path}: needs at least two frames after exclusions")
    targets = future_brake(samples, shift_s)
    keep = [i for i, target in enumerate(targets) if target is not None]
    samples = [samples[i] for i in keep]
    if len(samples) < 2:
        raise ValueError(f"{path}: needs at least two frames")
    print(f"Loading {len(samples)} frames from {path.name} "
          f"(skipped {len(every) - len(samples)} idle-start or unshiftable)", flush=True)
    images = torch.stack([load_image(sample.path).mul(255).round().byte()
                          for sample in samples])
    pedal = torch.tensor([sample.controls[2] for sample in samples])
    target = torch.tensor([targets[i] for i in keep])
    return samples, images, pedal, target, origin, startup_seconds


def normalize(images, device):
    return images.to(device, dtype=torch.float32).div_(255)


@torch.inference_mode()
def scores(model, images, device, classifier=False):
    model.eval()
    batches = []
    for batch in images.split(64):
        output = model(normalize(batch, device))
        batches.append((output.sigmoid() if classifier else output[:, 2]).cpu())
    return torch.cat(batches)


def tune_threshold(values, actual, samples):
    labels = actual >= BRAKE_ON
    positive_count = int(labels.sum())
    if not 0 < positive_count < len(labels):
        raise ValueError("Threshold tuning needs both braking and non-braking frames.")
    times = [sample.time for sample in samples]
    _, breaks = sample_breaks(times)
    actual_values = actual.tolist()
    choices = []
    for threshold in THRESHOLDS:
        predicted = values >= threshold
        hits = int((predicted & labels).sum())
        frame_f1 = 2 * hits / (positive_count + int(predicted.sum()))
        total, missed, _, release, isolated, actual_s, predicted_s = zone_stats(
            times, actual_values, predicted.tolist(), breaks, BRAKE_ON
        )
        if not total:
            raise ValueError("Threshold tuning needs at least one braking zone.")
        detected = total - len(missed)
        zone_f1 = 2 * detected / (total + detected + isolated)
        release_error = median(abs(value) for value in release) if release else math.inf
        choices.append((zone_f1, -release_error, -abs(predicted_s - actual_s), frame_f1, threshold))
    return max(choices)


def parse_exclusion(value):
    try:
        name, start, stop = value.split(":")
        start, stop = float(start), float(stop)
        if not name or not 0 <= start < stop < math.inf:
            raise ValueError()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use STINT:START_SECONDS:STOP_SECONDS.") from exc
    return name, start, stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True,
                        help="One or more training stint directories.")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("runs/brake_weighted/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/brake_classifier"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--shift-ms", type=float, default=0.0,
                        help="Train each frame on the brake pedal this many ms later.")
    parser.add_argument("--exclude", action="append", type=parse_exclusion, default=[])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fit-only", action="store_true", help="Freeze models without reading test frames.")
    modes.add_argument("--evaluate-only", action="store_true", help="Score the saved run on its reserved stint.")
    args = parser.parse_args()
    if not args.fit_only and args.test is None:
        parser.error("--test is required unless using --fit-only.")
    stints = [*args.train, args.validation]
    if args.test is not None:
        stints.append(args.test)
    if args.epochs < 1 or len({p.resolve() for p in stints}) != len(stints):
        parser.error("Use positive epochs and separate stint directories for every split.")

    if not 0 <= args.shift_ms < math.inf:
        parser.error("--shift-ms must be finite and nonnegative.")
    if any(name not in {path.name for path in stints} for name, _, _ in args.exclude):
        parser.error("Exclusions must name a stint in the split.")
    torch.manual_seed(42)
    device = check_device("cuda")
    checkpoint = args.output / "best.pt"
    metadata = args.output / "experiment.json"
    configuration = {"train": [str(stint) for stint in args.train],
                     "validation": str(args.validation), "test": str(args.test) if args.test else None,
                     "baseline": str(args.baseline), "shift_ms": args.shift_ms,
                     "exclusions": [list(window) for window in args.exclude]}
    model = BrakeClassifier().to(device)
    if not args.evaluate_only:
        shift_s = args.shift_ms / 1000
        loaded = [load_stint(stint, shift_s, args.exclude) for stint in args.train]
        train_images = torch.cat([stint[1] for stint in loaded])
        train_target = torch.cat([stint[3] for stint in loaded])
        del loaded
        val_samples, val_images, val_pedal, _, _, _ = load_stint(args.validation, exclusions=args.exclude)
        train_labels = (train_target >= BRAKE_ON).float()
        positives = int(train_labels.sum())
        if not 0 < positives < len(train_labels):
            raise ValueError("Training needs both braking and non-braking frames.")
        print(f"Training positives: {positives}/{len(train_labels)}")
        loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        weight = torch.tensor((len(train_labels) - positives) / positives, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        args.output.mkdir(parents=True, exist_ok=False)
        best = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            for images, labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(normalize(images, device)), labels.to(device))
                loss.backward()
                optimizer.step()
            val_scores = scores(model, val_images, device, classifier=True)
            result = tune_threshold(val_scores, val_pedal, val_samples)
            zone_f1, _, _, frame_f1, threshold = result
            print(f"Epoch {epoch:02}: validation zone score {zone_f1:.3f}, "
                  f"frame F1 {frame_f1:.3f} at {threshold:.2f}")
            if best is None or result > best:
                best = result
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "threshold": threshold, "validation_f1": frame_f1,
                            "validation_zone_score": zone_f1,
                            "image_size": list(IMAGE_SIZE), "road_crop": list(ROAD_CROP),
                            "actual_brake_threshold": BRAKE_ON, "shift_ms": args.shift_ms}, checkpoint)

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    baseline, baseline_saved = load_model(args.baseline, device)
    if baseline.speed_scale is not None:
        raise ValueError("Use the camera-only brake-weighted checkpoint as the baseline.")
    if (tuple(baseline_saved["image_size"]) != IMAGE_SIZE
            or tuple(baseline_saved["road_crop"]) != ROAD_CROP):
        raise ValueError("Baseline uses a different image crop or size.")
    if args.evaluate_only:
        experiment = json.loads(metadata.read_text(encoding="utf-8"))
        if experiment.get("test") is None:
            # A fit-only run can reserve a future recording that did not exist at fit time.
            configuration["test"] = None
        if any(experiment.get(key) != value for key, value in configuration.items()):
            parser.error("Evaluation arguments must match the frozen experiment.")
        baseline_threshold = experiment["baseline_threshold"]
    else:
        baseline_zone, _, _, baseline_f1, baseline_threshold = tune_threshold(
            scores(baseline, val_images, device), val_pedal, val_samples
        )
        experiment = dict(configuration, baseline_threshold=baseline_threshold,
                          baseline_validation_f1=baseline_f1, baseline_zone_score=baseline_zone,
                          classifier_threshold=saved["threshold"], classifier_validation_f1=saved["validation_f1"],
                          classifier_zone_score=saved["validation_zone_score"], epoch=saved["epoch"],
                          columns={"predicted_brake": "brake-weighted baseline",
                                   "comparison_brake": "brake classifier"})
        metadata.write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
        print(f"Frozen thresholds: weighted {baseline_threshold:.2f}, classifier {saved['threshold']:.2f}")
        if args.fit_only:
            return

    samples, test_images, actual, _, origin, startup_seconds = load_stint(
        args.test, exclusions=args.exclude, trim_start=False
    )
    baseline_scores = scores(baseline, test_images, device)
    classifier_scores = scores(model, test_images, device, classifier=True)
    decisions = [baseline_scores >= baseline_threshold, classifier_scores >= saved["threshold"]]
    report = args.output / "test_predictions.csv"
    with report.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["elapsed_s", "frame", "actual_brake", "predicted_brake",
                         "comparison_brake", "weighted_pedal", "classifier_score"])
        for i, sample in enumerate(samples):
            writer.writerow([sample.time - origin, sample.path.name, actual[i].item(),
                             int(decisions[0][i]), int(decisions[1][i]),
                             baseline_scores[i].item(), classifier_scores[i].item()])
    print(f"Startup ends at {startup_seconds:.6f} s. Use --startup-seconds {startup_seconds:.6f} for the report.")
    print(f"Test predictions: {report.resolve()}")


if __name__ == "__main__":
    main()
