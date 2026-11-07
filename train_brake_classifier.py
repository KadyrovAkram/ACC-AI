"""Train a brake classifier and evaluate frozen models on a separate ACC stint."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shadow_drive import load_model
from train_imitation import DrivingCNN, IMAGE_SIZE, ROAD_CROP, check_device, load_image, read_session


BRAKE_ON = 0.2
THRESHOLDS = [step / 100 for step in range(5, 100, 5)]


class BrakeClassifier(DrivingCNN):
    def __init__(self):
        super().__init__()
        self.head[-1] = nn.Linear(self.head[-1].in_features, 1)

    def forward(self, images):
        return self.head(self.features(images)).squeeze(1)


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
    tolerance = 0.5 * float(torch.tensor(times).diff().median())
    targets = []
    later = 0
    for sample in samples:
        goal = sample.time + shift_s
        while later + 1 < len(samples) and abs(times[later + 1] - goal) <= abs(times[later] - goal):
            later += 1
        targets.append(samples[later].controls[2] if abs(times[later] - goal) <= tolerance else None)
    return targets


def load_stint(path, shift_s=0.0, size=IMAGE_SIZE, crop=ROAD_CROP):
    """Return samples, images, current brake pedal and the (possibly shifted) training target."""
    every = read_session(path)
    samples = skip_idle_start(every)
    targets = future_brake(samples, shift_s)
    keep = [i for i, target in enumerate(targets) if target is not None]
    samples = [samples[i] for i in keep]
    if len(samples) < 2:
        raise ValueError(f"{path}: needs at least two frames")
    print(f"Loading {len(samples)} frames from {path.name} "
          f"(skipped {len(every) - len(samples)} idle-start or unshiftable)", flush=True)
    images = torch.stack([load_image(sample.path, size, crop).mul(255).round().byte()
                          for sample in samples])
    pedal = torch.tensor([sample.controls[2] for sample in samples])
    target = torch.tensor([targets[i] for i in keep])
    return samples, images, pedal, target


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


def tune_threshold(values, actual):
    if not actual.any() or actual.all():
        raise ValueError("Threshold tuning needs both braking and non-braking frames.")
    choices = []
    for threshold in THRESHOLDS:
        predicted = values >= threshold
        hits = int((predicted & actual).sum())
        false = int((predicted & ~actual).sum())
        f1 = 2 * hits / (int(actual.sum()) + int(predicted.sum()))
        choices.append((f1, -false, threshold))
    f1, _, threshold = max(choices)
    return threshold, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True,
                        help="One or more training stint directories.")
    for split in ("validation", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("runs/brake_weighted/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/brake_classifier"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--shift-ms", type=float, default=0.0,
                        help="Train each frame on the brake pedal this many ms later.")
    args = parser.parse_args()
    stints = [*args.train, args.validation, args.test]
    if args.epochs < 1 or len({p.resolve() for p in stints}) != len(stints):
        parser.error("Use positive epochs and separate stint directories for every split.")

    torch.manual_seed(42)
    device = check_device("cuda")
    shift_s = args.shift_ms / 1000
    loaded = [load_stint(stint, shift_s) for stint in args.train]
    train_images = torch.cat([images for _, images, _, _ in loaded])
    train_target = torch.cat([target for _, _, _, target in loaded])
    _, val_images, val_pedal, val_target = load_stint(args.validation, shift_s)
    train_labels = (train_target >= BRAKE_ON).float()
    val_labels = val_target >= BRAKE_ON
    val_current = val_pedal >= BRAKE_ON
    positives = int(train_labels.sum())
    if not 0 < positives < len(train_labels):
        raise ValueError("Training needs both braking and non-braking frames.")
    print(f"Training positives: {positives}/{len(train_labels)}")
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True)
    model = BrakeClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weight = torch.tensor((len(train_labels) - positives) / positives, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "best.pt"
    best = (-1.0, float("-inf"))
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(normalize(images, device)), labels.to(device))
            loss.backward()
            optimizer.step()
        val_scores = scores(model, val_images, device, classifier=True)
        threshold, f1 = tune_threshold(val_scores, val_labels)
        false = int(((val_scores >= threshold) & ~val_labels).sum())
        print(f"Epoch {epoch:02}: validation F1 {f1:.3f} at {threshold:.2f} ({false} extra frames)")
        if (f1, -false) > best:
            best = (f1, -false)
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "threshold": threshold, "validation_f1": f1,
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
    baseline_threshold, baseline_f1 = tune_threshold(scores(baseline, val_images, device), val_current)
    print(f"Frozen thresholds: weighted {baseline_threshold:.2f} (F1 {baseline_f1:.3f}), "
          f"classifier {saved['threshold']:.2f} (F1 {saved['validation_f1']:.3f})")

    samples, test_images, actual, _ = load_stint(args.test)
    baseline_scores = scores(baseline, test_images, device)
    classifier_scores = scores(model, test_images, device, classifier=True)
    decisions = [baseline_scores >= baseline_threshold, classifier_scores >= saved["threshold"]]
    report = args.output / "test_predictions.csv"
    with report.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["elapsed_s", "frame", "actual_brake", "predicted_brake",
                         "comparison_brake", "weighted_pedal", "classifier_score"])
        for i, sample in enumerate(samples):
            writer.writerow([sample.time - samples[0].time, sample.path.name, actual[i].item(),
                             int(decisions[0][i]), int(decisions[1][i]),
                             baseline_scores[i].item(), classifier_scores[i].item()])
    with (args.output / "experiment.json").open("x", encoding="utf-8") as file:
        json.dump({"train": [str(stint) for stint in args.train], "validation": str(args.validation), "test": str(args.test),
                   "baseline": str(args.baseline), "baseline_threshold": baseline_threshold,
                   "baseline_validation_f1": baseline_f1, "classifier_threshold": saved["threshold"],
                   "classifier_validation_f1": saved["validation_f1"], "epoch": saved["epoch"], "shift_ms": args.shift_ms,
                   "columns": {"predicted_brake": "brake-weighted baseline",
                               "comparison_brake": "brake classifier"}}, file, indent=2)
    print(f"Test predictions: {report.resolve()}")


if __name__ == "__main__":
    main()
