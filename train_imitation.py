"""Train and evaluate a behavioral cloning model for ACC."""
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CONTROLS = ("steering", "throttle", "brake")
BOUNDS = ((-1, 1), (0, 1), (0, 1))
IMAGE_SIZE = (192, 40)  # Width, height
ROAD_CROP = (145 / 640, 100 / 360, 1.0, 198 / 360)  # Relative to the recorded view
BRAKE_THRESHOLD = 0.05
HARD_BRAKE_THRESHOLD = 0.5
STRONG_STEER_THRESHOLD = 0.15
SPEED_SCALE_KMH = 300.0
SEED = 42
HISTORY_MAX_GAP_S = 0.16


@dataclass(frozen=True)
class Sample:
    path: Path
    time: float
    controls: tuple[float, ...]
    speed_kmh: float


def read_session(session, use_speed=False):
    samples = []
    seen = set()
    csv_path = session / "telemetry.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        time_column = "capture_time" if "capture_time" in columns else "timestamp"
        missing = {"frame", time_column, *CONTROLS} - columns
        if use_speed and "speed_kmh" not in columns:
            missing.add("speed_kmh")
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        for row in reader:
            location = f"{csv_path}:{reader.line_num}"
            name = (row["frame"] or "").strip()
            if not name or Path(name).name != name or name in seen:
                raise ValueError(f"{location}: invalid or duplicate frame {name!r}")
            seen.add(name)
            path = session / "frames" / name
            if not path.is_file():
                raise FileNotFoundError(f"{location}: missing {path}")
            try:
                time = float(row[time_column])
                controls = tuple(float(row[key]) for key in CONTROLS)
                speed_kmh = float(row["speed_kmh"]) if use_speed else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{location}: invalid timestamp or label") from exc
            if not math.isfinite(time):
                raise ValueError(f"{location}: non-finite timestamp")
            for key, value, (low, high) in zip(CONTROLS, controls, BOUNDS):
                if not low <= value <= high:
                    raise ValueError(f"{location}: {key}={value} outside [{low}, {high}]")
            if not math.isfinite(speed_kmh):
                raise ValueError(f"{location}: non-finite speed_kmh")
            samples.append(Sample(path, time, controls, speed_kmh))
    if not samples:
        raise ValueError(f"{csv_path}: no rows")
    return sorted(samples, key=lambda sample: sample.time)


def split_sessions(sessions):
    if len(sessions) > 1:
        val_count = math.ceil(len(sessions) * 0.2)
        train = [sample for session in sessions[:-val_count] for sample in session]
        val = [sample for session in sessions[-val_count:] for sample in session]
        print("Split: newest session(s) held out whole.")
    else:
        samples = sessions[0]
        boundary = int(0.8 * len(samples))
        val = samples[boundary:]
        train = [sample for sample in samples[:boundary] if sample.time < val[0].time - 1.0]
        print(f"Split: chronological final 20% held out, {boundary - len(train)} gap frames.")
    if len(train) < 2 or len(val) < 2:
        raise ValueError("Need at least two samples on each side after splitting.")
    return train, val


def prepare_image(image, size=IMAGE_SIZE, crop=ROAD_CROP):
    dimensions = (image.width, image.height) * 2
    box = tuple(round(fraction * length) for fraction, length in zip(crop, dimensions))
    rgb = image.crop(box).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    pixels = np.array(rgb, dtype=np.uint8)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


def load_image(path, size=IMAGE_SIZE, crop=ROAD_CROP):
    try:
        with Image.open(path) as image:
            return prepare_image(image, size, crop)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode image {path}") from exc


class DrivingDataset(Dataset):
    def __init__(self, samples, history_frames=1, sample_history=1):
        self.windows = []
        self.samples = []
        self.cache = {} if sample_history > 1 else None
        for index in range(sample_history - 1, len(samples)):
            window = samples[index - sample_history + 1:index + 1]
            if any(left.path.parent != right.path.parent or
                   not 0 < right.time - left.time <= HISTORY_MAX_GAP_S
                   for left, right in zip(window, window[1:])):
                continue
            self.windows.append(tuple(frame.path for frame in window[-history_frames:]))
            self.samples.append(window[-1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        controls = torch.tensor(sample.controls, dtype=torch.float32)
        speed = torch.tensor([sample.speed_kmh], dtype=torch.float32)
        paths = self.windows[index]
        if self.cache is None:
            return load_image(paths[0]), controls, speed
        images = []
        for path in paths:
            if path not in self.cache:
                self.cache[path] = load_image(path).mul(255).round().byte()
            images.append(self.cache[path])
        return torch.cat(images).float().div_(255), controls, speed


class DrivingCNN(nn.Module):
    def __init__(self, speed_scale=None, history_frames=1):
        super().__init__()
        self.speed_scale = speed_scale
        self.history_frames = history_frames
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 3 * 5 + int(speed_scale is not None), 64),
                                  nn.ReLU(), nn.Linear(64, 3))
        if history_frames > 1:
            # Start with the same current-frame model; learn the older channels from zero.
            current = self.features[0]
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Conv2d(3 * history_frames, 16, 5, stride=2, padding=2)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, -3:].copy_(current.weight)
                expanded.bias.copy_(current.bias)
            self.features[0] = expanded

    def forward(self, images, speed_kmh=None):
        features = self.features(images)
        if self.speed_scale is not None:
            features = torch.cat((features.flatten(1), speed_kmh / self.speed_scale), dim=1)
        raw = self.head(features)
        return torch.cat((raw[:, :1].tanh(), raw[:, 1:].sigmoid()), dim=1)


class BrakeClassifier(DrivingCNN):
    def __init__(self):
        super().__init__()
        self.head[-1] = nn.Linear(self.head[-1].in_features, 1)

    def forward(self, images):
        return self.head(self.features(images)).squeeze(1)


def label_tensor(samples):
    return torch.tensor([sample.controls for sample in samples], dtype=torch.float32)


def describe(name, samples):
    labels = label_tensor(samples)
    sessions = sorted({sample.path.parent.parent.name for sample in samples})
    print(f"{name}: {len(samples)} frames from {', '.join(sessions)}")
    for key, values in zip(CONTROLS, labels.T):
        print(f"  {key:8s} min={values.min():+.3f} max={values.max():+.3f} "
              f"mean={values.mean():+.3f} std={values.std(unbiased=False):.3f}")
        if values.min() == values.max():
            print(f"  Warning: constant {key} labels in {name}.")
    count = int((labels[:, 2] > BRAKE_THRESHOLD).sum())
    print(f"  Brake > {BRAKE_THRESHOLD}: {count}/{len(samples)}")


def check_device(requested):
    print(f"PyTorch {torch.__version__} | CUDA build {torch.version.cuda}")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Check nvidia-smi and install the cu128 wheel "
                               "in the Python environment running this script.")
        capability = torch.cuda.get_device_capability()
        print(f"GPU: {torch.cuda.get_device_name()} | compute {capability} | "
              f"wheel architectures {torch.cuda.get_arch_list()}")
        probe = nn.Conv2d(3, 4, 3).to(device)
        probe(torch.ones(1, 3, 16, 16, device=device)).mean().backward()
        torch.cuda.synchronize()
        print("CUDA convolution forward and backward passed.")
    return device


@torch.inference_mode()
def predict_batches(model, loader, device):
    model.eval()
    return torch.cat([
        model(images.to(device), speed.to(device)).cpu()
        for images, _, speed in loader
    ])


def show_metrics(name, predictions, targets):
    errors = (predictions - targets).abs()
    values = "  ".join(f"{key}={mae:.4f}" for key, mae in zip(CONTROLS, errors.mean(0)))
    braking = targets[:, 2] > BRAKE_THRESHOLD
    active_error = f"{errors[braking, 2].mean():.4f}" if braking.any() else "n/a"
    print(f"{name}: MAE  {values}  |  brake MAE on active frames={active_error}")


def train(args, device):
    torch.manual_seed(SEED)
    if args.train_stints:
        train_samples = [sample for path in args.train_stints
                         for sample in read_session(path, use_speed=args.use_speed)]
        val_samples = read_session(args.validation_stint, use_speed=args.use_speed)
    else:
        paths = sorted(path for path in args.data.glob("vision_session_*") if path.is_dir())
        if not paths:
            raise ValueError(f"No vision_session_* directories in {args.data.resolve()}")
        train_samples, val_samples = split_sessions([
            read_session(path, use_speed=args.use_speed) for path in paths
        ])
    train_dataset = DrivingDataset(train_samples, args.history_frames, args.sample_history)
    val_dataset = DrivingDataset(val_samples, args.history_frames, args.sample_history)
    train_samples, val_samples = train_dataset.samples, val_dataset.samples
    if len(train_samples) < 2 or len(val_samples) < 2:
        raise ValueError("Need at least two valid frame windows on both sides of the split.")
    describe("Train", train_samples)
    describe("Validation", val_samples)
    print(f"Model input: {args.history_frames} frame(s); anchor window: {args.sample_history} frame(s).")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    targets = label_tensor(val_samples)
    baseline = label_tensor(train_samples).median(dim=0).values
    speed_scale = SPEED_SCALE_KMH if args.use_speed else None
    model = DrivingCNN(speed_scale, args.history_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.1, reduction="none")
    best_mae = math.inf
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for images, controls, speed in train_loader:
            optimizer.zero_grad()
            controls = controls.to(device)
            errors = loss_fn(model(images.to(device), speed.to(device)), controls)
            hard_braking = controls[:, 2] >= HARD_BRAKE_THRESHOLD
            extra_brake_loss = errors[:, 2] * hard_braking * (args.hard_brake_weight - 1)
            strong_steering = controls[:, 0].abs() >= STRONG_STEER_THRESHOLD
            extra_steer_loss = errors[:, 0] * strong_steering * (args.strong_steer_weight - 1)
            loss = (errors.sum(dim=1) + extra_brake_loss + extra_steer_loss).mean() / len(CONTROLS)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * images.size(0)
        predictions = predict_batches(model, val_loader, device)
        val_mae = (predictions - targets).abs().mean().item()
        if not math.isfinite(val_mae):
            raise RuntimeError("Validation MAE is non-finite.")
        print(f"Epoch {epoch:02d}  train loss={total_loss.item()/len(train_samples):.4f}  "
              f"validation mean MAE={val_mae:.4f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation_mae": val_mae,
                "hard_brake_weight": args.hard_brake_weight,
                "strong_steer_weight": args.strong_steer_weight,
                "speed_scale_kmh": speed_scale,
                "history_frames": args.history_frames,
                "controls": list(CONTROLS),
                "image_size": list(IMAGE_SIZE),
                "road_crop": list(ROAD_CROP),
                "preprocessing": "road crop, RGB, bilinear, CHW, pixels / 255",
            }, checkpoint)

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    predictions = predict_batches(model, val_loader, device)
    show_metrics("Train-median baseline", baseline.expand_as(targets), targets)
    show_metrics(f"Best CNN (epoch {saved['epoch']})", predictions, targets)
    with (args.output / "validation_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["session", "frame", "capture_or_timestamp",
                         *[f"actual_{key}" for key in CONTROLS],
                         *[f"predicted_{key}" for key in CONTROLS]])
        for sample, prediction in zip(val_samples, predictions.tolist()):
            writer.writerow([sample.path.parent.parent.name, sample.path.name, sample.time,
                             *sample.controls, *prediction])
    print(f"Saved checkpoint and validation predictions in {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/milestone4"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hard-brake-weight", type=float, default=1.0,
                        help="Loss multiplier for brake targets at or above 50 percent.")
    parser.add_argument("--strong-steer-weight", type=float, default=1.0,
                        help="Loss multiplier when absolute steering is at least 0.15.")
    parser.add_argument("--train-stints", nargs="+", type=Path,
                        help="Use these stints for training instead of the automatic split.")
    parser.add_argument("--validation-stint", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-gpu", action="store_true")
    mode.add_argument("--predict", type=Path, metavar="FRAME")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--history-frames", type=int, choices=(1, 3), default=1)
    parser.add_argument("--sample-history", type=int, choices=(1, 3),
                        help="Require complete history even for a single-frame control run.")
    parser.add_argument("--previous-frames", type=Path, nargs=2, metavar=("OLDEST", "PREVIOUS"),
                        help="Two earlier frames for a three-frame prediction.")
    parser.add_argument("--use-speed", action="store_true", help="Train with recorded speed.")
    parser.add_argument("--speed-kmh", type=float, help="Current speed for a single-frame prediction.")
    args = parser.parse_args()
    if args.sample_history is None:
        args.sample_history = args.history_frames
    if args.sample_history < args.history_frames:
        parser.error("--sample-history must be at least --history-frames")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not math.isfinite(args.hard_brake_weight) or args.hard_brake_weight < 1:
        parser.error("--hard-brake-weight must be finite and at least 1")
    if not math.isfinite(args.strong_steer_weight) or args.strong_steer_weight < 1:
        parser.error("--strong-steer-weight must be finite and at least 1")
    if bool(args.train_stints) != bool(args.validation_stint):
        parser.error("--train-stints and --validation-stint must be provided together")
    if args.train_stints:
        stints = [*args.train_stints, args.validation_stint]
        if len({stint.resolve() for stint in stints}) != len(stints):
            parser.error("Training and validation stint paths must be distinct")
    if args.predict and not args.checkpoint:
        parser.error("--predict requires --checkpoint")
    if args.check_gpu and args.device != "cuda":
        parser.error("--check-gpu requires --device cuda")
    device = check_device(args.device)
    if args.check_gpu:
        return
    if args.predict:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        speed_scale = saved.get("speed_scale_kmh")
        if speed_scale is not None and (args.speed_kmh is None or not math.isfinite(args.speed_kmh)):
            parser.error("This checkpoint requires a finite --speed-kmh.")
        speed = torch.tensor([[args.speed_kmh]], device=device) if speed_scale is not None else None
        history_frames = saved.get("history_frames", 1)
        paths = [*(args.previous_frames or []), args.predict]
        if len(paths) != history_frames:
            parser.error(f"This checkpoint requires {history_frames} frame(s).")
        model = DrivingCNN(speed_scale, history_frames).to(device).eval()
        model.load_state_dict(saved["model_state"])
        image = torch.cat([load_image(path, tuple(saved["image_size"]), saved["road_crop"])
                           for path in paths])
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(device), speed)[0].tolist()
        print(dict(zip(saved["controls"], prediction)))
        return
    train(args, device)


if __name__ == "__main__":
    main()
