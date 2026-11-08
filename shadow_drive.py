"""Log live ACC controls and model predictions without sending any inputs."""
import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
import time

import cv2
from mss import MSS
import numpy as np
from PIL import Image
from pyaccsharedmemory import accSharedMemory
import torch

from capture_cadence import CaptureCadence
from train_imitation import (CONTROLS, HISTORY_MAX_GAP_S, BrakeClassifier, DrivingCNN,
                             check_device, prepare_image)


def prepare_capture(screenshot):
    # Match the recorder's resize and JPEG settings before applying the model crop.
    bgr = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode the captured frame.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


@torch.inference_mode()
def predict(model, image, saved, device, speed_kmh):
    frames = (image,) if isinstance(image, Image.Image) else tuple(image)
    frames = frames[-model.history_frames:]
    if len(frames) != model.history_frames:
        raise ValueError(f"This checkpoint requires {model.history_frames} consecutive frames.")
    tensor = torch.cat([prepare_image(frame, tuple(saved["image_size"]), saved["road_crop"])
                        for frame in frames])
    images = tensor.unsqueeze(0).to(device)
    speed = images.new_tensor([[speed_kmh]]) if model.speed_scale is not None else None
    return model(images, speed)[0].tolist()


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = DrivingCNN(saved.get("speed_scale_kmh"), saved.get("history_frames", 1)).to(device).eval()
    model.load_state_dict(saved["model_state"])
    return model, saved


@torch.inference_mode()
def brake_score(model, image, saved, device):
    tensor = prepare_image(image, tuple(saved["image_size"]), saved["road_crop"])
    return model(tensor.unsqueeze(0).to(device)).sigmoid().item()


def run(args):
    device = check_device("cuda")
    model, saved = load_model(args.checkpoint, device)
    comparison = None
    if args.compare_checkpoint:
        comparison = load_model(args.compare_checkpoint, device)
        print(f"Comparing {args.checkpoint} with {args.compare_checkpoint}")

    required_history = max(model.history_frames, comparison[0].history_frames if comparison else 1)
    if required_history > 1 and args.fps != 10:
        raise SystemExit("Use --fps 10 with a frame-history checkpoint.")

    brake_classifier = None
    if args.brake_classifier:
        brake_checkpoint = torch.load(args.brake_classifier, map_location=device, weights_only=True)
        brake_classifier = BrakeClassifier().to(device).eval()
        brake_classifier.load_state_dict(brake_checkpoint["model_state"])
        print(f"Brake classifier: {args.brake_classifier} at cutoff {brake_checkpoint['threshold']:.3f}")

    # Initialize inference before the capture clock starts.
    warmup = Image.new("RGB", (640, 360))
    predict(model, [warmup] * required_history, saved, device, 0.0)
    if comparison:
        other_model, other_saved = comparison
        predict(other_model, [warmup] * required_history, other_saved, device, 0.0)

    if brake_classifier is not None:
        brake_score(brake_classifier, warmup, brake_checkpoint, device)

    folder = Path("runs/shadow")
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "shadow_compare" if comparison or brake_classifier is not None else "shadow"
    log = folder / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    print(f"Target capture rate: {args.fps} Hz.")
    print("Switch to ACC on your primary monitor. Capture starts in 5 seconds.")
    time.sleep(5)
    acc = accSharedMemory()
    cadence = CaptureCadence(args.fps, args.seconds)
    missing = 0
    waiting_for_history = 0
    history = deque(maxlen=required_history)
    last_capture = None
    try:
        with MSS() as screen, log.open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            columns = ["elapsed_s", "telemetry_delay_ms", "prediction_latency_ms",
                       "speed_kmh", *[f"actual_{key}" for key in CONTROLS],
                       *[f"predicted_{key}" for key in CONTROLS]]
            if comparison:
                columns.extend(["comparison_latency_ms",
                                *[f"comparison_{key}" for key in CONTROLS]])
            if brake_classifier is not None:
                columns.extend(["classifier_brake", "classifier_score", "classifier_latency_ms"])
            writer.writerow(columns)
            monitor = screen.primary_monitor
            for before in cadence:
                frame = screen.grab(monitor)
                capture = (before + time.perf_counter()) / 2
                state = acc.read_shared_memory()
                telemetry = time.perf_counter()
                if state is None:
                    missing += 1
                    history.clear()
                else:
                    car = state.Physics
                    image = prepare_capture(frame)
                    if last_capture is not None and capture - last_capture > HISTORY_MAX_GAP_S:
                        history.clear()
                    last_capture = capture
                    history.append(image)
                    if len(history) < required_history:
                        waiting_for_history += 1
                        continue
                    guesses = predict(model, history, saved, device, car.speed_kmh)
                    ready = time.perf_counter()
                    row = [capture - cadence.start, 1000 * (telemetry - capture),
                           1000 * (ready - capture), car.speed_kmh,
                           car.steer_angle, car.gas, car.brake, *guesses]
                    if comparison:
                        other_model, other_saved = comparison
                        other_guesses = predict(other_model, history, other_saved, device, car.speed_kmh)
                        row.extend([1000 * (time.perf_counter() - capture), *other_guesses])
                    if brake_classifier is not None:
                        score = brake_score(brake_classifier, image, brake_checkpoint, device)
                        row.extend([int(score >= brake_checkpoint["threshold"]), score,
                                    1000 * (time.perf_counter() - capture)])
                    writer.writerow(row)
                    cadence.timestamps.append(capture)
                    if len(cadence.timestamps) % (5 * args.fps) == 0:
                        file.flush()
                        print(f"Logged {len(cadence.timestamps)} frames", flush=True)
    except KeyboardInterrupt:
        print("Stopped recording.")
    finally:
        acc.close()
    print(f"Logged {len(cadence.timestamps)} frames, skipped {missing} without telemetry.")
    if required_history > 1:
        print(f"Skipped {waiting_for_history} frames while building consecutive history.")
    cadence.summary()
    print(f"Log: {log.resolve()}")
    if not cadence.timestamps:
        raise SystemExit("No telemetry logged. Start a live drive in ACC and try again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/milestone5/best.pt"))
    parser.add_argument("--compare-checkpoint", type=Path,
                        help="Log a second model's predictions for the same frames.")
    parser.add_argument("--brake-classifier", type=Path,
                        help="Log a brake-only classifier alongside the driving model.")
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds and --fps must be positive")
    run(args)


if __name__ == "__main__":
    main()
