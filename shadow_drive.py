"""Log live ACC controls and model predictions without sending any inputs."""
import argparse
import csv
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
from train_imitation import CONTROLS, DrivingCNN, check_device, prepare_image


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
    tensor = prepare_image(image, tuple(saved["image_size"]), saved["road_crop"])
    images = tensor.unsqueeze(0).to(device)
    speed = images.new_tensor([[speed_kmh]]) if model.speed_scale is not None else None
    return model(images, speed)[0].tolist()


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = DrivingCNN(saved.get("speed_scale_kmh")).to(device).eval()
    model.load_state_dict(saved["model_state"])
    return model, saved


def run(args):
    device = check_device("cuda")
    model, saved = load_model(args.checkpoint, device)
    comparison = None
    if args.compare_checkpoint:
        comparison = load_model(args.compare_checkpoint, device)
        print(f"Comparing {args.checkpoint} with {args.compare_checkpoint}")

    # Initialize inference before the capture clock starts.
    warmup = Image.new("RGB", (640, 360))
    predict(model, warmup, saved, device, 0.0)
    if comparison:
        other_model, other_saved = comparison
        predict(other_model, warmup, other_saved, device, 0.0)

    folder = Path("runs/shadow")
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "shadow_compare" if comparison else "shadow"
    log = folder / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    print(f"Target capture rate: {args.fps} Hz.")
    print("Switch to ACC on your primary monitor. Capture starts in 5 seconds.")
    time.sleep(5)
    acc = accSharedMemory()
    cadence = CaptureCadence(args.fps, args.seconds)
    missing = 0
    try:
        with MSS() as screen, log.open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            columns = ["elapsed_s", "telemetry_delay_ms", "prediction_latency_ms",
                       "speed_kmh", *[f"actual_{key}" for key in CONTROLS],
                       *[f"predicted_{key}" for key in CONTROLS]]
            if comparison:
                columns.extend(["comparison_latency_ms",
                                *[f"comparison_{key}" for key in CONTROLS]])
            writer.writerow(columns)
            monitor = screen.primary_monitor
            for before in cadence:
                frame = screen.grab(monitor)
                capture = (before + time.perf_counter()) / 2
                state = acc.read_shared_memory()
                telemetry = time.perf_counter()
                if state is None:
                    missing += 1
                else:
                    car = state.Physics
                    image = prepare_capture(frame)
                    guesses = predict(model, image, saved, device, car.speed_kmh)
                    ready = time.perf_counter()
                    row = [capture - cadence.start, 1000 * (telemetry - capture),
                           1000 * (ready - capture), car.speed_kmh,
                           car.steer_angle, car.gas, car.brake, *guesses]
                    if comparison:
                        other_model, other_saved = comparison
                        other_guesses = predict(other_model, image, other_saved, device, car.speed_kmh)
                        row.extend([1000 * (time.perf_counter() - capture), *other_guesses])
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
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds and --fps must be positive")
    run(args)


if __name__ == "__main__":
    main()
