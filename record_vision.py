"""Record synchronized ACC screenshots and telemetry for imitation learning."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import cv2
from mss import MSS
import numpy as np
from pyaccsharedmemory import accSharedMemory

from capture_cadence import CaptureCadence


FPS = 10
DURATION = 240
START_DELAY = 5
IMAGE_SIZE = (640, 360)


def run(args):
    session = args.data / f"vision_session_{datetime.now():%Y%m%d_%H%M%S_%f}"
    frames = session / "frames"
    frames.mkdir(parents=True)
    print(f"Recording for {args.seconds} seconds at a target of {args.fps} Hz.")
    print(f"Switch to ACC on your primary monitor within {START_DELAY} seconds.")
    time.sleep(START_DELAY)
    cadence = CaptureCadence(args.fps, args.seconds)
    missing = 0
    acc = accSharedMemory()
    try:
        with MSS() as screen, (session / "telemetry.csv").open("x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "timestamp", "capture_time", "telemetry_time", "sync_delay_ms",
                             "speed_kmh", "rpm", "gear", "throttle", "brake", "steering"])
            monitor = screen.primary_monitor
            for before in cadence:
                screenshot = screen.grab(monitor)
                capture = (before + time.perf_counter()) / 2
                state = acc.read_shared_memory()
                telemetry = time.perf_counter()
                if state is None:
                    missing += 1
                    continue
                car = state.Physics
                bgr = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_BGRA2BGR)
                image = cv2.resize(bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                name = f"frame_{len(cadence.timestamps):06d}.jpg"
                if not cv2.imwrite(str(frames / name), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    raise RuntimeError(f"Could not save {frames / name}")
                writer.writerow([name, time.time(), capture, telemetry, 1000 * (telemetry - capture),
                                 car.speed_kmh, car.rpm, car.gear, car.gas, car.brake, car.steer_angle])
                cadence.timestamps.append(capture)
                if len(cadence.timestamps) % (5 * args.fps) == 0:
                    file.flush()
                    print(f"Recorded {len(cadence.timestamps)} frames | Speed: {car.speed_kmh:.1f} km/h")
    except KeyboardInterrupt:
        print("Stopped recording.")
    finally:
        acc.close()
    print(f"Finished! Saved {len(cadence.timestamps)} frames, skipped {missing} without telemetry.")
    cadence.summary()
    print(f"Dataset: {session.resolve()}")
    if not cadence.timestamps:
        raise SystemExit("No telemetry logged. Start a live drive in ACC and try again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seconds", type=int, default=DURATION)
    parser.add_argument("--data", type=Path, help="Root directory for recording sessions.")
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds and --fps must be positive")
    if args.data is None:
        args.data = Path("data" if args.fps == 5 else f"data_{args.fps}hz")
    run(args)


if __name__ == "__main__":
    main()
