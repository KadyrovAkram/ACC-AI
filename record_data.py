import csv
import time
from datetime import datetime
from pathlib import Path

from pyaccsharedmemory import accSharedMemory

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

filename = DATA_DIR / f"drive_{datetime.now():%Y%m%d_%H%M%S}.csv"

acc = accSharedMemory()

columns = [
    "timestamp",
    "speed_kmh",
    "rpm",
    "gear",
    "throttle",
    "brake",
    "steering",
    "heading",
    "pitch",
    "roll",
    "lap_position"
]

print("ACC AI Data Recorder")
print(f"Saving data to: {filename}")
print("Press Ctrl+C to stop recording.")

samples = 0

try:
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        while True:
            data = acc.read_shared_memory()

            if data is not None:
                p = data.Physics
                g = data.Graphics

                writer.writerow([
                    time.time(),
                    p.speed_kmh,
                    p.rpm,
                    p.gear,
                    p.gas,
                    p.brake,
                    p.steer_angle,
                    p.heading,
                    p.pitch,
                    p.roll,
                    g.normalized_car_position
                ])

                samples += 1

                if samples % 20 == 0:
                    file.flush()
                    print(
                        f"Samples: {samples} | "
                        f"Speed: {p.speed_kmh:.1f} km/h"
                    )

            time.sleep(0.05)

except KeyboardInterrupt:
    print(f"\nRecording stopped. Saved {samples} samples.")

finally:
    acc.close()
