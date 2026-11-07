
import time
from pyaccsharedmemory import accSharedMemory

acc = accSharedMemory()

print("Reading ACC telemetry. Press Ctrl+C to stop.")

try:
    while True:
        data = acc.read_shared_memory()

        if data is not None:
            car = data.Physics

            print(
                f"Speed: {car.speed_kmh:.1f} km/h | "
                f"RPM: {car.rpm} | "
                f"Throttle: {car.gas:.2f} | "
                f"Brake: {car.brake:.2f} | "
                f"Steering: {car.steer_angle:.2f}"
            )
        else:
            print("No telemetry available. Is ACC running?")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped recording.")

finally:
    acc.close()
