"""Schedule live samples against a monotonic clock and summarize logged cadence."""
import math
from statistics import median
import time


class CaptureCadence:
    def __init__(self, fps, seconds):
        self.fps = fps
        self.seconds = seconds
        self.skipped = 0
        self.timestamps = []

    def __iter__(self):
        self.start = time.perf_counter()
        interval = 1 / self.fps
        deadline = self.start + self.seconds
        slot = 0
        while (scheduled := self.start + slot * interval) < deadline:
            time.sleep(max(0, scheduled - time.perf_counter()))
            before = time.perf_counter()
            if before >= deadline:
                break
            yield before
            slot += 1
            # Skip expired slots after slow work instead of taking catch-up frames.
            first_future_slot = min(math.ceil((time.perf_counter() - self.start) / interval),
                                    math.ceil(self.seconds * self.fps))
            if first_future_slot > slot:
                self.skipped += first_future_slot - slot
                slot = first_future_slot

    def summary(self):
        print(f"Target {self.fps} Hz | skipped capture slots: {self.skipped}")
        if len(self.timestamps) < 2:
            print("Need at least two logged frames to measure cadence.")
            return
        gaps = [b - a for a, b in zip(self.timestamps, self.timestamps[1:])]
        rate = len(gaps) / (self.timestamps[-1] - self.timestamps[0])
        print(f"Logged {rate:.2f} Hz | median gap {median(gaps)*1000:.0f} ms | "
              f"largest gap {max(gaps)*1000:.0f} ms")
