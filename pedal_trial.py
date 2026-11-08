"""Short solo ACC pedal trials with manual steering and a keyboard override."""
import argparse
from collections import Counter
import csv
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime
import math
from pathlib import Path
import threading
import time

from mss import MSS
from PIL import Image
from pyaccsharedmemory import ACC_SESSION_TYPE, ACC_STATUS, accSharedMemory

from capture_cadence import CaptureCadence
from shadow_drive import load_model, predict, prepare_capture
from train_imitation import check_device


ARM = 0x77  # F8
STOP = 0x78  # F9
STALE_SECONDS = 0.25
BRAKE_ON = 0.25
BRAKE_OFF = 0.15
BRAKE_RELEASE_SECONDS = 0.20
THROTTLE_COOLDOWN_SECONDS = 0.20
BRAKE_PULSE_SECONDS = 0.20
SOLO_MODES = {ACC_SESSION_TYPE.ACC_PRACTICE, ACC_SESSION_TYPE.ACC_HOTLAP,
              ACC_SESSION_TYPE.ACC_HOTSTINT}


class KeyboardInput(c.Structure):
    _fields_ = [("vk", w.WORD), ("scan", w.WORD), ("flags", w.DWORD),
                ("time", w.DWORD), ("extra", c.c_size_t)]


class MouseInput(c.Structure):
    _fields_ = [("x", w.LONG), ("y", w.LONG), ("data", w.DWORD),
                ("flags", w.DWORD), ("time", w.DWORD), ("extra", c.c_size_t)]


class InputPayload(c.Union):
    _fields_ = [("keyboard", KeyboardInput), ("mouse", MouseInput)]


class Input(c.Structure):
    _fields_ = [("type", w.DWORD), ("payload", InputPayload)]


class WindowsKeys:
    def __init__(self, throttle, brake):
        self.user = c.WinDLL("user32", use_last_error=True)
        self.kernel = c.WinDLL("kernel32", use_last_error=True)
        self.user.GetForegroundWindow.restype = w.HWND
        self.user.GetAsyncKeyState.argtypes = [c.c_int]
        self.user.GetAsyncKeyState.restype = c.c_short
        self.user.GetWindowThreadProcessId.argtypes = [w.HWND, c.POINTER(w.DWORD)]
        self.user.MonitorFromWindow.argtypes = [w.HWND, w.DWORD]
        self.user.MonitorFromWindow.restype = w.HANDLE
        self.user.MonitorFromPoint.argtypes = [w.POINT, w.DWORD]
        self.user.MonitorFromPoint.restype = w.HANDLE
        self.user.MapVirtualKeyW.argtypes = [w.UINT, w.UINT]
        self.user.MapVirtualKeyW.restype = w.UINT
        self.user.SendInput.argtypes = [w.UINT, c.POINTER(Input), c.c_int]
        self.user.SendInput.restype = w.UINT
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR,
                                                          c.POINTER(w.DWORD)]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.vk = {"throttle": ord(throttle), "brake": ord(brake)}
        self.scans = {name: self.user.MapVirtualKeyW(code, 0) for name, code in self.vk.items()}
        if not all(self.scans.values()):
            raise ValueError("A pedal key could not be mapped.")

    def foreground(self):
        return self.user.GetForegroundWindow()

    def down(self, code):
        return bool(self.user.GetAsyncKeyState(code) & 0x8000)

    def acc_window(self):
        hwnd = self.foreground()
        if not hwnd or self.user.MonitorFromWindow(hwnd, 2) != self.user.MonitorFromPoint(w.POINT(), 1):
            return None
        pid = w.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, c.byref(pid))
        handle = self.kernel.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return None
        try:
            path = c.create_unicode_buffer(1024)
            length = w.DWORD(len(path))
            if self.kernel.QueryFullProcessImageNameW(handle, 0, path, c.byref(length)):
                if Path(path.value).name.lower() == "ac2-win64-shipping.exe":
                    return hwnd
        finally:
            self.kernel.CloseHandle(handle)
        return None

    def send(self, pedal, pressed):
        flags = 0x0008 | (0 if pressed else 0x0002)
        event = Input(1, InputPayload(keyboard=KeyboardInput(0, self.scans[pedal], flags, 0, 0)))
        if self.user.SendInput(1, c.byref(event), c.sizeof(event)) != 1:
            raise RuntimeError(f"Windows rejected the {'press' if pressed else 'release'} for {pedal}.")


class BrakePulse:
    """Translate a brake duty request into key holds at the watchdog cadence."""
    def __init__(self):
        self.started = None
        self.duty = 0.0

    def update(self, pedal, duty, now):
        if pedal != "brake" or duty <= 0:
            self.started = None
            return None if pedal == "brake" else pedal
        if self.started is None or now - self.started >= BRAKE_PULSE_SECONDS:
            self.started = now
            self.duty = duty
        if now - self.started < BRAKE_PULSE_SECONDS * self.duty:
            return "brake"
        return None


def brake_duty(pedal, score, cap):
    if pedal != "brake":
        return 0.0
    if cap is None:
        return 1.0
    return min(score, cap) if score > BRAKE_OFF else 0.0


class PedalGuard:
    """Own injected keys in one thread, independently of inference."""
    def __init__(self, keys, live, arm_seconds):
        self.keys = keys
        self.live = live
        self.arm_seconds = arm_seconds
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = (None, 0.0, None, 0.0)
        self.input_issue = None
        self.armed = False
        self.held = None
        self.reason = None
        self.armed_at = None
        self.armed_duration = 0.0

    def offer(self, hwnd, captured, pedal, issue=None, *, duty=1.0):
        with self.lock:
            self.latest = (hwnd, captured, pedal, duty)
            self.input_issue = issue
            return self.armed

    def note_stale_telemetry(self):
        with self.lock:
            self.input_issue = "ACC physics packet stopped updating"

    def apply(self, pedal):
        if pedal == self.held:
            return
        if self.held is not None:
            if self.live:
                self.keys.send(self.held, False)
            self.held = None
        if pedal is not None:
            if self.live:
                self.keys.send(pedal, True)
            self.held = pedal

    def watch(self):
        previous_arm = self.keys.down(ARM)
        expires = 0.0
        armed_window = None
        pulse = BrakePulse()
        try:
            while not self.stop.wait(.01):
                now = time.perf_counter()
                with self.lock:
                    hwnd, captured, pedal, duty = self.latest
                    foreground = self.keys.foreground()
                    valid = hwnd and foreground == hwnd and now - captured <= STALE_SECONDS
                    arm = self.keys.down(ARM)
                    if self.keys.down(STOP):
                        self.reason = "F9 override"
                        break
                    if self.armed and (not valid or hwnd != armed_window or now >= expires):
                        if now >= expires:
                            self.reason = "trial timer ended"
                        elif foreground != armed_window:
                            self.reason = "ACC lost foreground focus"
                        else:
                            self.reason = self.input_issue or "no fresh prediction for 250 ms"
                        break
                    if arm and not previous_arm and not valid and not self.armed:
                        print(f"F8 ignored: {self.input_issue or 'waiting for a fresh ACC prediction'}.",
                              flush=True)
                    if arm and not previous_arm and valid and not self.armed:
                        if not any(self.keys.down(code) for code in self.keys.vk.values()):
                            self.armed = True
                            armed_window = hwnd
                            self.armed_at = now
                            expires = now + self.arm_seconds
                            print(f"Armed for up to {self.arm_seconds:g} s. Steer manually. F9 stops.", flush=True)
                    previous_arm = arm
                    if self.armed:
                        self.apply(pulse.update(pedal, duty, now))
        except Exception as exc:
            self.reason = str(exc)
        finally:
            with self.lock:
                self.armed = False
                if self.armed_at is not None:
                    self.armed_duration = time.perf_counter() - self.armed_at
                try:
                    self.apply(None)
                except RuntimeError as exc:
                    self.reason = f"{self.reason or 'Stopped'}; {exc} Tap the pedal key to release it."
                finally:
                    self.stop.set()


def telemetry_issue(state, speed_limit):
    if state is None:
        return "ACC telemetry unavailable"
    if state.Static.is_online:
        return "online session"
    if state.Graphics.status != ACC_STATUS.ACC_LIVE:
        return f"ACC is not live ({state.Graphics.status.name})"
    if state.Graphics.session_type not in SOLO_MODES:
        return f"unsupported ACC mode ({state.Graphics.session_type})"
    if state.Graphics.is_in_pit:
        return "car is in the pits"
    speed = state.Physics.speed_kmh
    if not math.isfinite(speed) or speed < 0:
        return "invalid ACC speed telemetry"
    if speed > speed_limit:
        return f"speed cap reached ({speed:.1f} km/h, limit {speed_limit:g})"
    return None


def choose_pedal(throttle, brake):
    if not all(math.isfinite(value) for value in (throttle, brake)):
        raise ValueError("The model returned a non-finite pedal prediction.")
    if brake >= BRAKE_ON:
        return "brake"
    if throttle >= .5:
        return "throttle"
    return None


class PedalFilter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.braking = False
        self.release_since = None
        self.throttle_after = 0.0

    def update(self, raw_pedal, brake, now):
        if raw_pedal == "brake":
            self.braking = True
            self.release_since = None
            return "brake"
        if self.braking:
            if brake > BRAKE_OFF:
                self.release_since = None
                return "brake"
            if self.release_since is None:
                self.release_since = now
            if now - self.release_since < BRAKE_RELEASE_SECONDS:
                return "brake"
            self.braking = False
            self.throttle_after = now + THROTTLE_COOLDOWN_SECONDS
        if now < self.throttle_after:
            return None
        return raw_pedal


def run(args):
    keys = WindowsKeys(args.throttle_key, args.brake_key)
    device = check_device("cuda")
    model, saved = load_model(args.checkpoint, device)
    predict(model, Image.new("RGB", (640, 360)), saved, device, 0.0)
    folder = Path("runs/pedal_trials")
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"pedals_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    guard = PedalGuard(keys, args.live, args.arm_seconds)
    worker = threading.Thread(target=guard.watch)
    print(f"{'LIVE keyboard input' if args.live else 'DRY RUN'} | "
          f"F8 arms for up to {args.arm_seconds:g} s below {args.max_speed:g} km/h; F9 stops.")
    if args.brake_duty_cap is not None:
        print(f"Brake pulses enabled, up to {args.brake_duty_cap:.0%} key-down time.")
    print("Switch to solo Practice, Hotlap, or Hotstint on the primary monitor. Ready in 5 seconds.")
    time.sleep(5)
    acc = accSharedMemory()
    previous_packet = None
    previous_capture = None
    pedal_filter = PedalFilter()
    cadence = CaptureCadence(10, args.seconds)
    rejected = Counter()
    try:
        with MSS() as screen, log.open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["elapsed_s", "armed", "speed_kmh", "acc_throttle", "acc_brake",
                             "predicted_throttle", "predicted_brake", "requested_pedal", "phase", "raw_pedal", "brake_duty"])
            monitor = screen.primary_monitor
            worker.start()
            announced = False
            for before in cadence:
                if guard.stop.is_set():
                    break
                hwnd = keys.acc_window()
                if hwnd is None:
                    pedal_filter.reset()
                    issue = "ACC left the primary monitor or lost focus"
                    rejected[issue] += 1
                    guard.offer(None, 0, None, issue)
                    continue
                frame = screen.grab(monitor)
                captured = (before + time.perf_counter()) / 2
                state = acc.read_shared_memory()
                issue = telemetry_issue(state, args.max_speed)
                if issue or keys.foreground() != hwnd:
                    pedal_filter.reset()
                    issue = issue or "ACC lost foreground focus"
                    rejected[issue.split(" (")[0]] += 1
                    guard.offer(None, 0, None, issue)
                    continue
                car = state.Physics
                if car.packed_id == previous_packet:
                    rejected["ACC physics packet repeated"] += 1
                    guard.note_stale_telemetry()
                    continue
                previous_packet = car.packed_id
                _, throttle, brake = predict(model, prepare_capture(frame), saved, device, car.speed_kmh)
                if previous_capture is not None and captured - previous_capture > STALE_SECONDS:
                    pedal_filter.reset()
                previous_capture = captured
                raw_pedal = choose_pedal(throttle, brake)
                pedal = pedal_filter.update(raw_pedal, brake, captured)
                duty = brake_duty(pedal, brake, args.brake_duty_cap)
                armed = guard.offer(hwnd, captured, pedal, duty=duty)
                if not announced:
                    announced = True
                    print("Eligible ACC session found. Press F8 when ready.", flush=True)
                writer.writerow([captured - cadence.start, int(armed), car.speed_kmh, car.gas,
                                 car.brake, throttle, brake, pedal or "coast", "prediction", raw_pedal or "coast", duty])

            # The worker has released its key before setting stop. Keep logging
            # briefly so ACC can confirm that the pedal value actually fell.
            if guard.stop.is_set() and guard.armed_at is not None:
                for sampled in CaptureCadence(10, 0.5):
                    if keys.acc_window() is None:
                        break
                    state = acc.read_shared_memory()
                    if state is None:
                        continue
                    car = state.Physics
                    writer.writerow([sampled - cadence.start, 0, car.speed_kmh, car.gas,
                                     car.brake, "", "", "", "after_release", "", 0.0])
    except KeyboardInterrupt:
        print("Stopped from terminal.")
    finally:
        guard.stop.set()
        if worker.ident is not None:
            worker.join()
        acc.close()
    print(f"Stopped: {guard.reason or 'capture ended'}. Armed for {guard.armed_duration:.2f} s.")
    if rejected and guard.armed_at is None:
        print("Skipped samples: " + ", ".join(f"{count} {reason}" for reason, count in rejected.most_common()))
    print(f"Log: {log.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/brake_weighted/best.pt"))
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--arm-seconds", type=float, default=8)
    parser.add_argument("--max-speed", type=float, default=120)
    parser.add_argument("--live", action="store_true", help="Send pedal keys after F8.")
    parser.add_argument("--throttle-key")
    parser.add_argument("--brake-key")
    parser.add_argument("--brake-duty-cap", type=float,
                        help="Pulse the brake key using the model score, capped at this fraction (0–1).")
    args = parser.parse_args()
    if args.brake_duty_cap is not None and not 0 < args.brake_duty_cap <= 1:
        parser.error("--brake-duty-cap must be in (0, 1].")
    if args.live and (not args.throttle_key or not args.brake_key):
        parser.error("--live requires both ACC pedal key bindings.")
    args.throttle_key = (args.throttle_key or "W").upper()
    args.brake_key = (args.brake_key or "S").upper()
    if any(len(key) != 1 or not "A" <= key <= "Z" for key in (args.throttle_key, args.brake_key)):
        parser.error("Use one letter A-Z for each pedal key.")
    if args.throttle_key == args.brake_key:
        parser.error("Throttle and brake must use different keys.")
    if args.seconds <= 0 or not 0 < args.arm_seconds <= 15 or not 0 < args.max_speed <= 300:
        parser.error("Use positive durations, at most 15 armed seconds, and a speed cap up to 300 km/h.")
    run(args)


if __name__ == "__main__":
    main()
