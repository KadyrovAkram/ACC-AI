# ACC-AI

An Assetto Corsa Competizione imitation-learning project. The goal is to drive a valid autonomous lap at Monza within 2% of a published benchmark for the same car and conditions.

The project records cockpit screenshots and driver inputs, trains PyTorch models, and compares their live predictions with a human driver in shadow mode. The current scripts log predictions while the human controls the car.

## Setup

Use Windows, Python 3.12, and an NVIDIA GPU for the training and shadow-mode commands below. The setup was tested on an RTX 5060 Laptop GPU with PyTorch 2.10.0 and CUDA 12.8.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe train_imitation.py --check-gpu
```

## Record and train

Start a solo ACC session on the primary monitor. Keep the car, camera view, and driving aids consistent across recordings. Run the recorder and switch back to ACC during the five-second countdown.

```powershell
.\.venv\Scripts\python.exe -u record_vision.py --fps 10 --seconds 240
```

Each session contains a `telemetry.csv` and a `frames` directory. The recorder saves 10 Hz sessions under `data_10hz` and reports the actual cadence. Record several separate stints. The imitation trainer uses the newest session or sessions for validation when several are present.

```powershell
.\.venv\Scripts\python.exe -u train_imitation.py --data data_10hz --hard-brake-weight 3 --output runs\brake_weighted
```

Use a fresh output directory for each training run. For a brake classifier experiment, use explicit, separate stints for training, threshold selection, and final evaluation. Run `train_brake_classifier.py --help` for the multi-stint and label-shift options.

## Compare predictions while you drive

```powershell
.\.venv\Scripts\python.exe -u shadow_drive.py --fps 10 --seconds 60 --checkpoint runs\brake_weighted\best.pt
```

Pass `--compare-checkpoint` with a second imitation model to log both predictions on the same frames. Use the CSV path printed by shadow mode with `report_braking.py` to score missed presses, onset and release timing, and isolated brake calls.

```powershell
.\.venv\Scripts\python.exe report_braking.py "runs\shadow\YOUR_LOG.csv"
```

## Project layout

| Script | Purpose |
| --- | --- |
| `telemetry.py`, `record_data.py`, `analyze_data.py` | Read, record, and inspect ACC telemetry |
| `record_vision.py`, `capture_cadence.py` | Pair screenshots with telemetry at a measured capture rate |
| `validate_vision.py` | Inspect a legacy recording under `data` |
| `train_imitation.py` | Train steering, throttle, and brake regressors |
| `train_brake_classifier.py` | Train and evaluate a brake on/off classifier |
| `shadow_drive.py`, `report_braking.py` | Compare live predictions and summarize braking events |

Recordings, model weights, run logs, and Python environments are kept locally. See [experiment notes](docs/experiments.md) for the results so far.
