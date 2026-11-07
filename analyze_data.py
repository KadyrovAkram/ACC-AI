
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

files = sorted(Path("data").glob("drive_*.csv"))

if not files:
    raise SystemExit("No recorded driving data found.")

df = pd.read_csv(files[-1])

print("\nDataset information:")
print(df.describe())
print(f"\nTotal samples: {len(df)}")

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

time_s = df["timestamp"] - df["timestamp"].iloc[0]

signals = [
    ("speed_kmh", "Speed (km/h)"),
    ("throttle", "Throttle"),
    ("brake", "Brake"),
    ("steering", "Steering")
]

for ax, (column, label) in zip(axes, signals):
    ax.plot(time_s, df[column])
    ax.set_ylabel(label)
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()
