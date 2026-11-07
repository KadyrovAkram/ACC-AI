from pathlib import Path

import cv2
import pandas as pd

sessions = sorted(
    Path("data").glob("vision_session_*")
)

if not sessions:
    raise SystemExit("No vision sessions found.")

session = sessions[-1]

df = pd.read_csv(session / "telemetry.csv")

print(f"Session: {session}")
print(f"Recorded samples: {len(df)}")

missing = [
    name for name in df["frame"]
    if not (session / "frames" / name).exists()
]

print(f"Missing images: {len(missing)}")

if df.empty:
    raise SystemExit("Dataset is empty.")

print("\nTelemetry statistics:")
print(df[[
    "speed_kmh", "throttle", "brake", "steering",
    "sync_delay_ms"
]].describe())

# Display the middle frame
sample = df.iloc[len(df) // 2]

image = cv2.imread(
    str(session / "frames" / sample["frame"])
)

if image is None:
    raise SystemExit("Could not open the sample image.")

print("\nSample data:")
print(sample.to_string())

cv2.imshow("ACC AI - Sample Frame", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
