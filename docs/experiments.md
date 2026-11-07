# Monza experiment notes

The goal is a valid autonomous lap within 2% of a published benchmark for the same car and conditions. First verify manual takeover and complete a clean solo lap, then demonstrate repeatability before chasing lap time.

## What the runs showed

- The first 60-second run verified CUDA training and checkpoint reload. Its validation segment contained no braking, and a constant baseline outscored the CNN.
- With separate full stints, the CNN beat the constant baseline on all three controls. Weighting errors during hard braking improved predicted pedal strength.
- In a paired 5 Hz shadow run, the brake-weighted model detected 23 of 23 scored brake events, versus 21 for the original. It also tended to release later.
- The recorder and paired shadow mode sustained about 10 Hz in live ACC tests.
- On the first reserved 10 Hz stint, the weighted model detected 21 of 22 brake events versus 19 for the binary classifier. The classifier produced fewer isolated brake calls, two versus five before excluding startup. The models used different training recordings.
- In a follow-up on the same stint, shifting the classifier target 100 ms left median onset near 100 ms late. A 200 ms shift moved it to about 200 ms early. Neither shift increased the number of detected events.

## Next evaluation

Use several fresh stints for training and validation. Hold out a new stint until the model, threshold, and event scoring rules are frozen. Track missed presses, responses after the actual release, early or late release, and isolated brake calls. Score startup separately from the driving portion.
