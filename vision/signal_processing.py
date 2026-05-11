import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def extract_micro_movements(position_history, fps=30):
    """
    Separate macro head movement from micro facial movements via low-pass subtraction.
    Returns (twitches: list[dict], velocity: np.ndarray).
    """
    positions = np.array(position_history, dtype=float)
    n = len(positions)
    if n < 15:
        return [], np.zeros(max(0, n - 1))

    nyq = fps / 2.0
    low = min(0.5 / nyq, 0.99)
    try:
        b, a = butter(2, low, btype="low")
        macro = filtfilt(b, a, positions, axis=0)
    except Exception:
        macro = positions

    micro = positions - macro
    velocity = np.linalg.norm(np.diff(micro, axis=0), axis=1)

    if len(velocity) < 3:
        return [], velocity

    mean_v, std_v = float(np.mean(velocity)), float(np.std(velocity))
    threshold = mean_v + 2.5 * std_v
    if threshold == 0:
        return [], velocity

    peaks, _ = find_peaks(velocity, height=threshold, distance=3)
    twitches = [{"frame": int(p), "amplitude": round(float(velocity[p]), 5)} for p in peaks]
    return twitches, velocity


def compute_blink_rate(ear_history, fps=30, threshold=0.21):
    """Count blinks in EAR history and return per-minute rate."""
    if len(ear_history) < fps:
        return 0.0

    blinks = 0
    in_blink = False
    for ear in ear_history:
        if ear < threshold and not in_blink:
            blinks += 1
            in_blink = True
        elif ear >= threshold:
            in_blink = False

    duration_min = len(ear_history) / fps / 60.0
    return blinks / duration_min if duration_min > 0 else 0.0
