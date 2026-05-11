"""
MindScape — Somatic Stream Demo  (Robust v3)

Fixes over v2:
  • Head pose stabilization via solvePnP — all landmark motion is now
    head-motion-corrected before micro-movement analysis
  • Breathing via PCA on multi-ROI optical flow + FFT peak (not mean-flow)
  • Swallow via LK sparse tracking + IDLE→RISING→PEAK→FALLING state machine
  • rPPG heart-rate from forehead green channel + bandpass FFT
  • 30-second adaptive calibration — personal baseline z-score thresholds
  • Triangle-deformation strain added to twitch detection
  • All state machines properly implemented (no cooldown-counter hacks)
  • Gaze velocity normalized by inter-pupillary distance (resolution-invariant)

Run:  python3 demo_somatic_stream.py
Open: http://localhost:5001/video_html
"""
import os, sys, queue, threading, time
from collections import deque, defaultdict

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response
from scipy.signal import butter, filtfilt, find_peaks

FPS = 30   # assumed camera frame rate

# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
_ZONES = {
    "r_eye_outer": 33,  "r_eye_inner": 133, "r_eye_top": 159, "r_eye_bot": 145,
    "l_eye_outer": 362, "l_eye_inner": 263, "l_eye_top": 386, "l_eye_bot": 374,
    "r_brow_in":   107, "r_brow_mid":  66,  "r_brow_out": 46,
    "l_brow_in":   336, "l_brow_mid":  296, "l_brow_out": 276,
    "nose_tip":    1,   "r_nostril":   129, "l_nostril":  358,
    "lip_r":       61,  "lip_l":       291, "lip_top":    13,  "lip_bot": 14,
    "r_cheek":     116, "l_cheek":     345,
    "r_jaw":       172, "l_jaw":       397, "chin":       152,
    "r_temple":    234, "l_temple":    454,
}

# Triangle triplets for strain measurement (neighboring zones that should be rigid)
_STRAIN_TRIS = [
    ("r_eye_outer", "r_eye_inner", "r_eye_top"),
    ("l_eye_outer", "l_eye_inner", "l_eye_top"),
    ("r_brow_in",   "r_brow_mid",  "r_brow_out"),
    ("l_brow_in",   "l_brow_mid",  "l_brow_out"),
    ("lip_r",       "lip_l",       "lip_top"),
    ("r_cheek",     "nose_tip",    "lip_r"),
    ("l_cheek",     "nose_tip",    "lip_l"),
    ("r_jaw",       "chin",        "lip_bot"),
    ("l_jaw",       "chin",        "lip_bot"),
]

# Rigid landmarks for head-pose reference (bony landmarks, minimal soft-tissue motion)
_RIGID = [6, 168, 197, 195, 5]   # nose bridge + root

# solvePnP reference model (3D coordinates in mm, standard head model)
_FACE_3D = np.array([
    [0.0,    0.0,    0.0],   # 1  nose tip
    [0.0,  -63.6,  -12.5],   # 152 chin
    [-43.3,  32.7,  -26.0],  # 263 left eye outer
    [43.3,   32.7,  -26.0],  # 33  right eye outer
    [-28.9, -28.9,  -24.1],  # 291 left mouth corner
    [28.9,  -28.9,  -24.1],  # 61  right mouth corner
], dtype=np.float64)
_PNP_IDX = [1, 152, 263, 33, 291, 61]

# Eye / iris indices
_R_EAR_PTS = (33, 160, 158, 133, 153, 144)
_L_EAR_PTS = (362, 385, 387, 263, 373, 380)
_R_IRIS, _L_IRIS = 468, 473
_R_EYE_OUT, _R_EYE_IN = 33, 133
_L_EYE_OUT, _L_EYE_IN = 362, 263

# Contour sequences for drawing
_EYE_R_C = [33, 160, 158, 133, 153, 144]
_EYE_L_C = [362, 385, 387, 263, 373, 380]
_R_BROW_S = [107, 66, 105, 63, 70, 46]
_L_BROW_S = [336, 296, 334, 293, 300, 276]
_LIP_OUT  = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
_LIP_IN   = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Forehead ROI landmarks for rPPG
_FOREHEAD_LM = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152]  # outer boundary, simplified

# ═══════════════════════════════════════════════════════════════════════════════
# COLOURS (BGR)
# ═══════════════════════════════════════════════════════════════════════════════
_TEAL  = (0, 210, 180);  _RED   = (30, 40, 220);  _AMBER = (0, 190, 240)
_WHITE = (230, 230, 230); _BLUE  = (210, 130, 30); _GREEN = (50, 200, 70)
_DARK  = (18, 18, 22);    _GREY  = (90, 90, 100);  _PINK  = (160, 80, 200)

_STATE_COL = {
    "FOCUSED":  (0, 200, 120), "CONSUMED":  (0, 230, 255),
    "WANDERING":(0, 180, 240), "LOST":      (30, 40, 220),
    "RESTING":  (150,150,160), "VIGILANT":  (0, 130, 255),
    "PRESENT":  (180, 180, 90), "DROWSY":   (60, 80, 200),
    "ANXIOUS":  (0, 120, 255),
}

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def _bandpass(sig, lo, hi, fps=FPS):
    nyq = fps / 2.0
    lo_ = max(lo / nyq, 1e-4); hi_ = min(hi / nyq, 0.999)
    if lo_ >= hi_ or len(sig) < 18:
        return np.asarray(sig, float)
    try:
        b, a = butter(2, [lo_, hi_], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return np.asarray(sig, float)

def _lowpass(sig, cut, fps=FPS):
    nyq = fps / 2.0
    c = min(cut / nyq, 0.99)
    if len(sig) < 10:
        return np.asarray(sig, float)
    try:
        b, a = butter(2, c, btype="low")
        return filtfilt(b, a, sig)
    except Exception:
        return np.asarray(sig, float)

def _blink_rate(ear_hist, fps=FPS, thr=0.21):
    if len(ear_hist) < fps:
        return 0.0
    blinks, in_b = 0, False
    for e in ear_hist:
        if e < thr and not in_b:
            blinks += 1; in_b = True
        elif e >= thr:
            in_b = False
    dur = len(ear_hist) / fps / 60.0
    return blinks / dur if dur > 0 else 0.0

def _fft_peak(sig, fps, lo_hz, hi_hz):
    """Return dominant frequency (Hz) in [lo_hz, hi_hz] via FFT magnitude peak."""
    n    = len(sig)
    if n < fps * 4:
        return 0.0
    win  = np.hanning(n)
    fft  = np.abs(np.fft.rfft((sig - np.mean(sig)) * win))
    freq = np.fft.rfftfreq(n, 1.0 / fps)
    mask = (freq >= lo_hz) & (freq <= hi_hz)
    if not mask.any():
        return 0.0
    return float(freq[mask][np.argmax(fft[mask])])

# ═══════════════════════════════════════════════════════════════════════════════
# HEAD POSE ESTIMATOR  — solvePnP → yaw/pitch/roll, stabilization offset
# ═══════════════════════════════════════════════════════════════════════════════
class HeadPoseEstimator:
    """
    Estimates head rotation (yaw/pitch/roll) each frame using solvePnP on 6
    canonical landmarks. Also computes a 2-D translational drift vector from a
    rolling mean of the bony-landmark centroid, used to subtract rigid head
    translation from all zone trajectories.
    """
    def __init__(self):
        self.yaw   = 0.0
        self.pitch = 0.0
        self.roll  = 0.0
        self.rvec  = None
        self.tvec  = None
        # Rolling centroid of rigid landmarks for translation stabilization
        self._centroid_buf = deque(maxlen=FPS * 3)   # 3s
        self.drift = np.zeros(2)   # pixels: current frame centroid – rolling mean
        self._cam_matrix = None    # filled on first frame

    def update(self, lm, fw, fh):
        if self._cam_matrix is None:
            self._cam_matrix = np.array(
                [[fw, 0, fw / 2], [0, fw, fh / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1))
        pts2d = np.array([[lm[i].x * fw, lm[i].y * fh] for i in _PNP_IDX], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            _FACE_3D, pts2d, self._cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            self.rvec = rvec; self.tvec = tvec
            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            self.pitch = float(np.degrees(np.arctan2(-R[2,0], sy)))
            self.yaw   = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
            self.roll  = float(np.degrees(np.arctan2(R[2,1], R[2,2])))

        # Translation stabilization
        rigid_pts = np.array([(lm[i].x * fw, lm[i].y * fh) for i in _RIGID])
        centroid  = rigid_pts.mean(axis=0)
        self._centroid_buf.append(centroid)
        mean_c    = np.mean(self._centroid_buf, axis=0)
        self.drift = centroid - mean_c   # pixels the head has drifted this frame

    def stabilize(self, x_px, y_px):
        """Remove translational head drift from a pixel position."""
        return x_px - self.drift[0], y_px - self.drift[1]

# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE CALIBRATOR  — 30-second personal baseline, then rolling z-scores
# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveCalibrator:
    CAL_FRAMES = FPS * 30   # 30 seconds

    def __init__(self):
        self._raw: dict[str, list] = defaultdict(list)
        self._base: dict[str, float] = {}
        self._std:  dict[str, float] = {}
        self.calibrated = False
        self._fc = 0

    def add(self, key: str, value: float):
        self._fc += 1
        if not self.calibrated:
            self._raw[key].append(value)
            if self._fc >= self.CAL_FRAMES:
                self._finish()

    def _finish(self):
        for k, vals in self._raw.items():
            if len(vals) > 10:
                self._base[k] = float(np.mean(vals))
                self._std[k]  = max(float(np.std(vals)), 1e-5)
        self.calibrated = True

    def z(self, key: str, value: float) -> float:
        if key not in self._base:
            return 0.0
        return (value - self._base[key]) / self._std[key]

    def threshold(self, key: str, z: float = 2.5) -> float:
        if key not in self._base:
            return float("inf")
        return self._base[key] + z * self._std[key]

    @property
    def progress(self) -> float:
        return min(1.0, self._fc / self.CAL_FRAMES)

# ═══════════════════════════════════════════════════════════════════════════════
# rPPG HEART-RATE ESTIMATOR  (green channel forehead ROI + bandpass FFT)
# ═══════════════════════════════════════════════════════════════════════════════
class RPPGAnalyzer:
    """
    Extracts remote photoplethysmography signal from the forehead ROI.
    Uses the green channel (highest absorption contrast for hemoglobin).
    FFT peak in 0.75-3.0 Hz band → heart rate.
    ~10s of data needed for reliable rate.
    """
    BUF_SEC = 12

    def __init__(self):
        self._buf  = deque(maxlen=FPS * self.BUF_SEC)
        self.wave  = deque(maxlen=FPS * 6)   # 6s display
        self.bpm   = 0.0
        self.hrv   = 0.0    # std of inter-beat intervals (proxy)
        self._fc   = 0

    def update(self, frame, lm, fw, fh):
        self._fc += 1
        # Forehead ROI: between brows and hairline, center of face
        # Use landmarks: 10 (top-center), 338/297 (sides), 9 (bottom)
        try:
            # Simple forehead box: above brow midpoint, center third
            brow_y  = int(min(lm[70].y, lm[300].y) * fh)  # outer brow tops
            fore_y0 = max(0,  int(lm[10].y * fh))           # ~hairline
            fore_y1 = max(fore_y0 + 5, brow_y - 5)
            cx      = int(lm[168].x * fw)                   # nose bridge x
            hw      = max(20, int(fw * 0.12))
            fore_x0 = max(0, cx - hw); fore_x1 = min(fw, cx + hw)
            if fore_y1 > fore_y0 and fore_x1 > fore_x0:
                roi    = frame[fore_y0:fore_y1, fore_x0:fore_x1]
                g_mean = float(np.mean(roi[:, :, 1]))   # green channel
                self._buf.append(g_mean)
                self.wave.append(g_mean)
        except Exception:
            pass

        if self._fc % FPS == 0 and len(self._buf) >= FPS * 6:
            sig  = np.array(self._buf, dtype=float)
            # Detrend then bandpass
            detrended = sig - _lowpass(sig, 0.5)
            bp   = _bandpass(detrended, 0.75, 3.0)
            peak_hz = _fft_peak(bp, FPS, 0.75, 3.0)
            if peak_hz > 0:
                self.bpm = round(peak_hz * 60, 1)
            # HRV proxy: std of 1-frame differences in bandpassed signal
            self.hrv = round(float(np.std(np.diff(bp[-FPS*3:]))), 4)

# ═══════════════════════════════════════════════════════════════════════════════
# BREATHING ANALYZER  — PCA on multi-ROI flow + FFT rate
# ═══════════════════════════════════════════════════════════════════════════════
class BreathingAnalyzer:
    """
    Three ROIs: left-shoulder, right-shoulder, chest-center.
    Dense Farneback optical flow → PCA → dominant respiratory component per ROI.
    Signal is the projection of all flow vectors onto the first PC (respiratory axis).
    Rate via FFT peak, depth via signal amplitude.
    """
    ROI_W = 100   # downscaled width for each ROI

    def __init__(self):
        self._bufs  = [deque(maxlen=FPS * 15) for _ in range(3)]
        self.wave   = deque(maxlen=FPS * 8)
        self._prevs = [None, None, None]
        self.rate   = 0.0
        self.depth  = "—"
        self._fc    = 0
        self._combined = deque(maxlen=FPS * 15)

    def _get_rois(self, frame, chin_y, face_h, fw, fh):
        y0  = min(int(chin_y + 0.05 * face_h), fh - 10)
        y1  = min(int(chin_y + 1.55 * face_h), fh)
        if y1 - y0 < 20:
            return None
        third = fw // 3
        rois  = [
            frame[y0:y1, 0:third],         # left shoulder
            frame[y0:y1, 2*third:fw],      # right shoulder
            frame[y0:y1, third:2*third],   # chest center
        ]
        return rois

    def _flow_pca_signal(self, gray, prev, idx):
        """Dense flow → PCA → scalar respiratory signal."""
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        # Reshape to (N, 2) and run 1-component PCA
        vecs = flow.reshape(-1, 2)
        # Subtract mean (remove camera drift component)
        vecs -= vecs.mean(axis=0)
        if len(vecs) < 10:
            return 0.0
        # SVD: first right-singular vector = dominant motion direction
        _, _, Vt = np.linalg.svd(vecs, full_matrices=False)
        pc1 = Vt[0]   # dominant direction (1-D unit vector)
        # Project all flow vectors onto PC1 → scalar signal
        proj = float(np.mean(vecs @ pc1))
        return proj

    def update(self, frame, chin_y, face_h, fw, fh):
        self._fc += 1
        rois = self._get_rois(frame, chin_y, face_h, fw, fh)
        if rois is None:
            return

        signals = []
        for i, roi in enumerate(rois):
            th, tw = roi.shape[:2]
            if tw < 10 or th < 10:
                continue
            scale = self.ROI_W / max(tw, 1)
            roi_s = cv2.resize(roi, (self.ROI_W, max(8, int(th * scale))))
            gray  = cv2.cvtColor(roi_s, cv2.COLOR_BGR2GRAY)
            if self._prevs[i] is not None and gray.shape == self._prevs[i].shape:
                if self._fc % 2 == 0:
                    sig = self._flow_pca_signal(gray, self._prevs[i], i)
                    self._bufs[i].append(sig)
                    signals.append(sig)
            self._prevs[i] = gray.copy()

        if signals:
            combined = float(np.mean(signals))
            self._combined.append(combined)
            self.wave.append(combined)

        if self._fc % FPS == 0 and len(self._combined) >= FPS * 6:
            arr      = np.array(self._combined)
            filtered = _bandpass(arr, 0.10, 0.60)
            peak_hz  = _fft_peak(filtered, FPS, 0.10, 0.55)
            if peak_hz > 0:
                self.rate = round(peak_hz * 60, 1)
            amp = float(np.std(filtered))
            if   amp < 0.015: self.depth = "SHALLOW"
            elif amp < 0.045: self.depth = "NORMAL"
            else:             self.depth = "DEEP"

    def roi_rect(self, chin_y, face_h, fw, fh):
        y0 = min(int(chin_y + 0.05 * face_h), fh - 10)
        y1 = min(int(chin_y + 1.55 * face_h), fh)
        return 0, y0, fw, y1

# ═══════════════════════════════════════════════════════════════════════════════
# NECK / SWALLOW DETECTOR  — LK sparse tracking + state machine
# ═══════════════════════════════════════════════════════════════════════════════
class NeckSwallowDetector:
    """
    Tracks ~24 sparse points in the neck ROI using Lucas-Kanade optical flow.
    Swallowing signature (laryngeal elevation): upward motion → brief hold →
    downward recovery, total 0.3–0.9 seconds.
    State machine: IDLE → RISING → PEAK → FALLING → RECOVERY
    """
    IDLE = 0; RISING = 1; PEAK = 2; FALLING = 3; RECOVERY = 4

    N_PTS    = 24       # tracked points
    RISE_THR = 0.5      # px/frame threshold for upward motion detection
    FALL_THR = 0.3
    MIN_RISE = FPS // 6  # ≥5 frames of rise to confirm swallow
    MAX_PEAK = FPS // 2  # ≤15 frames at peak before timeout

    def __init__(self):
        self._pts      = None    # current tracked points (N, 1, 2) float32
        self._prev_g   = None
        self._state    = self.IDLE
        self._sc       = 0       # frames in current state
        self._vy_buf   = deque(maxlen=FPS * 3)
        self.swallow_now   = False
        self.swallow_count = 0
        self._fc           = 0
        self._reinit_cd    = 0   # countdown to re-seed points

    def _seed_pts(self, gray, x0, y0, x1, y1):
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return
        corners = cv2.goodFeaturesToTrack(
            roi, maxCorners=self.N_PTS, qualityLevel=0.01,
            minDistance=6, blockSize=5)
        if corners is not None:
            # Offset back to full-frame coordinates
            corners[:, 0, 0] += x0; corners[:, 0, 1] += y0
            self._pts = corners.astype(np.float32)
        else:
            self._pts = None

    def update(self, frame, chin_y, face_h, fw, fh):
        self._fc += 1
        y0 = min(int(chin_y + 0.02 * face_h), fh - 10)
        y1 = min(int(chin_y + 0.60 * face_h), fh)
        x0 = max(0, fw // 4)
        x1 = min(fw, 3 * fw // 4)
        if y1 - y0 < 12 or x1 - x0 < 12:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Re-seed tracked points every ~3 seconds or on loss
        self._reinit_cd -= 1
        if self._reinit_cd <= 0 or self._pts is None or len(self._pts) < 4:
            self._seed_pts(gray, x0, y0, x1, y1)
            self._reinit_cd = FPS * 3

        if self._pts is not None and self._prev_g is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_g, gray, self._pts, None,
                winSize=(13, 13), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            good = status.ravel() == 1
            if good.sum() >= 4:
                old_g = self._pts[good]
                new_g = new_pts[good]
                # Mean vertical velocity (negative = upward in image coords)
                vy = float(np.mean(new_g[:, 0, 1] - old_g[:, 0, 1]))
                self._vy_buf.append(vy)
                self._pts = new_pts[good]
                self._run_state_machine(vy)
            else:
                self._pts = None

        self._prev_g = gray.copy()

    def _run_state_machine(self, vy):
        """Negative vy = upward motion = laryngeal rise = swallow start."""
        self._sc += 1
        if self._state == self.IDLE:
            if vy < -self.RISE_THR:
                self._state = self.RISING; self._sc = 1
            self.swallow_now = False

        elif self._state == self.RISING:
            if vy < -self.RISE_THR:
                pass   # keep rising
            elif self._sc >= self.MIN_RISE:
                self._state = self.PEAK; self._sc = 1
            else:
                self._state = self.IDLE   # too brief, not a swallow

        elif self._state == self.PEAK:
            if abs(vy) < self.RISE_THR * 0.5:
                pass   # holding
            elif vy > self.FALL_THR:
                self._state = self.FALLING; self._sc = 1
            elif self._sc > self.MAX_PEAK:
                self._state = self.IDLE   # timeout

        elif self._state == self.FALLING:
            if vy > 0 or abs(vy) < 0.2:
                # Recovery complete → confirmed swallow
                self.swallow_count += 1
                self.swallow_now    = True
                self._state = self.RECOVERY; self._sc = 1
            elif self._sc > FPS:
                self._state = self.IDLE

        elif self._state == self.RECOVERY:
            self.swallow_now = (self._sc < FPS // 4)
            if self._sc >= FPS // 2:
                self._state = self.IDLE

    def roi_rect(self, chin_y, face_h, fw, fh):
        y0 = min(int(chin_y + 0.02 * face_h), fh - 10)
        y1 = min(int(chin_y + 0.60 * face_h), fh)
        return max(0, fw // 4), y0, min(fw, 3 * fw // 4), y1

# ═══════════════════════════════════════════════════════════════════════════════
# BROW ANALYZER  — full shape + AU state machine for brow flash
# ═══════════════════════════════════════════════════════════════════════════════
class BrowAnalyzer:
    IDLE = 0; ASCENDING = 1; PEAK_HOLD = 2; DESCENDING = 3

    def __init__(self):
        self._r_h  = deque(maxlen=FPS * 5)
        self._l_h  = deque(maxlen=FPS * 5)
        self._fs   = self.IDLE   # flash state
        self._fsc  = 0
        self.au1 = self.au2 = self.au4 = False
        self.asym      = 0.0
        self.brow_flash = False

    def update(self, lm, fw, fh, cal: AdaptiveCalibrator):
        def brow_h(brow, eye):
            return float(np.mean([lm[i].y for i in eye]) - np.mean([lm[i].y for i in brow]))

        r_h = brow_h(_R_BROW_S, [160, 158, 157, 173])
        l_h = brow_h(_L_BROW_S, [385, 387, 388, 398])
        self._r_h.append(r_h); self._l_h.append(l_h)

        cal.add("brow_r", r_h); cal.add("brow_l", l_h)

        # AU4: brow furrow — inner brow distance narrows
        inner_gap = abs(lm[107].x - lm[336].x) * fw
        cal.add("inner_gap", inner_gap)
        self.au4  = cal.calibrated and (cal.z("inner_gap", inner_gap) < -1.5)

        # AU1: inner brow raise — inner higher than outer (relative)
        r_rise = lm[107].y - lm[46].y     # positive = inner lower (neutral)
        l_rise = lm[336].y - lm[276].y
        self.au1 = (r_rise > 0.012) or (l_rise > 0.012)

        # AU2: outer brow raise
        self.au2 = (lm[46].y < lm[107].y - 0.010) or (lm[276].y < lm[336].y - 0.010)

        # Asymmetry
        self.asym = abs(r_h - l_h) / (max(abs(r_h), abs(l_h), 1e-5))

        # Brow-flash state machine: rapid raise-and-return (<0.6s total)
        self._fsc += 1
        mean_h = (r_h + l_h) / 2.0
        if self._fs == self.IDLE:
            self.brow_flash = False
            if cal.calibrated and cal.z("brow_r", r_h) > 2.0:
                self._fs = self.ASCENDING; self._fsc = 1
        elif self._fs == self.ASCENDING:
            if cal.calibrated and cal.z("brow_r", r_h) < 1.0:
                if self._fsc < FPS // 2:   # ascent was rapid
                    self._fs = self.DESCENDING; self._fsc = 1
                else:
                    self._fs = self.IDLE
        elif self._fs == self.DESCENDING:
            self.brow_flash = True
            if cal.calibrated and cal.z("brow_r", r_h) < 0.3:
                self._fs = self.IDLE
            elif self._fsc > FPS:
                self._fs = self.IDLE

# ═══════════════════════════════════════════════════════════════════════════════
# GAZE ANALYZER  — IPD-normalized saccades + PERCLOS + microsaccades + entropy
# ═══════════════════════════════════════════════════════════════════════════════
class GazeAnalyzer:
    """
    Saccade velocity normalized by IPD (resolution-invariant).
    Adds PERCLOS (drowsiness), microsaccade rate (cognitive load),
    gaze entropy (exploratory vs focused), eye divergence (near/far focus),
    and hysteresis to prevent state flicker.
    """
    IDLE = 0; SACCADE = 1; FIXATION = 2

    SAC_VEL_IPD   = 0.15    # velocity > 15% IPD/frame → macro saccade
    SAC_MIN_AMP   = 0.06    # minimum macro saccade amplitude (IPD units)
    MICRO_MAX_AMP = 0.03    # < 3% IPD during fixation = microsaccade
    MICRO_MIN_VEL = 0.02
    PERCLOS_EAR   = 0.21    # EAR threshold for "eye closed"
    MIN_DWELL     = 2.0     # seconds a new state must hold before committing

    def __init__(self):
        self._iris_h   = deque(maxlen=FPS * 10)
        self._sac_ts   = deque(maxlen=200)
        self._sac_amps = deque(maxlen=200)
        self._micro_ts = deque(maxlen=400)
        self._ear_cl   = deque(maxlen=FPS * 60)  # bool: True = closed
        self._pos_x    = deque(maxlen=FPS * 30)  # for entropy
        self._pos_y    = deque(maxlen=FPS * 30)
        self._gs       = self.IDLE
        self._gsc      = 0
        self._fix_t0   = None
        self._sac_start_pos = (0.0, 0.0)
        self.fixation_dur   = 0.0
        self.saccade_rate   = 0.0   # macro saccades/min
        self.saccade_amp    = 0.0
        self.microsaccade_rate = 0.0  # microsaccades/min
        self.perclos        = 0.0   # fraction of time eyes closed (0-1)
        self.gaze_entropy   = 0.0   # Shannon entropy of scan path (bits, 0-4.6)
        self.divergence     = 0.0   # >0 = divergent (far/dissociated), <0 = convergent (near)
        self.gaze_h = self.gaze_v = 0.0
        self.gaze_dir       = "CENTER"
        self.state          = "RESTING"
        self.concentration  = 50
        self._fc            = 0
        self._ipd           = 0.1
        self._pending_state = "RESTING"
        self._pending_since = 0.0

    def update(self, lm, fw, fh, blink_rate, flat_affect):
        self._fc += 1
        if len(lm) <= 478:
            return
        now = self._fc / FPS

        r_ir = lm[_R_IRIS]; l_ir = lm[_L_IRIS]

        # ── IPD (smoothed) ────────────────────────────────────────────────────
        ipd = abs(r_ir.x - l_ir.x) + 1e-4
        self._ipd = ipd * 0.9 + self._ipd * 0.1

        # ── EAR → PERCLOS ────────────────────────────────────────────────────
        r_ear = sum(np.linalg.norm([(lm[a].x-lm[b].x)*fw,(lm[a].y-lm[b].y)*fh])
                    for a,b in [(160,144),(158,153)]) / (
                    2*max(np.linalg.norm([(lm[33].x-lm[133].x)*fw,(lm[33].y-lm[133].y)*fh]),1e-4))
        l_ear = sum(np.linalg.norm([(lm[a].x-lm[b].x)*fw,(lm[a].y-lm[b].y)*fh])
                    for a,b in [(385,380),(387,373)]) / (
                    2*max(np.linalg.norm([(lm[362].x-lm[263].x)*fw,(lm[362].y-lm[263].y)*fh]),1e-4))
        avg_ear = (r_ear + l_ear) / 2.0
        self._ear_cl.append(avg_ear < self.PERCLOS_EAR)
        if self._fc % FPS == 0 and len(self._ear_cl) >= FPS * 10:
            self.perclos = float(np.mean(list(self._ear_cl)[-FPS * 60:]))

        # ── Gaze offset (iris vs eye-corner midpoint) ─────────────────────
        r_mx = (lm[_R_EYE_OUT].x + lm[_R_EYE_IN].x) / 2
        r_my = (lm[_R_EYE_OUT].y + lm[_R_EYE_IN].y) / 2
        l_mx = (lm[_L_EYE_OUT].x + lm[_L_EYE_IN].x) / 2
        l_my = (lm[_L_EYE_OUT].y + lm[_L_EYE_IN].y) / 2
        off_h = ((r_ir.x - r_mx) + (l_ir.x - l_mx)) / 2.0
        off_v = ((r_ir.y - r_my) + (l_ir.y - l_my)) / 2.0
        self.gaze_h = float(off_h / self._ipd)
        self.gaze_v = float(off_v / self._ipd)

        # ── Eye divergence: positive = irises drifting outward (far focus / dissociation) ──
        r_div = (r_ir.x - r_mx) / (abs(lm[_R_EYE_OUT].x - lm[_R_EYE_IN].x) + 1e-4)
        l_div = (l_ir.x - l_mx) / (abs(lm[_L_EYE_OUT].x - lm[_L_EYE_IN].x) + 1e-4)
        self.divergence = float(r_div - l_div)   # smooth later if needed

        th_h, th_v = 0.32, 0.28
        if   self.gaze_h < -th_h and self.gaze_v < -th_v: self.gaze_dir = "UP-R"
        elif self.gaze_h >  th_h and self.gaze_v < -th_v: self.gaze_dir = "UP-L"
        elif self.gaze_h < -th_h and self.gaze_v >  th_v: self.gaze_dir = "DOWN-R"
        elif self.gaze_h >  th_h and self.gaze_v >  th_v: self.gaze_dir = "DOWN-L"
        elif self.gaze_h < -th_h: self.gaze_dir = "RIGHT"
        elif self.gaze_h >  th_h: self.gaze_dir = "LEFT"
        elif self.gaze_v < -th_v: self.gaze_dir = "UP"
        elif self.gaze_v >  th_v: self.gaze_dir = "DOWN"
        else:                      self.gaze_dir = "CENTER"

        # ── Iris position history (IPD-normalized) ────────────────────────
        ix = ((r_ir.x + l_ir.x) / 2.0) / self._ipd
        iy = ((r_ir.y + l_ir.y) / 2.0) / self._ipd
        self._iris_h.append((ix, iy, now))
        self._pos_x.append(self.gaze_h)
        self._pos_y.append(self.gaze_v)

        # ── Saccade + microsaccade state machine ──────────────────────────
        self._gsc += 1
        if len(self._iris_h) >= 3:
            px, py, _ = self._iris_h[-3]
            vel = np.hypot(ix - px, iy - py)

            if self._gs == self.IDLE or self._gs == self.FIXATION:
                if vel > self.SAC_VEL_IPD:
                    self._gs = self.SACCADE
                    self._sac_start_pos = (ix, iy)
                    self._gsc = 1
                    if self._fix_t0 is not None:
                        self.fixation_dur = now - self._fix_t0
                elif self._gs == self.IDLE:
                    self._gs = self.FIXATION; self._fix_t0 = now
                elif self._gs == self.FIXATION:
                    # microsaccade: small quick drift during fixation
                    if self.MICRO_MIN_VEL < vel < self.SAC_VEL_IPD * 0.6:
                        amp = np.hypot(ix - px, iy - py)
                        if amp < self.MICRO_MAX_AMP:
                            self._micro_ts.append(now)

            elif self._gs == self.SACCADE:
                if vel < self.SAC_VEL_IPD * 0.3:
                    amp = np.hypot(ix - self._sac_start_pos[0], iy - self._sac_start_pos[1])
                    if amp > self.SAC_MIN_AMP:
                        self._sac_ts.append(now)
                        self._sac_amps.append(float(amp))
                    self._gs = self.FIXATION; self._fix_t0 = now; self._gsc = 1
                elif self._gsc > FPS // 2:
                    self._gs = self.FIXATION; self._fix_t0 = now; self._gsc = 1

        # ── Per-second stats + gaze entropy ──────────────────────────────
        if self._fc % FPS == 0:
            win = 60.0; t_cut = now - win
            recent_ts  = [t for t in self._sac_ts  if t > t_cut]
            recent_amp = [a for t, a in zip(self._sac_ts, self._sac_amps) if t > t_cut]
            micro_ts   = [t for t in self._micro_ts if t > t_cut]
            self.saccade_rate     = len(recent_ts)
            self.saccade_amp      = float(np.mean(recent_amp)) if recent_amp else 0.0
            self.microsaccade_rate = len(micro_ts)

            # Gaze entropy: 5×5 histogram over last 30 s of scan path
            if len(self._pos_x) >= FPS * 5:
                xs = np.array(self._pos_x); ys = np.array(self._pos_y)
                H, _, _ = np.histogram2d(xs, ys, bins=5,
                                         range=[[-1.2, 1.2], [-1.0, 1.0]])
                H = H.flatten() + 1e-10
                H /= H.sum()
                self.gaze_entropy = float(-np.sum(H * np.log2(H)))

            self._classify(blink_rate, flat_affect)

    def _classify(self, blink_rate, flat_affect):
        new = self._compute_state(blink_rate, flat_affect)
        now = self._fc / FPS
        if new != self._pending_state:
            self._pending_state = new
            self._pending_since = now
        elif now - self._pending_since >= self.MIN_DWELL:
            if new != self.state:
                self.state = new
                conc_map = {
                    "CONSUMED": 95, "FOCUSED": 88, "PRESENT": 65,
                    "RESTING": 55, "WANDERING": 42, "VIGILANT": 35,
                    "ANXIOUS": 30, "LOST": 18, "DROWSY": 15,
                }
                self.concentration = conc_map.get(new, 55)

    def _compute_state(self, blink_rate, flat_affect):
        sr = self.saccade_rate; fd = self.fixation_dur
        br = blink_rate; fa = flat_affect; amp = self.saccade_amp
        pc = self.perclos; msr = self.microsaccade_rate; ge = self.gaze_entropy
        dv = self.divergence
        # DROWSY: high PERCLOS, or eyes very slow + blinking slowly (microsleep)
        if pc > 0.15 or (fa > 0.72 and sr < 10 and br < 6):
            return "DROWSY"
        # ANXIOUS: rapid blinking + high microsaccade rate + low flat affect
        if br > 26 and msr > 100 and fa < 0.38:
            return "ANXIOUS"
        # CONSUMED: very still gaze, low blink, engaged (watching something)
        if sr < 5 and br < 8 and fa < 0.50:
            return "CONSUMED"
        # FOCUSED: moderate fixation, long dwell
        if sr < 15 and fd > 1.8 and br < 16:
            return "FOCUSED"
        # LOST: chaotic high-amp saccades + high entropy
        if sr > 80 and amp > 0.18 and ge > 3.8:
            return "LOST"
        # VIGILANT: rapid scanning + elevated blink
        if sr > 55 and br > 22:
            return "VIGILANT"
        # RESTING: flat affect, normal blink, low saccades
        if fa > 0.60 and 10 <= br <= 22 and sr < 35:
            return "RESTING"
        # WANDERING: moderate saccades, small amplitude, high entropy
        if sr > 30 and amp < 0.14 and ge > 2.8:
            return "WANDERING"
        return "PRESENT"

# ═══════════════════════════════════════════════════════════════════════════════
# FACIAL MUSCLE LOGGER  — deformation vectors → named muscles → clinical log
# ═══════════════════════════════════════════════════════════════════════════════
class FacialMuscleLogger:
    """
    Translates per-landmark deformation residuals into named muscle activations
    with FACS Action Unit codes and one-line clinical interpretations.

    Method: for each muscle, project the residual displacement vectors of its
    attached landmarks onto the muscle's anatomical action direction (dot product).
    Positive dot product = muscle is contracting in its anatomical direction.

    Events are classified as:
    • brief  (0.3–1.5 s): microexpression, twitch, punctuation gesture
    • sustained (>2 s): emotional state, pain, cognitive load

    Composite patterns (Duchenne smile, fear cluster, asymmetric expression)
    are detected from co-activation of multiple muscles.
    """

    # Each entry: landmarks, action_direction(image: x→right y→down),
    #             name, AU code, brief label, sustained label, clinical note
    # Action direction = the direction muscle PULLS attached skin landmarks
    # (positive dot product with residual = muscle is active in its anatomical direction)
    MUSCLES = {
        # ── Forehead ───────────────────────────────────────────────────────────
        "frontalis_r": dict(
            lm=[107, 66, 46],         dir=(0.0, -1.0),
            name="Frontalis (R)",     au="AU2",
            brief="Right outer brow raise",
            sustained="Sustained right brow elevation",
            clinical="Surprise · fear · concern — unilateral = skepticism"),
        "frontalis_l": dict(
            lm=[336, 296, 276],       dir=(0.0, -1.0),
            name="Frontalis (L)",     au="AU1",
            brief="Left outer brow raise",
            sustained="Sustained left brow elevation",
            clinical="Surprise · fear · concern — unilateral = skepticism"),
        "frontalis_inner_r": dict(
            lm=[107, 55, 65],         dir=(-0.15, -1.0),
            name="Frontalis medial (R)", au="AU1",
            brief="Right inner brow raise",
            sustained="Medial brow pull sustained (R)",
            clinical="Inner worry line — distress, plea, empathic concern"),
        "frontalis_inner_l": dict(
            lm=[336, 285, 295],       dir=(0.15, -1.0),
            name="Frontalis medial (L)", au="AU1",
            brief="Left inner brow raise",
            sustained="Medial brow pull sustained (L)",
            clinical="Inner worry line — distress, plea, empathic concern"),
        # ── Brow depression / furrow ────────────────────────────────────────────
        "corrugator": dict(
            lm=[107, 336],            dir=None,   # special: convergence metric
            name="Corrugator supercilii", au="AU4",
            brief="Brow furrow (transient)",
            sustained="Sustained brow furrow",
            clinical="Distress · pain · cognitive load · anger — strongest negative affect marker"),
        "procerus": dict(
            lm=[6, 168, 197],         dir=(0.0, 0.8),
            name="Procerus",          au="AU9",
            brief="Nose-bridge wrinkle",
            sustained="Sustained nose bridge crease",
            clinical="Disgust · distress · pain — co-occurs with corrugator in grief"),
        # ── Eyelids ─────────────────────────────────────────────────────────────
        "lev_palpebrae_r": dict(
            lm=[159, 160, 158],       dir=(0.0, -1.0),
            name="Levator palpebrae (R)", au="AU5R",
            brief="Upper lid raise (R) — wide eye",
            sustained="Wide-eye state (R) sustained",
            clinical="Fear · hypervigilance · surprise · startle response"),
        "lev_palpebrae_l": dict(
            lm=[386, 385, 387],       dir=(0.0, -1.0),
            name="Levator palpebrae (L)", au="AU5L",
            brief="Upper lid raise (L) — wide eye",
            sustained="Wide-eye state (L) sustained",
            clinical="Fear · hypervigilance · surprise · startle response"),
        "orbicularis_oculi_r": dict(
            lm=[159, 145, 133, 33],   dir=None,   # special: EAR compression
            name="Orbicularis oculi (R)", au="AU6R",
            brief="Cheek raise / lid squeeze (R)",
            sustained="Duchenne lid squeeze sustained (R)",
            clinical="Genuine positive affect marker · pain wince · disgust"),
        "orbicularis_oculi_l": dict(
            lm=[386, 374, 263, 362],  dir=None,
            name="Orbicularis oculi (L)", au="AU6L",
            brief="Cheek raise / lid squeeze (L)",
            sustained="Duchenne lid squeeze sustained (L)",
            clinical="Genuine positive affect marker · pain wince · disgust"),
        # ── Cheek / smile ────────────────────────────────────────────────────────
        "zygomaticus_r": dict(
            lm=[61, 185, 40],         dir=(-0.6, -0.8),
            name="Zygomaticus major (R)", au="AU12R",
            brief="Lip corner pull up-lateral (R)",
            sustained="Smile maintained (R)",
            clinical="Positive affect — verify AU6 co-activation for Duchenne"),
        "zygomaticus_l": dict(
            lm=[291, 409, 270],       dir=(0.6, -0.8),
            name="Zygomaticus major (L)", au="AU12L",
            brief="Lip corner pull up-lateral (L)",
            sustained="Smile maintained (L)",
            clinical="Positive affect — verify AU6 co-activation for Duchenne"),
        # ── Lip depression / grief ───────────────────────────────────────────────
        "depressor_ao_r": dict(
            lm=[172, 61, 57],         dir=(0.3, 1.0),
            name="Depressor anguli oris (R)", au="AU15R",
            brief="Lip corner depress (R)",
            sustained="Mouth downturn sustained (R)",
            clinical="Sadness · grief · disappointment · contempt"),
        "depressor_ao_l": dict(
            lm=[397, 291, 287],       dir=(-0.3, 1.0),
            name="Depressor anguli oris (L)", au="AU15L",
            brief="Lip corner depress (L)",
            sustained="Mouth downturn sustained (L)",
            clinical="Sadness · grief · disappointment · contempt"),
        # ── Lip tension / suppression ────────────────────────────────────────────
        "orbicularis_oris": dict(
            lm=[13, 14, 0, 17, 61, 291], dir=None,  # special: inward compression
            name="Orbicularis oris",  au="AU18/20",
            brief="Lip compression / purse",
            sustained="Sustained lip suppression",
            clinical="Emotional suppression · held affect · tension · deliberate control"),
        "risorius": dict(
            lm=[61, 291],             dir=None,    # special: lateral stretch only
            name="Risorius",          au="AU20",
            brief="Horizontal lip stretch (fear grin)",
            sustained="Sustained fear-grin",
            clinical="Fear · nervous tension — horizontal not upward pull"),
        # ── Chin ────────────────────────────────────────────────────────────────
        "mentalis": dict(
            lm=[152, 175, 199],       dir=(0.0, -0.7),
            name="Mentalis",          au="AU17",
            brief="Chin raise / dimple",
            sustained="Sustained chin tension",
            clinical="Distress · about-to-cry · emotional suppression — pre-cry marker"),
        # ── Nose ────────────────────────────────────────────────────────────────
        "nasalis": dict(
            lm=[129, 358, 98, 327],   dir=(0.0, -0.5),
            name="Nasalis",           au="AU38",
            brief="Nostril flare",
            sustained="Sustained nostril change",
            clinical="Disgust · distress · autonomic arousal · respiratory effort"),
    }

    BRIEF_THR   = 0.4    # seconds — below = wait; 0.4–1.8s = brief
    SUSTAIN_THR = 2.0    # seconds — above = sustained state
    ACT_THR     = 0.32   # activation z-score threshold to consider a muscle active
    LOG_MAX     = 10     # lines kept in visible log

    def __init__(self):
        self.activations: dict[str, float] = {k: 0.0 for k in self.MUSCLES}
        self._active_since: dict[str, float | None] = {k: None for k in self.MUSCLES}
        self._last_logged:  dict[str, float]         = {}
        self._log: list[dict] = []
        self._fc  = 0

    def update(self, deform: "FaceMeshDeformationAnalyzer", lm, fw, fh,
               cal: "AdaptiveCalibrator"):
        if deform._ref is None or not cal.calibrated:
            return
        self._fc += 1
        now   = self._fc / FPS
        res   = deform.residuals     # (468, 2) — muscle-only deformation vectors
        mags  = deform.magnitudes
        scale = max(deform.scale, 0.8)

        for key, m in self.MUSCLES.items():
            idxs  = [i for i in m["lm"] if i < 468]
            if not idxs:
                continue

            d = m["dir"]
            if d is None:
                # Special metrics ─────────────────────────────────────────────
                if key == "corrugator":
                    # Convergence: right inner brow moves left (+x), left moves right (-x)
                    r_res = res[107]; l_res = res[336]
                    convergence = r_res[0] - l_res[0]          # >0 = converging
                    lowering    = (r_res[1] + l_res[1]) / 2.0  # >0 = downward
                    activation  = (convergence * 0.55 + lowering * 0.45) / scale
                elif key in ("orbicularis_oculi_r", "orbicularis_oculi_l"):
                    # Lid squeeze: magnitude of eye-region residuals inward/downward
                    activation = float(np.mean(mags[idxs])) / scale
                elif key == "orbicularis_oris":
                    # Lip inward compression: lip corners move medially
                    r_res = res[61]; l_res = res[291]
                    # Right corner moves right (+x) and left corner moves left (-x) = compress
                    compression = r_res[0] - l_res[0]
                    activation  = compression / scale
                elif key == "risorius":
                    # Pure lateral stretch: corners move apart horizontally
                    r_res = res[61]; l_res = res[291]
                    stretch    = -(r_res[0] - l_res[0])   # negative compression = stretch
                    # Only count if vertical component is small (not a smile)
                    r_vert = abs(r_res[1]); l_vert = abs(l_res[1])
                    if r_vert + l_vert < abs(stretch) * 0.5:
                        activation = stretch / scale
                    else:
                        activation = 0.0
                else:
                    activation = float(np.mean(mags[idxs])) / scale
            else:
                # Dot product with anatomical action direction ─────────────────
                d_arr = np.array(d, dtype=float)
                norm  = np.linalg.norm(d_arr)
                if norm < 1e-4:
                    activation = 0.0
                else:
                    d_arr /= norm
                    dots = [float(np.dot(res[i], d_arr)) for i in idxs]
                    activation = float(np.mean(dots)) / scale

            activation = float(np.clip(activation, -0.5, 3.0))
            self.activations[key] = activation

            # ── Duration tracking + event emission ───────────────────────────
            if activation > self.ACT_THR:
                if self._active_since[key] is None:
                    self._active_since[key] = now
                dur = now - self._active_since[key]
                last = self._last_logged.get(key, -999.0)
                if self.BRIEF_THR <= dur < self.SUSTAIN_THR and (now - last) > 4.0:
                    self._emit(key, m, "brief",    activation, dur, now)
                elif dur >= self.SUSTAIN_THR and (now - last) > self.SUSTAIN_THR:
                    self._emit(key, m, "sustained", activation, dur, now)
            else:
                self._active_since[key] = None

        self._composites(now)

    def _emit(self, key, m, kind, act, dur, now):
        intensity = "strong" if act > 0.70 else ("moderate" if act > 0.45 else "subtle")
        self._log.insert(0, {
            "t": now, "ts": time.strftime("%H:%M:%S"),
            "muscle": m["name"], "au": m["au"],
            "msg": m["sustained"] if kind == "sustained" else m["brief"],
            "clinical": m["clinical"],
            "intensity": intensity, "kind": kind, "act": round(act, 2),
        })
        self._log = self._log[:self.LOG_MAX]
        self._last_logged[key] = now

    def _composites(self, now):
        a = self.activations
        def _since_last(tag, cooldown):
            return (now - self._last_logged.get(tag, -999.0)) > cooldown

        # AU6 + AU12 = Duchenne (genuine smile)
        au6  = (a.get("orbicularis_oculi_r", 0) + a.get("orbicularis_oculi_l", 0)) / 2
        au12 = (a.get("zygomaticus_r",       0) + a.get("zygomaticus_l",       0)) / 2
        if au6 > 0.38 and au12 > 0.40 and _since_last("_duchenne", 4.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Zygomaticus + Orbicularis oculi",
                "au": "AU6+12", "msg": "Duchenne smile",
                "clinical": "GENUINE positive affect — highest clinical confidence",
                "intensity": "strong", "kind": "composite", "act": round((au6+au12)/2, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_duchenne"] = now

        # AU1+2+5 = fear cluster
        fron = (a.get("frontalis_r", 0) + a.get("frontalis_l", 0)) / 2
        lev  = (a.get("lev_palpebrae_r", 0) + a.get("lev_palpebrae_l", 0)) / 2
        if fron > 0.38 and lev > 0.35 and _since_last("_fear", 5.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Frontalis + Levator palpebrae",
                "au": "AU1+2+5", "msg": "Fear / alarm expression cluster",
                "clinical": "Autonomic fear · elevated arousal · hypervigilance",
                "intensity": "strong", "kind": "composite", "act": round((fron+lev)/2, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_fear"] = now

        # Corrugator + depressor anguli oris = sadness/grief pattern
        corr  = a.get("corrugator", 0)
        dep   = (a.get("depressor_ao_r", 0) + a.get("depressor_ao_l", 0)) / 2
        if corr > 0.40 and dep > 0.35 and _since_last("_grief", 5.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Corrugator + Depressor anguli oris",
                "au": "AU4+15", "msg": "Grief / sadness expression cluster",
                "clinical": "Sadness with tension — depression / grief presentation",
                "intensity": "strong", "kind": "composite", "act": round((corr+dep)/2, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_grief"] = now

        # Mentalis + orbicularis oris = emotional suppression
        ment = a.get("mentalis", 0)
        oris = a.get("orbicularis_oris", 0)
        if ment > 0.38 and oris > 0.35 and _since_last("_suppress", 5.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Mentalis + Orbicularis oris",
                "au": "AU17+18", "msg": "Emotional suppression cluster",
                "clinical": "Active affect suppression — patient holding back emotional response",
                "intensity": "moderate", "kind": "composite", "act": round((ment+oris)/2, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_suppress"] = now

        # ── ANGER / RAGE: AU4 (corrugator) + AU9 (procerus) + AU38 (nasalis) ──
        proc  = a.get("procerus",  0)
        nasal = a.get("nasalis",   0)
        corr2 = a.get("corrugator", 0)
        if corr2 > 0.42 and proc > 0.35 and nasal > 0.30 and _since_last("_anger", 5.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Corrugator + Procerus + Nasalis",
                "au": "AU4+9+38", "msg": "Anger / rage expression cluster",
                "clinical": "Active threat response — aggression, hostile arousal, defensive anger",
                "intensity": "strong", "kind": "composite", "act": round((corr2+proc+nasal)/3, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_anger"] = now

        # ── DISGUST: AU9 (nose wrinkle) + bilateral AU15 (lip corner down) ──
        dep2  = (a.get("depressor_ao_r", 0) + a.get("depressor_ao_l", 0)) / 2
        proc2 = a.get("procerus", 0)
        if proc2 > 0.32 and dep2 > 0.30 and corr < 0.25 and _since_last("_disgust", 5.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Procerus + Depressor anguli oris",
                "au": "AU9+15", "msg": "Disgust expression cluster",
                "clinical": "Aversion · visceral rejection — disgust, moral revulsion, nausea",
                "intensity": "moderate", "kind": "composite", "act": round((proc2+dep2)/2, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_disgust"] = now

        # ── CONTEMPT: unilateral AU12 (one lip corner raised, other neutral) ──
        zy_r  = a.get("zygomaticus_r", 0)
        zy_l  = a.get("zygomaticus_l", 0)
        zy_peak = max(zy_r, zy_l, 0.01)
        zy_asym = abs(zy_r - zy_l) / zy_peak
        if zy_asym > 0.60 and zy_peak > 0.38 and _since_last("_contempt", 6.0):
            side = "right" if zy_r > zy_l else "left"
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": f"Zygomaticus ({side} only)",
                "au": "AU12 unilateral", "msg": f"Contempt — {side} lip corner only",
                "clinical": "Social dominance signal · disdain · contempt (Ekman unilateral AU12)",
                "intensity": "moderate", "kind": "composite", "act": round(zy_asym, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_contempt"] = now

        # ── PRE-CRY: AU1 (inner brow raise) + AU4 + AU15 + AU17 sustained ──
        fron_in = (a.get("frontalis_inner_r", 0) + a.get("frontalis_inner_l", 0)) / 2
        ment2   = a.get("mentalis", 0)
        dep3    = (a.get("depressor_ao_r", 0) + a.get("depressor_ao_l", 0)) / 2
        corr3   = a.get("corrugator", 0)
        cry_sig = (fron_in + corr3 + dep3 + ment2) / 4
        if fron_in > 0.32 and corr3 > 0.30 and dep3 > 0.28 and ment2 > 0.28 and _since_last("_cry", 6.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Frontalis (inner) + Corrugator + Depressor AO + Mentalis",
                "au": "AU1+4+15+17", "msg": "Pre-cry / grief expression cluster",
                "clinical": "Imminent crying — profound sadness, bereavement, emotional pain peak",
                "intensity": "strong", "kind": "composite", "act": round(cry_sig, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_cry"] = now

        # ── ANXIETY: AU4 (corrugator) + AU20 (risorius fear grimace) ─────
        ris    = a.get("risorius", 0)
        corr4  = a.get("corrugator", 0)
        if corr4 > 0.35 and ris > 0.30 and fron_in < 0.25 and _since_last("_anxiety", 5.0):
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": "Corrugator + Risorius",
                "au": "AU4+20", "msg": "Anxiety / tension grimace",
                "clinical": "Anticipatory anxiety · social threat · chronic tension — not acute fear",
                "intensity": "moderate", "kind": "composite", "act": round((corr4+ris)/2, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_anxiety"] = now

        # ── Facial asymmetry (neurological / leakage flag) ────────────────
        r_expr = a.get("zygomaticus_r", 0) + a.get("frontalis_r", 0) + a.get("depressor_ao_r", 0)
        l_expr = a.get("zygomaticus_l", 0) + a.get("frontalis_l", 0) + a.get("depressor_ao_l", 0)
        peak   = max(r_expr, l_expr, 0.01)
        asym   = abs(r_expr - l_expr) / peak
        if asym > 0.55 and peak > 0.5 and _since_last("_asym", 6.0):
            side = "right" if r_expr > l_expr else "left"
            self._log.insert(0, {
                "t": now, "ts": time.strftime("%H:%M:%S"),
                "muscle": f"Facial asymmetry ({side} dominant)",
                "au": "ASYM", "msg": f"Asymmetric expression — {side} side leading",
                "clinical": "Microexpression leakage · possible neurological flag · deception signal",
                "intensity": "moderate", "kind": "composite", "act": round(asym, 2),
            })
            self._log = self._log[:self.LOG_MAX]
            self._last_logged["_asym"] = now

    @property
    def active_now(self) -> list[str]:
        return [f"{m['name']} ({m['au']})" for k, m in self.MUSCLES.items()
                if self.activations.get(k, 0) > self.ACT_THR]


def _draw_muscle_log(frame, logger: FacialMuscleLogger, w, h):
    """
    Right-side scrolling clinical log.
    Each entry shows: timestamp · AU code · muscle event · clinical note.
    """
    if not logger._log:
        return

    pw     = 310
    lh     = 40           # pixels per log line
    vis    = min(7, len(logger._log))
    pan_h  = vis * lh + 20
    px0    = w - pw - 4
    py0    = 232          # below rPPG badge

    ov = frame.copy()
    cv2.rectangle(ov, (px0-3, py0), (w-3, py0 + pan_h), _DARK, -1)
    cv2.addWeighted(ov, 0.86, frame, 0.14, 0, frame)
    cv2.rectangle(frame, (px0-3, py0), (w-3, py0 + pan_h), (60, 60, 80), 1)

    cv2.putText(frame, "MUSCLE  LOG", (px0, py0 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, _GREY, 1, cv2.LINE_AA)

    now = logger._fc / FPS
    for i, entry in enumerate(logger._log[:vis]):
        y   = py0 + 20 + i * lh
        age = now - entry["t"]
        # Color by kind / intensity
        if   entry["kind"] == "composite":              col = _PINK
        elif entry["intensity"] == "strong":            col = _RED
        elif entry["intensity"] == "moderate":          col = _AMBER
        else:                                           col = _TEAL

        # Dim older entries slightly
        dim = max(0.55, 1.0 - age / 30.0)
        col = tuple(int(c * dim) for c in col)

        # Line 1: timestamp + AU
        cv2.putText(frame, f"{entry['ts']}  {entry['au']}  [{entry['act']:.2f}]",
                    (px0, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.27, _GREY, 1, cv2.LINE_AA)
        # Line 2: muscle event (bold color)
        cv2.putText(frame, entry["msg"],
                    (px0, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)
        # Line 3: clinical interpretation
        cv2.putText(frame, entry["clinical"],
                    (px0, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.26, _GREY, 1, cv2.LINE_AA)
        cv2.line(frame, (px0, y + 38), (w - 4, y + 38), (40, 40, 55), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL-MESH DEFORMATION ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════
# Tessellation edges (loaded once — ~1400 edges covering all 468 landmarks)
_TESS_EDGES: list | None = None

def _tess():
    global _TESS_EDGES
    if _TESS_EDGES is None:
        _TESS_EDGES = list(mp.solutions.face_mesh.FACEMESH_TESSELATION)
    return _TESS_EDGES


class FaceMeshDeformationAnalyzer:
    """
    Maps all 468 FaceMesh landmarks per frame.

    Algorithm:
      1. During calibration (first 30s) — accumulate per-landmark mean positions
         as the personal "neutral face" reference.
      2. Each frame — fit an affine transform from reference→current using only
         bony rigid landmarks (nose bridge). This captures global head motion.
      3. Apply the affine to all 468 reference positions → "predicted" positions
         assuming purely rigid motion.
      4. Residuals = actual − predicted = non-linear facial muscle deformation.

    Why this beats zone tracking:
    • Full face coverage — correlated deformations across 40+ landmarks
      (a genuine Duchenne smile, cheek rise + orbital squeeze) are visible.
    • Mathematically removes head motion before detection — not just
      subtracting a rolling centroid as before.
    • No manual zone tuning — every landmark participates equally.
    """

    # Bony landmarks minimally affected by soft-tissue motion
    RIGID_IDX = [6, 168, 197, 195, 5, 1, 4, 19, 94]   # nose bridge + cartilage

    # Named facial regions → landmark index lists for region-score aggregation
    REGIONS = {
        "FOREHEAD": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     109,  67, 103,  54,  21, 162, 127, 234,  93, 132,  58, 172],
        "L_BROW":   [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
        "R_BROW":   [107,  66, 105,  63,  70,  46,  53,  52,  65,  55],
        "L_EYE":    [362, 385, 387, 263, 373, 380, 374, 386, 382, 381,
                     398, 384, 395, 279, 330, 368, 264, 256, 252, 253],
        "R_EYE":    [33,  160, 158, 133, 153, 144, 145, 159, 157, 156,
                     173, 163, 144,   7, 246,  30,  29,  27,  28,  56],
        "NOSE":     [1,  2,  5,  6, 19, 94, 98, 99, 129, 358, 327, 326,
                     0, 45, 275, 44, 237, 218, 440,  79,  80, 166, 399],
        "L_CHEEK":  [116, 123, 147, 213, 192, 214, 210, 202, 204, 194,
                      93, 132,  58, 172, 136, 150, 149, 176],
        "R_CHEEK":  [345, 352, 376, 433, 412, 434, 430, 422, 424, 418,
                     322, 361, 288, 397, 365, 379, 378, 400],
        "MOUTH":    [61, 185,  40,  39,  37,   0, 267, 269, 270, 409,
                     291, 375, 321, 405, 314,  17,  84, 181,  91, 146,
                      78,  95,  88, 178,  87,  14, 317, 402, 318, 324,
                     308, 415, 310, 311, 312,  13,  82,  81,  80, 191],
        "JAW":      [172, 136, 150, 149, 176, 148, 152, 377, 400, 378,
                     379, 365, 397, 288, 361, 323, 454, 356, 389, 251],
    }

    def __init__(self):
        self._ref: np.ndarray | None = None   # (468, 2) neutral mean positions (pixels)
        self._ref_n   = 0                      # frames accumulated during calibration
        self._cal_done = False
        self.residuals  = np.zeros((468, 2))   # (468, 2) deformation vectors
        self.magnitudes = np.zeros(468)        # (468,)  deformation magnitudes
        self.region_scores: dict[str, float] = {}
        self._norm_max  = deque(maxlen=FPS * 5)  # rolling 95th-percentile for color scaling
        self.scale = 1.0    # current normalization value (px)

    def update(self, lm, fw, fh, calibrated: bool):
        n = len(lm)
        if n < 468:
            return
        curr = np.array([(lm[i].x * fw, lm[i].y * fh) for i in range(468)], dtype=np.float32)

        # Calibration phase: accumulate running mean
        if not calibrated:
            self._ref_n += 1
            if self._ref is None:
                self._ref = curr.copy()
            else:
                α = 1.0 / self._ref_n
                self._ref = (1 - α) * self._ref + α * curr
            return

        if self._ref is None:
            self._ref = curr.copy()
            return

        # ── Affine fit on rigid landmarks ──────────────────────────────────────
        src = self._ref[self.RIGID_IDX]        # reference bony points (N, 2)
        dst = curr[self.RIGID_IDX]             # current bony points (N, 2)
        src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float32)])
        # Least-squares: finds A (3×2) mapping reference → current via rigid set
        A, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)

        # ── Predict all 468 positions under this affine ──────────────────────
        ref_h  = np.hstack([self._ref, np.ones((468, 1), dtype=np.float32)])
        predicted = ref_h @ A                  # (468, 2) — rigid-only prediction

        # ── Residuals: what's left after removing head motion ─────────────────
        self.residuals  = curr - predicted     # (468, 2) — pure muscle deformation
        self.magnitudes = np.linalg.norm(self.residuals, axis=1)  # (468,)

        # Rolling normalization (95th percentile of magnitudes, 5s window)
        self._norm_max.append(float(np.percentile(self.magnitudes, 95)))
        self.scale = max(float(np.mean(self._norm_max)) * 2.2, 0.8)

        # Region scores (mean deformation / scale → 0–1)
        for region, idxs in self.REGIONS.items():
            valid = [i for i in idxs if i < 468]
            if valid:
                self.region_scores[region] = min(
                    1.0, float(np.mean(self.magnitudes[valid])) / self.scale)

    def top_k_zones(self, k=30) -> np.ndarray:
        """Return indices of the k most-deformed landmarks."""
        return np.argsort(self.magnitudes)[-k:]


def _draw_deformation_overlay(frame, deform: FaceMeshDeformationAnalyzer,
                               lm, fw, fh, mesh_alpha=0.60, arrow_alpha=0.85):
    """
    Draws the full tessellated face mesh colored by residual deformation magnitude.

    • Each edge is colored on a teal→amber→red gradient (low→high deformation).
    • The top 15% most-deformed landmarks get deformation-vector arrows showing
      the direction and magnitude of their muscle-driven displacement.
    • Drawn on a blended overlay so the live camera image shows through.
    """
    if deform._ref is None:
        return frame

    tess  = _tess()
    mags  = deform.magnitudes
    scale = max(deform.scale, 0.5)
    h, w  = frame.shape[:2]

    # ── Colored mesh wireframe ──────────────────────────────────────────────
    mesh_overlay = frame.copy()
    for i_lm, j_lm in tess:
        if i_lm >= 468 or j_lm >= 468:
            continue
        mag = max(mags[i_lm], mags[j_lm]) / scale
        mag = float(np.clip(mag, 0.0, 1.0))
        # teal → amber → red
        if mag < 0.5:
            t = mag * 2.0
            col = (int(_TEAL[0]*(1-t) + _AMBER[0]*t),
                   int(_TEAL[1]*(1-t) + _AMBER[1]*t),
                   int(_TEAL[2]*(1-t) + _AMBER[2]*t))
        else:
            t = (mag - 0.5) * 2.0
            col = (int(_AMBER[0]*(1-t) + _RED[0]*t),
                   int(_AMBER[1]*(1-t) + _RED[1]*t),
                   int(_AMBER[2]*(1-t) + _RED[2]*t))
        x1, y1 = int(lm[i_lm].x * fw), int(lm[i_lm].y * fh)
        x2, y2 = int(lm[j_lm].x * fw), int(lm[j_lm].y * fh)
        cv2.line(mesh_overlay, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)

    cv2.addWeighted(mesh_overlay, mesh_alpha, frame, 1.0 - mesh_alpha, 0, frame)

    # ── Deformation-vector arrows at high-residual landmarks ────────────────
    arrow_overlay = frame.copy()
    top_k = deform.top_k_zones(k=int(468 * 0.15))   # top 15%
    for idx in top_k:
        px_ = int(lm[idx].x * fw)
        py_ = int(lm[idx].y * fh)
        res = deform.residuals[idx]
        # Scale arrow length: max 18px for the largest deformation
        arrow_scale = min(18.0, 18.0 * mags[idx] / scale)
        ex = int(px_ + res[0] / max(mags[idx], 1e-4) * arrow_scale)
        ey = int(py_ + res[1] / max(mags[idx], 1e-4) * arrow_scale)
        ex = max(0, min(fw - 1, ex)); ey = max(0, min(fh - 1, ey))
        if abs(ex - px_) + abs(ey - py_) > 3:
            mag_n = mags[idx] / scale
            col = _RED if mag_n > 0.75 else (_AMBER if mag_n > 0.4 else _TEAL)
            cv2.arrowedLine(arrow_overlay, (px_, py_), (ex, ey),
                            col, 1, cv2.LINE_AA, tipLength=0.45)

    cv2.addWeighted(arrow_overlay, arrow_alpha, frame, 1.0 - arrow_alpha, 0, frame)
    return frame


def _draw_region_scores(frame, deform: FaceMeshDeformationAnalyzer, w, h):
    """
    Left-side stacked bar chart: per-region mean deformation score.
    Teal = calm, amber = moderate, red = active contraction.
    """
    if not deform.region_scores:
        return
    regions = list(deform.REGIONS.keys())
    bar_w = 88; bar_h = 9; gap = 2
    total_h = len(regions) * (bar_h + gap) + 24
    px0 = 10; py0 = 52

    ov = frame.copy()
    cv2.rectangle(ov, (px0-3, py0-3), (px0 + bar_w + 56, py0 + total_h), _DARK, -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, "MUSCLE DEFORMATION", (px0, py0 + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, _GREY, 1, cv2.LINE_AA)

    for i, region in enumerate(regions):
        score = deform.region_scores.get(region, 0.0)
        by = py0 + 18 + i * (bar_h + gap)
        fill = int(bar_w * min(1.0, score))
        col = _RED if score > 0.70 else (_AMBER if score > 0.35 else _TEAL)
        cv2.rectangle(frame, (px0, by), (px0 + bar_w, by + bar_h), (32, 32, 42), -1)
        if fill > 0:
            cv2.rectangle(frame, (px0, by), (px0 + fill, by + bar_h), col, -1)
        cv2.putText(frame, region[:7], (px0 + bar_w + 4, by + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, _GREY, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# BSV AGGREGATOR  — head-stabilized micro-movements + triangle strain
# ═══════════════════════════════════════════════════════════════════════════════
def compute_bsv(zone_bufs, ear_buf, geom_buf, brow: BrowAnalyzer,
                gaze: GazeAnalyzer, head: HeadPoseEstimator, cal: AdaptiveCalibrator):
    """
    Returns a BSV dict. All landmark positions in zone_bufs are already
    head-stabilized (drift removed). Twitch detection uses both velocity
    AND triangle-deformation strain.
    """
    ear_list  = list(ear_buf)
    geom_list = list(geom_buf)
    if len(ear_list) < 10:
        return {}

    blink_r = _blink_rate(ear_list)

    # Micro-movement velocity per zone (head-stabilized)
    all_tw, all_vel = [], []
    zone_vel = {}
    for zone, buf in zone_bufs.items():
        hist = list(buf)
        if len(hist) < 15:
            continue
        pos  = np.array(hist, dtype=float)
        macro = _lowpass(pos, 0.5)
        micro = pos - macro
        vel  = np.linalg.norm(np.diff(micro, axis=0), axis=1)
        zone_vel[zone] = float(np.mean(vel))
        all_vel.extend(vel.tolist())
        # Adaptive threshold
        cal.add(f"vel_{zone}", float(np.mean(vel)))
        thr = cal.threshold(f"vel_{zone}", z=2.5) if cal.calibrated else (np.mean(vel) + 2.5 * np.std(vel))
        peaks, _ = find_peaks(vel, height=thr, distance=3)
        for p in peaks:
            all_tw.append({"zone": zone, "frame": int(p)})

    # Triangle deformation strain
    zone_positions = {z: list(zone_bufs[z])[-1] if zone_bufs[z] else None for z in _ZONES}
    strain_zones = set()
    for tri in _STRAIN_TRIS:
        if not all(zone_positions[z] is not None for z in tri):
            continue
        # Compare triangle edge lengths now vs 15 frames ago
        for z in tri:
            buf = list(zone_bufs[z])
            if len(buf) < 16:
                break
            pts_now  = np.array([zone_positions[z] for z in tri])
            pts_past = np.array([zone_bufs[z][-16] for z in tri])
            def edge_len(pts, i, j):
                return np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
            strain = abs(edge_len(pts_now, 0, 1) - edge_len(pts_past, 0, 1)) + \
                     abs(edge_len(pts_now, 1, 2) - edge_len(pts_past, 1, 2))
            cal.add(f"strain_{tri[0]}", strain)
            if cal.calibrated and cal.z(f"strain_{tri[0]}", strain) > 2.5:
                strain_zones.update(tri)
            break

    agitation = min(1.0, float(np.mean(all_vel)) * 250.0) if all_vel else 0.0

    recent = geom_list[-30:] if len(geom_list) >= 30 else geom_list
    if recent:
        ls   = float(np.std([g["lip_width"] for g in recent]))
        bs   = float(np.std([g["brow_gap"]  for g in recent]))
        flat = max(0.0, 1.0 - (ls + bs) * 80.0)
        alp  = float(np.mean([g["lip_width"] for g in recent]))
        abw  = float(np.mean([g["brow_gap"]  for g in recent]))
        val  = float(np.clip((alp - 0.35) * 4.0 - (0.04 - abw) * 8.0, -1.0, 1.0))
    else:
        flat = 0.5; val = 0.0

    aus = []
    if geom_list:
        last = geom_list[-1]
        if last["blink"]:            aus.append("AU46")
        if last["brow_gap"] < 0.03:  aus.append("AU4")
        if last["lip_width"] > 0.40: aus.append("AU12")
        if last["lip_open"]  > 0.02: aus.append("AU25/26")
    if brow.au1: aus.append("AU1")
    if brow.au2: aus.append("AU2")
    if brow.au4 and "AU4" not in aus: aus.append("AU4")

    # Combine velocity + strain twitches
    vel_zones = list(set(t["zone"] for t in all_tw[-12:]))
    twitch_zones = list(set(vel_zones) | strain_zones)

    return {
        "facial_valence":     round(val,       3),
        "facial_arousal":     round(agitation, 3),
        "blink_rate_per_min": round(blink_r,   1),
        "flat_affect_score":  round(flat,      3),
        "active_aus":         aus,
        "twitch_zones":       twitch_zones,
        "strain_zones":       list(strain_zones),
        "au1": brow.au1, "au2": brow.au2, "au4": brow.au4,
        "brow_asymmetry":    round(brow.asym, 3),
        "brow_flash":        brow.brow_flash,
        "gaze_state":        gaze.state,
        "concentration":     gaze.concentration,
        "saccade_rate":      round(gaze.saccade_rate, 1),
        "fixation_dur":      round(gaze.fixation_dur, 2),
        "head_yaw":          round(head.yaw,   1),
        "head_pitch":        round(head.pitch, 1),
        "head_roll":         round(head.roll,  1),
        "face_detected":     True,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# EMOTION CLASSIFIER  — EfficientNet-B0 ONNX (AffectNet 8-class, ~15 MB)
#   Runs in a daemon thread; main loop pushes face crops every 15 frames.
#   Zero tensorflow — pure onnxruntime.
# ═══════════════════════════════════════════════════════════════════════════════
class EmotionClassifier:
    LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    COLORS = [
        (140, 140, 140),  # Neutral — grey
        (50,  200,  70),  # Happy   — green
        (200, 120,  30),  # Sad     — blue-ish
        (0,   230, 255),  # Surprise— cyan
        (160,  60, 200),  # Fear    — purple
        (30,  160,  80),  # Disgust — olive-green
        (30,   40, 220),  # Anger   — red
        (0,   190, 240),  # Contempt— amber
    ]
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "emotion_enet.onnx")

    def __init__(self):
        self._sess   = None   # loaded inside daemon thread
        self._q      = queue.Queue(maxsize=2)
        self._lock   = threading.Lock()
        self.emotion = "loading…"
        self.scores  = np.zeros(8, dtype=np.float32)
        self._ready  = False
        t = threading.Thread(target=self._run, daemon=True, name="EmotionNet")
        t.start()

    def _run(self):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(self.MODEL_PATH,
                                          providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name
        self._ready = True
        while True:
            crop = self._q.get()   # blocks until a crop arrives
            if crop is None:
                break
            try:
                x = self._preprocess(crop)
                logits = self._sess.run(None, {self._in_name: x})[0][0]  # (8,)
                probs  = self._softmax(logits)
                top    = int(np.argmax(probs))
                with self._lock:
                    self.scores  = probs
                    self.emotion = self.LABELS[top]
            except Exception:
                pass

    def push(self, face_crop_bgr):
        if not self._ready:
            return
        if self._q.full():
            try: self._q.get_nowait()
            except queue.Empty: pass
        try: self._q.put_nowait(face_crop_bgr)
        except queue.Full: pass

    def get(self):
        with self._lock:
            return self.emotion, self.scores.copy()

    def _preprocess(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        x   = rgb.astype(np.float32) / 255.0
        x   = (x - self.MEAN) / self.STD
        return x.transpose(2, 0, 1)[np.newaxis]   # (1,3,224,224)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x)); return e / e.sum()


def _face_crop(frame, lm, fw, fh, pad=0.22):
    """Return tight face crop from MediaPipe landmarks with padding."""
    xs = [l.x for l in lm]; ys = [l.y for l in lm]
    x0 = max(0,  int((min(xs) - pad) * fw))
    y0 = max(0,  int((min(ys) - pad) * fh))
    x1 = min(fw, int((max(xs) + pad) * fw))
    y1 = min(fh, int((max(ys) + pad) * fh))
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


def _draw_emotion_panel(frame, clf: EmotionClassifier, w, h):
    """Left-side emotion panel: dominant label + 8 confidence bars."""
    emotion, scores = clf.get()
    top_idx  = int(np.argmax(scores))
    top_conf = float(scores[top_idx])

    px0 = 8; py0 = 135; pw = 148; lh = 14
    pan_h = 22 + 8 * lh + 4
    ov = frame.copy()
    cv2.rectangle(ov, (px0, py0), (px0 + pw, py0 + pan_h), _DARK, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px0 + pw, py0 + pan_h),
                  EmotionClassifier.COLORS[top_idx], 1)

    # Dominant emotion headline
    cv2.putText(frame, "EMOTION", (px0 + 2, py0 + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, _GREY, 1, cv2.LINE_AA)
    lbl = f"{emotion}  {top_conf*100:.0f}%"
    cv2.putText(frame, lbl, (px0 + 2, py0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, EmotionClassifier.COLORS[top_idx],
                1, cv2.LINE_AA)

    # 8 mini probability bars
    bar_max = 100
    for i, (label, score, col) in enumerate(zip(
            EmotionClassifier.LABELS, scores, EmotionClassifier.COLORS)):
        y = py0 + 28 + i * lh
        cv2.putText(frame, label[:7], (px0 + 2, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, _GREY, 1, cv2.LINE_AA)
        bx0 = px0 + 56; bx1 = px0 + pw - 4
        bw  = bx1 - bx0
        cv2.rectangle(frame, (bx0, y + 2), (bx1, y + 10), (38, 38, 50), -1)
        fill = int(bw * float(score))
        if fill > 0:
            cv2.rectangle(frame, (bx0, y + 2), (bx0 + fill, y + 10), col, -1)
        # highlight top bar
        if i == top_idx:
            cv2.rectangle(frame, (bx0, y + 2), (bx1, y + 10), col, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════════════════════
def _px(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def _draw_frame(frame, lm_list, bsv, breath, neck, gaze, brow, rppg, head, cal, deform=None, muscle_logger=None, emotion_clf=None):
    h, w = frame.shape[:2]

    if lm_list:
        lm = lm_list
        tw_zones    = set(bsv.get("twitch_zones", []))
        strain_zones= set(bsv.get("strain_zones", []))

        # ── Full-mesh deformation overlay (drawn first, lowest z-order) ──────
        if deform is not None and deform._ref is not None:
            _draw_deformation_overlay(frame, deform, lm, w, h)

        # Contours
        for pts, col in [(_EYE_R_C, _TEAL), (_EYE_L_C, _TEAL)]:
            cv2.polylines(frame, [np.array([_px(lm,i,w,h) for i in pts], np.int32)], True, col, 1, cv2.LINE_AA)
        for pts in [_LIP_OUT, _LIP_IN]:
            cv2.polylines(frame, [np.array([_px(lm,i,w,h) for i in pts], np.int32)], True, _TEAL, 1, cv2.LINE_AA)

        # Brow shape lines
        for pts in [_R_BROW_S, _L_BROW_S]:
            col = _RED if brow.au4 else _AMBER
            thick = 2 if brow.au4 else 1
            cv2.polylines(frame, [np.array([_px(lm,i,w,h) for i in pts], np.int32)], False, col, thick, cv2.LINE_AA)
        if brow.brow_flash:
            fx, fy = _px(lm, 10, w, h)
            cv2.putText(frame, "BROW FLASH", (fx - 40, fy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, _GREEN, 1, cv2.LINE_AA)
        # AU1/2 highlight dots
        for idx_, au_flag, col_ in [(107, brow.au1, _GREEN), (336, brow.au1, _GREEN),
                                    (46,  brow.au2, _GREEN), (276, brow.au2, _GREEN)]:
            if au_flag:
                cv2.circle(frame, _px(lm,idx_,w,h), 5, col_, -1, cv2.LINE_AA)

        # Brow asymmetry line
        line_col = _RED if brow.asym > 0.25 else (50, 50, 65)
        cv2.line(frame, _px(lm,66,w,h), _px(lm,296,w,h), line_col, 1, cv2.LINE_AA)

        # 28-zone dots — color encodes twitch type
        for zone, idx in _ZONES.items():
            x, y = _px(lm, idx, w, h)
            if zone in strain_zones and zone in tw_zones:
                col = _PINK; r = 5
            elif zone in strain_zones:
                col = _AMBER; r = 4
            elif zone in tw_zones:
                col = _RED; r = 4
            else:
                col = _TEAL; r = 2
            cv2.circle(frame, (x, y), r, col, -1, cv2.LINE_AA)
            if zone in tw_zones:
                cv2.circle(frame, (x, y), r + 5, col, 1, cv2.LINE_AA)

        # Iris rings + IPD-normalized gaze arrow
        if len(lm) > 478:
            for iris_idx in (_R_IRIS, _L_IRIS):
                cv2.circle(frame, _px(lm,iris_idx,w,h), 6, _BLUE, 1, cv2.LINE_AA)
            nose = _px(lm, 6, w, h)
            ipd_px = abs(lm[_R_IRIS].x - lm[_L_IRIS].x) * w
            ax = int(nose[0] - gaze.gaze_h * ipd_px * 1.8)
            ay = int(nose[1] + gaze.gaze_v * ipd_px * 1.8)
            ax = max(5, min(w-5, ax)); ay = max(5, min(h-5, ay))
            cv2.arrowedLine(frame, nose, (ax, ay), _BLUE, 2, cv2.LINE_AA, tipLength=0.3)

        # Neck ROI + LK swallow
        chin_y_px = int(lm[152].y * h)
        face_h_px = int((lm[152].y - lm[6].y) * h)
        nx0, ny0, nx1, ny1 = neck.roi_rect(chin_y_px, face_h_px, w, h)
        # Draw tracked neck points
        if neck._pts is not None:
            for pt in neck._pts:
                px_ = (int(pt[0,0]), int(pt[0,1]))
                s_col = _RED if neck.swallow_now else (60, 60, 80)
                cv2.circle(frame, px_, 2, s_col, -1, cv2.LINE_AA)
        ov = frame.copy()
        b_col = _RED if neck.swallow_now else (30, 30, 45)
        cv2.rectangle(ov, (nx0, ny0), (nx1, ny1), b_col, -1)
        cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
        cv2.rectangle(frame, (nx0, ny0), (nx1, ny1), b_col, 1)
        sm_label = {neck.IDLE:"NECK:IDLE", neck.RISING:"NECK:RISE↑",
                    neck.PEAK:"NECK:PEAK", neck.FALLING:"NECK:FALL↓",
                    neck.RECOVERY:"SWALLOW!"}
        sm_col   = _RED if neck._state == neck.RECOVERY else _GREY
        cv2.putText(frame, sm_label.get(neck._state, ""), (nx0 + 4, ny0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, sm_col, 1, cv2.LINE_AA)

        # Head pose compass (small, top-left area)
        _draw_head_compass(frame, head, 70, 95, 28)

        # AU labels
        active_aus = bsv.get("active_aus", [])
        au_anch = {
            "AU1": (lm[107].x-0.06, lm[107].y-0.05),
            "AU2": (lm[46].x+0.01,  lm[46].y-0.05),
            "AU4": (lm[66].x-0.02,  lm[66].y-0.04),
            "AU12":(lm[61].x-0.07,  lm[61].y+0.03),
            "AU25/26":(lm[14].x,    lm[14].y+0.05),
            "AU46":(lm[159].x,      lm[159].y-0.04),
        }
        for au, (ax, ay) in au_anch.items():
            if au in active_aus:
                cv2.putText(frame, au, (int(ax*w), int(ay*h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, _AMBER, 1, cv2.LINE_AA)

    # Calibration overlay
    if not cal.calibrated:
        prog = cal.progress
        bw = int((w - 40) * prog)
        cv2.rectangle(frame, (20, h//2 - 18), (w-20, h//2 + 18), (20,20,30), -1)
        cv2.rectangle(frame, (20, h//2 - 18), (20 + bw, h//2 + 18), _TEAL, -1)
        cv2.putText(frame, f"CALIBRATING PERSONAL BASELINE  {int(prog*100)}%",
                    (30, h//2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1, cv2.LINE_AA)

    _draw_breath_panel(frame, breath, w, h)
    _draw_gaze_badge(frame, gaze, w)
    _draw_concentration_bar(frame, gaze.concentration, w)
    _draw_rppg_badge(frame, rppg, w, h)
    if deform is not None:
        _draw_region_scores(frame, deform, w, h)
    if muscle_logger is not None:
        _draw_muscle_log(frame, muscle_logger, w, h)
    if emotion_clf is not None:
        _draw_emotion_panel(frame, emotion_clf, w, h)
    _draw_top_bar(frame, lm_list is not None, head, w)
    _draw_bottom_hud(frame, bsv, breath, gaze, neck, rppg, w, h)
    return frame

def _draw_head_compass(frame, head, cx, cy, r):
    """Small yaw/pitch indicator circle."""
    cv2.circle(frame, (cx, cy), r, (40, 40, 55), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (70, 70, 90), 1, cv2.LINE_AA)
    # Yaw → horizontal; pitch → vertical dot
    dx = int(np.clip(head.yaw   / 35.0, -1, 1) * (r - 5))
    dy = int(np.clip(head.pitch / 25.0, -1, 1) * (r - 5))
    cv2.circle(frame, (cx + dx, cy + dy), 4, _TEAL, -1, cv2.LINE_AA)
    cv2.putText(frame, "HEAD", (cx - 16, cy + r + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, _GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Y{head.yaw:+.0f}°", (cx - 18, cy + r + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, _GREY, 1, cv2.LINE_AA)

def _draw_breath_panel(frame, breath: BreathingAnalyzer, w, h):
    pw, ph = 118, 78; px0 = w - pw - 4; py0 = 100
    ov = frame.copy()
    cv2.rectangle(ov, (px0, py0), (px0+pw, py0+ph), _DARK, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px0+pw, py0+ph), (55, 55, 70), 1)
    cv2.putText(frame, "BREATHING", (px0+4, py0+13), cv2.FONT_HERSHEY_SIMPLEX, 0.28, _GREY, 1, cv2.LINE_AA)
    r_str = f"{breath.rate:.0f} bpm" if breath.rate > 0 else "—"
    d_col = _RED if breath.depth == "SHALLOW" else (_PINK if breath.depth == "DEEP" else _GREEN)
    cv2.putText(frame, r_str, (px0+4, py0+27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, breath.depth, (px0+68, py0+27), cv2.FONT_HERSHEY_SIMPLEX, 0.32, d_col, 1, cv2.LINE_AA)
    wave = list(breath.wave)
    if len(wave) > 4:
        cy0 = py0 + 32; ch = ph - 36
        mn, mx = min(wave), max(wave); rng = mx - mn if mx != mn else 1e-6
        pts = [(px0 + 2 + int(i*(pw-4)/max(len(wave)-1,1)),
                cy0 + int((1-(v-mn)/rng)*ch)) for i, v in enumerate(wave)]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], _GREEN, 1, cv2.LINE_AA)

def _draw_gaze_badge(frame, gaze: GazeAnalyzer, w):
    col = _STATE_COL.get(gaze.state, _GREY)
    bx = w - 168; by0 = 36
    # Panel height: state + 4 metric rows
    pan_h = 118
    ov = frame.copy()
    cv2.rectangle(ov, (bx-4, by0), (w-4, by0+pan_h), _DARK, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (bx-4, by0), (w-4, by0+pan_h), col, 1)

    # State headline
    cv2.putText(frame, "GAZE STATE", (bx, by0+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, _GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, gaze.state,   (bx, by0+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col,  1, cv2.LINE_AA)
    cv2.putText(frame, gaze.gaze_dir,(bx+90, by0+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, _BLUE, 1, cv2.LINE_AA)

    # PERCLOS bar
    py = by0 + 44
    pc_pct = int(gaze.perclos * 100)
    pc_col = _RED if gaze.perclos > 0.15 else (_AMBER if gaze.perclos > 0.08 else _TEAL)
    cv2.putText(frame, f"PERCLOS {pc_pct:2d}%", (bx, py+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, pc_col, 1, cv2.LINE_AA)
    bw = 80; fill = int(bw * min(gaze.perclos / 0.25, 1.0))
    cv2.rectangle(frame, (bx+80, py+2), (bx+80+bw, py+10), (38,38,50), -1)
    cv2.rectangle(frame, (bx+80, py+2), (bx+80+fill, py+10), pc_col, -1)

    # Microsaccade rate
    msr_col = _AMBER if gaze.microsaccade_rate > 80 else _GREY
    cv2.putText(frame, f"μSAC  {gaze.microsaccade_rate:.0f}/m", (bx, py+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, msr_col, 1, cv2.LINE_AA)

    # Gaze entropy
    ge_col = _AMBER if gaze.gaze_entropy > 3.0 else _GREY
    cv2.putText(frame, f"ENTROPY {gaze.gaze_entropy:.1f}b", (bx, py+42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, ge_col, 1, cv2.LINE_AA)

    # Divergence indicator
    div_txt = "NEAR" if gaze.divergence < -0.15 else ("DISS" if gaze.divergence > 0.20 else "MID")
    div_col = _AMBER if div_txt == "DISS" else (_TEAL if div_txt == "NEAR" else _GREY)
    cv2.putText(frame, f"DIV   {div_txt}", (bx+80, py+42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, div_col, 1, cv2.LINE_AA)

    # Fixation duration
    cv2.putText(frame, f"FIX  {gaze.fixation_dur:.1f}s", (bx+80, py+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, _GREY, 1, cv2.LINE_AA)

def _draw_concentration_bar(frame, score, w):
    bw = 182; bh = 8; bx = 10; by = 35
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (38, 38, 50), -1)
    fill = int(bw * score / 100)
    col  = _RED if score < 35 else (_AMBER if score < 60 else _GREEN)
    cv2.rectangle(frame, (bx, by), (bx+fill, by+bh), col, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (60,60,75), 1)
    cv2.putText(frame, f"CONCENTRATION  {score}", (bx+bw+6, by+8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, _GREY, 1, cv2.LINE_AA)

def _draw_rppg_badge(frame, rppg: RPPGAnalyzer, w, h):
    if rppg.bpm == 0:
        return
    bx = w - 168; by = 185
    cv2.rectangle(frame, (bx-4, by), (w-4, by+38), _DARK, -1)
    cv2.rectangle(frame, (bx-4, by), (w-4, by+38), _PINK, 1)
    cv2.putText(frame, "HEART RATE", (bx, by+13), cv2.FONT_HERSHEY_SIMPLEX, 0.28, _GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{rppg.bpm:.0f} bpm", (bx, by+30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _PINK, 1, cv2.LINE_AA)
    # rPPG mini waveform
    wave = list(rppg.wave)
    if len(wave) > 10:
        mn, mx = min(wave), max(wave); rng = mx - mn if mx != mn else 1e-6
        pw = 80; ph = 18; wx0 = bx + 80; wy0 = by + 18
        pts = [(wx0 + int(i*pw/max(len(wave)-1,1)), wy0 + int((1-(v-mn)/rng)*ph))
               for i, v in enumerate(wave[-80:])]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], _PINK, 1, cv2.LINE_AA)

def _draw_top_bar(frame, face_ok, head, w):
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,28), _DARK, -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    lbl = "SOMATIC ACTIVE" if face_ok else "NO FACE DETECTED"
    cv2.putText(frame, lbl, (10,19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                _TEAL if face_ok else _RED, 1, cv2.LINE_AA)
    cv2.putText(frame, "MINDSCAPE  SOMATIC  v3", (w-220, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GREY, 1, cv2.LINE_AA)

def _draw_bottom_hud(frame, bsv, breath, gaze, neck, rppg, w, h):
    hud_h = 82; y0 = h - hud_h
    ov = frame.copy()
    cv2.rectangle(ov, (0, y0), (w, h), _DARK, -1)
    cv2.addWeighted(ov, 0.84, frame, 0.16, 0, frame)
    cv2.line(frame, (0, y0), (w, y0), (55,55,68), 1)

    def hud(label, val, x, col=_WHITE):
        cv2.putText(frame, label, (x, y0+18), cv2.FONT_HERSHEY_SIMPLEX, 0.27, _GREY,  1, cv2.LINE_AA)
        cv2.putText(frame, val,   (x, y0+34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col,    1, cv2.LINE_AA)

    cw = (w - 20) // 8
    hud("VALENCE",   f"{bsv.get('facial_valence',0):+.2f}",     10)
    hud("AGITATION", f"{bsv.get('facial_arousal',0):.2f}",       10+cw)
    hud("BLINK/MIN", f"{bsv.get('blink_rate_per_min',0):.0f}",   10+cw*2)
    hud("FLAT AFCT", f"{bsv.get('flat_affect_score',0):.2f}",    10+cw*3)
    hud("SACCADE/M", f"{gaze.saccade_rate:.0f}",                  10+cw*4, _BLUE)
    hud("FIXATION",  f"{gaze.fixation_dur:.1f}s",                 10+cw*5, _BLUE)
    hud("BREATH",    f"{breath.rate:.0f}bpm",                     10+cw*6, _GREEN)
    hud("SWALLOW",   str(neck.swallow_count),                     10+cw*7, _RED if neck.swallow_now else _WHITE)

    def hud2(label, val, x, col=_GREY):
        cv2.putText(frame, label, (x, y0+50), cv2.FONT_HERSHEY_SIMPLEX, 0.25, _GREY, 1, cv2.LINE_AA)
        cv2.putText(frame, val,   (x, y0+64), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col,   1, cv2.LINE_AA)

    aus_str = " ".join(bsv.get("active_aus", [])) or "—"
    hud2("AUs",       aus_str,                                          10)
    hud2("BROW",      f"A1:{int(bsv.get('au1',0))} A2:{int(bsv.get('au2',0))} A4:{int(bsv.get('au4',0))}",
                                                                         10+cw*2, _AMBER)
    hud2("HEAD",      f"Y{bsv.get('head_yaw',0):+.0f}° P{bsv.get('head_pitch',0):+.0f}°",
                                                                         10+cw*4, _TEAL)
    hud2("HR(rPPG)",  f"{rppg.bpm:.0f}bpm" if rppg.bpm else "cal...",   10+cw*6, _PINK)

    # Twitch zone pills
    pill_x = 10
    for z in bsv.get("twitch_zones", [])[:7]:
        lbl = z.upper()
        (tw,_),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.26, 1)
        bg = _PINK if z in bsv.get("strain_zones",[]) else (50, 20, 80)
        cv2.rectangle(frame, (pill_x-2, y0+68), (pill_x+tw+6, y0+76), bg, -1)
        cv2.putText(frame, lbl, (pill_x+2, y0+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, _RED, 1, cv2.LINE_AA)
        pill_x += tw + 11

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK / MJPEG SERVER
# ═══════════════════════════════════════════════════════════════════════════════
_fq: queue.Queue = queue.Queue(maxsize=4)
_bsv_lock = threading.Lock(); _bsv: dict = {}
_flask = Flask(__name__)

_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;display:flex;align-items:center;justify-content:center;height:100vh;overflow:hidden}
img{width:100%;height:100%;object-fit:cover;border-radius:10px}
</style></head><body><img src="/video_feed"></body></html>"""

@_flask.route("/video_html")
def video_html(): return _HTML, 200, {"Content-Type": "text/html"}

@_flask.route("/video_feed")
def video_feed():
    def gen():
        while True:
            try:
                fb = _fq.get(timeout=1.0)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n"
            except queue.Empty: continue
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def _start_flask():
    import logging; logging.getLogger("werkzeug").setLevel(logging.ERROR)
    _flask.run(host="127.0.0.1", port=5001, threaded=True, use_reloader=False)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP  — camera on main thread (macOS AVFoundation requirement)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n  MindScape — Somatic Stream  v3 (Robust)")
    print("  ─────────────────────────────────────────")

    threading.Thread(target=_start_flask, daemon=True).start()
    time.sleep(0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: camera unavailable. Grant Terminal camera access.")
        sys.exit(1)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {fw}×{fh}")
    print("  ✓  http://localhost:5001/video_html")
    print("  Calibrating for 30 seconds — sit still and relaxed.\n")

    BUFL = FPS * 5
    zone_bufs = {z: deque(maxlen=BUFL) for z in _ZONES}
    ear_buf   = deque(maxlen=BUFL)
    geom_buf  = deque(maxlen=BUFL)

    head   = HeadPoseEstimator()
    cal    = AdaptiveCalibrator()
    breath = BreathingAnalyzer()
    neck   = NeckSwallowDetector()
    gaze   = GazeAnalyzer()
    brow   = BrowAnalyzer()
    rppg   = RPPGAnalyzer()
    deform        = FaceMeshDeformationAnalyzer()
    muscle_logger = FacialMuscleLogger()
    emotion_clf   = EmotionClassifier()
    fc            = 0

    mp_mesh = mp.solutions.face_mesh
    with mp_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as mesh:
        try:
            while True:
                ret, frame = cap.read()
                if not ret: time.sleep(0.033); continue

                fc += 1
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)
                lm_list = None

                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0].landmark
                    lm_list = lm

                    # Head pose + translational stabilization
                    head.update(lm, fw, fh)

                    # EAR
                    r_ear = sum(np.linalg.norm([(lm[a].x-lm[b].x)*fw,(lm[a].y-lm[b].y)*fh])
                                for a,b in [(160,144),(158,153)]) / (2*max(np.linalg.norm(
                                    [(lm[33].x-lm[133].x)*fw,(lm[33].y-lm[133].y)*fh]),1e-4))
                    l_ear = sum(np.linalg.norm([(lm[a].x-lm[b].x)*fw,(lm[a].y-lm[b].y)*fh])
                                for a,b in [(385,380),(387,373)]) / (2*max(np.linalg.norm(
                                    [(lm[362].x-lm[263].x)*fw,(lm[362].y-lm[263].y)*fh]),1e-4))
                    avg_ear = (r_ear + l_ear) / 2.0
                    ear_buf.append(avg_ear)

                    brow_gap  = (abs(lm[160].y-lm[107].y)+abs(lm[385].y-lm[336].y))/2.0
                    lip_width = abs(lm[61].x-lm[291].x)
                    lip_open  = abs(lm[13].y-lm[14].y)
                    geom_buf.append({"ear":avg_ear,"blink":avg_ear<0.21,
                                     "brow_gap":brow_gap,"lip_width":lip_width,"lip_open":lip_open})

                    # HEAD-STABILIZED zone positions (subtract translational drift)
                    for zone, idx in _ZONES.items():
                        sx, sy = head.stabilize(lm[idx].x * fw, lm[idx].y * fh)
                        zone_bufs[zone].append((sx / fw, sy / fh, lm[idx].z))

                    deform.update(lm, fw, fh, cal.calibrated)
                    muscle_logger.update(deform, lm, fw, fh, cal)
                    brow.update(lm, fw, fh, cal)

                    # Push face crop to emotion classifier every ~0.5 s
                    if fc % 15 == 0:
                        crop = _face_crop(frame, lm, fw, fh)
                        if crop is not None:
                            emotion_clf.push(crop)
                    gaze.update(lm, fw, fh,
                                _bsv.get("blink_rate_per_min", 15.0),
                                _bsv.get("flat_affect_score", 0.5))
                    rppg.update(frame, lm, fw, fh)

                    chin_y = int(lm[152].y * fh)
                    face_h = int((lm[152].y - lm[6].y) * fh)
                    breath.update(frame, chin_y, face_h, fw, fh)
                    neck.update(frame, chin_y, face_h, fw, fh)

                    # Aggregate BSV every second
                    if fc % FPS == 0:
                        snap = compute_bsv(zone_bufs, ear_buf, geom_buf, brow, gaze, head, cal)
                        snap.update({"breath_rate": breath.rate, "breath_depth": breath.depth,
                                     "swallow_count": neck.swallow_count,
                                     "heart_rate_bpm": rppg.bpm})
                        with _bsv_lock:
                            _bsv.update(snap)
                        if cal.calibrated:
                            print(f"  [{gaze.state:10s}] "
                                  f"val={snap.get('facial_valence',0):+.2f} "
                                  f"ag={snap.get('facial_arousal',0):.2f} "
                                  f"blink={snap.get('blink_rate_per_min',0):.0f} "
                                  f"sac={gaze.saccade_rate:.0f}/m "
                                  f"breath={breath.rate:.0f}bpm "
                                  f"HR={rppg.bpm:.0f} "
                                  f"sw={neck.swallow_count} "
                                  f"conc={gaze.concentration}",
                                  end="\r", flush=True)
                        else:
                            print(f"  Calibrating... {int(cal.progress*100)}%",
                                  end="\r", flush=True)
                else:
                    with _bsv_lock: _bsv["face_detected"] = False
                    breath.update(frame, fh*0.75, fh*0.25, fw, fh)
                    neck.update(frame, fh*0.75, fh*0.25, fw, fh)

                with _bsv_lock: bsv_snap = dict(_bsv)
                annotated = _draw_frame(frame.copy(), lm_list, bsv_snap,
                                        breath, neck, gaze, brow, rppg, head, cal,
                                        deform, muscle_logger, emotion_clf)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    fb = bytes(buf)
                    try: _fq.put_nowait(fb)
                    except queue.Full:
                        try: _fq.get_nowait()
                        except queue.Empty: pass
                        try: _fq.put_nowait(fb)
                        except queue.Full: pass

                time.sleep(0.033)

        except KeyboardInterrupt:
            print("\n\n  Stopped.")
        finally:
            cap.release()

if __name__ == "__main__":
    main()
