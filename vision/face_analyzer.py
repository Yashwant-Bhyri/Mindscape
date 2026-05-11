import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque

from .signal_processing import extract_micro_movements, compute_blink_rate

# Overlay colour palette (BGR)
_C_TEAL   = (180, 210, 0)    # key landmark dots
_C_RED    = (0, 60, 220)     # twitch-zone highlights
_C_AMBER  = (0, 190, 240)    # AU labels
_C_WHITE  = (230, 230, 230)  # HUD text
_C_BLUE   = (200, 120, 30)   # gaze iris ring
_C_GREEN  = (60, 200, 80)    # positive valence tint
_C_DARK   = (18, 18, 22)     # panel background

# Landmark subsets for contour drawing
_EYE_R_CONTOUR = [33, 160, 158, 133, 153, 144]
_EYE_L_CONTOUR = [362, 385, 387, 263, 373, 380]
_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

FPS = 30
BUFFER_LEN = FPS * 5  # 5-second rolling window

# EAR 6-point indices (p1=outer, p2/p3=upper, p4=inner, p5/p6=lower)
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
_R_EAR = (33, 160, 158, 133, 153, 144)   # subject's right eye
_L_EAR = (362, 385, 387, 263, 373, 380)  # subject's left eye

# Brow inner corners (nasal end — closest to nose bridge)
_R_BROW_INNER = 107
_L_BROW_INNER = 336

# Lip geometry
_LIP_R_CORNER = 61
_LIP_L_CORNER = 291
_LIP_UPPER_CENTER = 13
_LIP_LOWER_CENTER = 14

# Eye midpoint landmarks for gaze offset computation
_R_EYE_OUT = 33
_R_EYE_IN = 133
_L_EYE_OUT = 362
_L_EYE_IN = 263

# Iris centers — only available with refine_landmarks=True (indices 468-477)
_R_IRIS = 468
_L_IRIS = 473

# Facial zones tracked for micro-movement analysis, mapped to their landmark index
_ZONE_INDICES = {
    "r_eye_outer": 33,
    "l_eye_outer": 362,
    "lip_r_corner": 61,
    "lip_l_corner": 291,
    "r_brow_inner": 107,
    "l_brow_inner": 336,
}


def _ear(lm, pts):
    p1, p2, p3, p4, p5, p6 = pts
    v = (
        np.linalg.norm([lm[p2].x - lm[p6].x, lm[p2].y - lm[p6].y])
        + np.linalg.norm([lm[p3].x - lm[p5].x, lm[p3].y - lm[p5].y])
    ) / 2.0
    h = np.linalg.norm([lm[p1].x - lm[p4].x, lm[p1].y - lm[p4].y])
    return v / (h + 1e-6)


def _geometry(lm):
    """Extract per-frame geometric features from FaceMesh landmarks."""
    ear = (_ear(lm, _R_EAR) + _ear(lm, _L_EAR)) / 2.0

    # Brow-to-upper-lid gap: tighter = AU4 (brow furrow, distress/concentration)
    brow_gap = (
        abs(lm[160].y - lm[_R_BROW_INNER].y)
        + abs(lm[385].y - lm[_L_BROW_INNER].y)
    ) / 2.0

    # Lip width: wider = AU12 (lip corner pull, positive affect / smile)
    lip_width = abs(lm[_LIP_R_CORNER].x - lm[_LIP_L_CORNER].x)
    lip_open = abs(lm[_LIP_UPPER_CENTER].y - lm[_LIP_LOWER_CENTER].y)

    # Gaze offset: iris displaced from eye midpoint (darting = hypervigilance)
    gaze = 0.0
    if len(lm) > 478:
        r_mid = (lm[_R_EYE_OUT].x + lm[_R_EYE_IN].x) / 2.0
        l_mid = (lm[_L_EYE_OUT].x + lm[_L_EYE_IN].x) / 2.0
        gaze = ((lm[_R_IRIS].x - r_mid) + (lm[_L_IRIS].x - l_mid)) / 2.0

    return {
        "ear": float(ear),
        "blink": ear < 0.21,
        "brow_gap": float(brow_gap),
        "lip_width": float(lip_width),
        "lip_open": float(lip_open),
        "gaze_offset": float(gaze),
    }


class VisualBSVAnalyzer:
    """
    Background thread that continuously captures camera frames, runs MediaPipe
    FaceMesh, and aggregates a Visual Behavioral State Vector (BSV) over a
    rolling 5-second window. Designed to run alongside the audio pipeline
    without OOM conflicts — CPU-only, bounded buffers.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._thread = None
        self._cam_idx = 0
        self.running = False
        self.face_detected = False
        self.frame_count = 0

        self._zone_bufs = {z: deque(maxlen=BUFFER_LEN) for z in _ZONE_INDICES}
        self._ear_buf = deque(maxlen=BUFFER_LEN)
        self._geom_buf = deque(maxlen=BUFFER_LEN)
        self.latest: dict = {}

        # Frame storage for MJPEG stream
        self._latest_frame = None       # BGR numpy array, last captured
        self._latest_landmarks = None   # MediaPipe landmark list
        self._frame_h = 480
        self._frame_w = 640

    def start(self, cam_idx=0):
        if self.running:
            return True
        self._cam_idx = cam_idx
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VisualBSV")
        self._thread.start()
        time.sleep(0.8)  # allow camera to open before returning

        # Start MJPEG stream server (idempotent)
        from . import stream_server
        stream_server.start(self)

        return self.running

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_visual_bsv(self) -> dict:
        with self._lock:
            return dict(self.latest)

    def _loop(self):
        cap = cv2.VideoCapture(self._cam_idx)
        if not cap.isOpened():
            self.running = False
            return

        mp_mesh = mp.solutions.face_mesh
        with mp_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mesh:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.033)
                    continue

                self.frame_count += 1
                h, w = frame.shape[:2]
                self._frame_h, self._frame_w = h, w

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)

                if result.multi_face_landmarks:
                    self.face_detected = True
                    lm = result.multi_face_landmarks[0].landmark

                    for zone, idx in _ZONE_INDICES.items():
                        self._zone_bufs[zone].append((lm[idx].x, lm[idx].y, lm[idx].z))

                    g = _geometry(lm)
                    self._ear_buf.append(g["ear"])
                    self._geom_buf.append(g)

                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                        self._latest_landmarks = list(lm)
                else:
                    self.face_detected = False
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                        self._latest_landmarks = None

                # Recompute aggregate BSV every second
                if self.frame_count % FPS == 0:
                    self._aggregate()

                time.sleep(0.033)

        cap.release()

    def get_annotated_frame_jpeg(self) -> bytes | None:
        """
        Return a JPEG-encoded BGR frame with the full somatic overlay drawn on it.
        Includes: landmark dots at key zones, eye/lip contours, AU labels,
        twitch-zone highlights, live BSV HUD strip at bottom, status bar at top.
        Thread-safe — reads frame and BSV under separate locks.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()
            lm = self._latest_landmarks  # may be None

        with self._lock:
            bsv = dict(self.latest)

        h, w = frame.shape[:2]

        def px(lmk_x, lmk_y):
            return int(lmk_x * w), int(lmk_y * h)

        if lm is not None:
            # --- Eye contours ---
            for indices, color in [(_EYE_R_CONTOUR, _C_TEAL), (_EYE_L_CONTOUR, _C_TEAL)]:
                pts = np.array([px(lm[i].x, lm[i].y) for i in indices], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

            # --- Lip outer contour ---
            pts = np.array([px(lm[i].x, lm[i].y) for i in _LIP_OUTER], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=_C_TEAL, thickness=1, lineType=cv2.LINE_AA)

            # --- Key zone landmark dots (normal: teal, twitch: red) ---
            twitch_zones = set(bsv.get("twitch_zones", []))
            for zone, idx in _ZONE_INDICES.items():
                x, y = px(lm[idx].x, lm[idx].y)
                is_twitching = zone in twitch_zones
                color = _C_RED if is_twitching else _C_TEAL
                radius = 4 if is_twitching else 3
                cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
                if is_twitching:
                    cv2.circle(frame, (x, y), radius + 4, _C_RED, 1, lineType=cv2.LINE_AA)

            # --- Iris rings ---
            if len(lm) > 478:
                for iris_idx in (_R_IRIS, _L_IRIS):
                    ix, iy = px(lm[iris_idx].x, lm[iris_idx].y)
                    cv2.circle(frame, (ix, iy), 5, _C_BLUE, 1, lineType=cv2.LINE_AA)

            # --- AU labels near their anatomical anchor points ---
            active_aus = bsv.get("active_aus", [])
            au_anchors = {
                "AU4":    (lm[107].x, lm[107].y - 0.04),  # above right brow
                "AU12":   (lm[61].x - 0.04, lm[61].y + 0.03),  # left of lip corner
                "AU25/26":(lm[14].x, lm[14].y + 0.04),  # below lower lip
                "AU46":   (lm[159].x, lm[159].y - 0.04),  # above right eye
            }
            for au, (ax, ay) in au_anchors.items():
                if au in active_aus:
                    cx, cy = px(ax, ay)
                    cv2.putText(frame, au, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.38, _C_AMBER, 1, cv2.LINE_AA)

            # --- Brow-gap annotation line ---
            ry1 = px(lm[160].x, lm[160].y)
            ry2 = px(lm[_R_BROW_INNER].x, lm[_R_BROW_INNER].y)
            cv2.line(frame, ry1, ry2, _C_AMBER, 1, cv2.LINE_AA)

        # === Top status bar ===
        bar_h = 28
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), _C_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        face_label = "SOMATIC ACTIVE" if lm is not None else "NO FACE DETECTED"
        face_color = _C_TEAL if lm is not None else _C_RED
        cv2.putText(frame, face_label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, face_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "MINDSCAPE  SOMATIC", (w - 190, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 120), 1, cv2.LINE_AA)

        # === Bottom HUD strip ===
        hud_h = 72
        y0 = h - hud_h
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, y0), (w, h), _C_DARK, -1)
        cv2.addWeighted(overlay2, 0.82, frame, 0.18, 0, frame)

        def _hud_text(label, val, x, y):
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (130, 130, 150), 1, cv2.LINE_AA)
            cv2.putText(frame, str(val), (x, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, _C_WHITE, 1, cv2.LINE_AA)

        col_w = w // 5
        valence = bsv.get("facial_valence", 0.0)
        arousal = bsv.get("facial_arousal", 0.0)
        blink   = bsv.get("blink_rate_per_min", 0.0)
        gaze    = bsv.get("gaze_stability", 1.0)
        flat    = bsv.get("flat_affect_score", 0.0)

        _hud_text("VALENCE",  f"{valence:+.2f}",       8,           y0 + 20)
        _hud_text("AGITATION", f"{arousal:.2f}",        col_w + 4,   y0 + 20)
        _hud_text("BLINK/MIN", f"{blink:.0f}",          col_w * 2 + 4, y0 + 20)
        _hud_text("GAZE STAB", f"{gaze:.2f}",           col_w * 3 + 4, y0 + 20)
        _hud_text("FLAT AFCT", f"{flat:.2f}",           col_w * 4 + 4, y0 + 20)

        # Separator line
        cv2.line(frame, (0, y0), (w, y0), (50, 50, 60), 1)

        # Twitch zone pills in bottom strip
        zones = bsv.get("twitch_zones", [])
        if zones:
            pill_x = 8
            for z in zones[:4]:
                label = z.replace("_", " ").upper()
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(frame, (pill_x - 2, y0 + 46), (pill_x + tw + 6, y0 + 64),
                              (40, 20, 80), -1)
                cv2.putText(frame, label, (pill_x + 2, y0 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, _C_RED, 1, cv2.LINE_AA)
                pill_x += tw + 14

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return bytes(buf) if ok else None

    def _aggregate(self):
        ear_list = list(self._ear_buf)
        geom_list = list(self._geom_buf)
        if len(ear_list) < 10:
            return

        blink_rate = compute_blink_rate(ear_list)

        # Micro-movement analysis across all facial zones
        all_twitches, all_vel = [], []
        for zone, buf in self._zone_bufs.items():
            hist = list(buf)
            if len(hist) < 15:
                continue
            twitches, vel = extract_micro_movements(hist)
            for t in twitches:
                t["zone"] = zone
            all_twitches.extend(twitches)
            all_vel.extend(vel.tolist())

        # Psychomotor agitation: RMS of micro-movement velocity, scaled to [0,1]
        agitation = min(1.0, max(0.0, float(np.mean(all_vel)) * 300.0)) if all_vel else 0.0

        if geom_list:
            recent = geom_list[-30:]

            # Flat affect: low variance in expressivity markers sustained over time
            lip_std = float(np.std([g["lip_width"] for g in recent]))
            brow_std = float(np.std([g["brow_gap"] for g in recent]))
            flat_affect = max(0.0, 1.0 - (lip_std + brow_std) * 80.0)

            # Facial valence heuristic: wider lip (smile) → positive; tighter brow → negative
            avg_lip = float(np.mean([g["lip_width"] for g in recent]))
            avg_brow = float(np.mean([g["brow_gap"] for g in recent]))
            facial_valence = float(np.clip(
                (avg_lip - 0.35) * 4.0 - (0.04 - avg_brow) * 8.0,
                -1.0, 1.0
            ))

            # Gaze stability: high variance = darting gaze (hypervigilance, PTSD)
            offsets = [g["gaze_offset"] for g in recent]
            gaze_stability = max(0.0, 1.0 - float(np.std(offsets)) * 40.0)
        else:
            flat_affect = 0.5
            facial_valence = 0.0
            gaze_stability = 1.0

        # Action Unit detection from last frame geometry
        active_aus = []
        if geom_list:
            last = geom_list[-1]
            if last["blink"]:
                active_aus.append("AU46")           # Wink/blink
            if last["brow_gap"] < 0.03:
                active_aus.append("AU4")            # Brow lowering (distress)
            if last["lip_width"] > 0.40:
                active_aus.append("AU12")           # Lip corner pull (smile)
            if last["lip_open"] > 0.02:
                active_aus.append("AU25/26")        # Lips part (speaking, surprise)

        twitch_zones = list(set(t["zone"] for t in all_twitches[-10:]))

        with self._lock:
            self.latest = {
                "facial_valence": round(facial_valence, 3),
                "facial_arousal": round(agitation, 3),
                "blink_rate_per_min": round(blink_rate, 1),
                "gaze_stability": round(gaze_stability, 3),
                "flat_affect_score": round(flat_affect, 3),
                "psychomotor_agitation_score": round(agitation, 3),
                "micro_twitch_events": all_twitches[-10:],
                "active_aus": active_aus,
                "twitch_zones": twitch_zones,
                "face_detected": self.face_detected,
            }
