"""
ae_hud.py

Provides image-quality metrics, a simple AutoExposureController, and HUD drawing.
This module is intentionally small and focused so it can be imported by a runner.
"""
import time
import cv2
import numpy as np
from typing import Tuple

# Thresholds used to decide severity
THRESH = {
    "sharp_warn": -10.0, "sharp_crit": -20.0,
    "mean_warn":  10.0,  "mean_crit":  15.0,
    "std_warn":   -10.0, "std_crit":   -20.0,
    "clip_warn":   1.0,  "clip_crit":   2.0,
}

def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
    return float(np.mean(gx**2 + gy**2))

def exposure(gray: np.ndarray):
    mean = float(np.mean(gray))
    std  = float(np.std(gray))
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    total = gray.size
    clip = float(hist[:2].sum() + hist[254:].sum()) / total
    return mean, std, clip

def analyze(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean, std, clip = exposure(gray)
    sharp = tenengrad(gray)
    return {"image": img_bgr, "mean": mean, "std": std, "clip": clip, "sharp": sharp}

def sev_label(value, warn, crit, invert=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "CRIT", (66,66,255)
    if invert:
        level = abs(value)
        if level >= crit: return "CRIT", (66,66,255)
        if level >= warn: return "WARN", (32,176,255)
        return "OK", (107,191,32)
    else:
        if value <= crit: return "CRIT", (66,66,255)
        if value <= warn: return "WARN", (32,176,255)
        return "OK", (107,191,32)

def compute_metrics(master, live):
    sharp_pct  = 100.0*(live['sharp']/master['sharp'] - 1.0) if master['sharp'] > 1e-9 else float('nan')
    mean_delta = live['mean'] - master['mean']
    std_pct    = 100.0*(live['std']/master['std'] - 1.0) if master['std'] > 1e-9 else float('nan')
    clip_pp    = (live['clip'] - master['clip']) * 100.0

    sev_sharp, col_sharp = sev_label(sharp_pct, THRESH['sharp_warn'], THRESH['sharp_crit'])
    s_mean,_ = sev_label(mean_delta, THRESH['mean_warn'], THRESH['mean_crit'], invert=True)
    s_std,_  = sev_label(std_pct,    THRESH['std_warn'],  THRESH['std_crit'])
    s_clip,_ = sev_label(clip_pp,    THRESH['clip_warn'], THRESH['clip_crit'], invert=True)
    sev_map = {"OK":1,"WARN":2,"CRIT":3}
    lighting_lvl = max(sev_map[s_mean], sev_map[s_std], sev_map[s_clip])
    sev_light = {1:"OK",2:"WARN",3:"CRIT"}[lighting_lvl]
    col_light = {1:(107,191,32),2:(32,176,255),3:(66,66,255)}[lighting_lvl]

    suggestions = []
    if sev_sharp in ("WARN","CRIT"):
        suggestions.append("FOCUS CAMERA")
    if sev_light in ("WARN","CRIT"):
        if mean_delta < -THRESH['mean_warn']:
            suggestions.append("Increase brightness/exposure")
        elif mean_delta > THRESH['mean_warn']:
            suggestions.append("Decrease brightness/exposure")
        if std_pct < THRESH['std_warn']:
            suggestions.append("Increase contrast")
        if clip_pp > THRESH['clip_warn']:
            suggestions.append("Reduce exposure to avoid clipping")

    return {
        "sharp_pct": sharp_pct,
        "mean_delta": mean_delta,
        "std_pct": std_pct,
        "clip_pp": clip_pp,
        "sev_sharp": sev_sharp, "col_sharp": col_sharp,
        "sev_light": sev_light, "col_light": col_light,
        "suggestion": " | ".join(suggestions) if suggestions else "—"
    }

def make_tinted_master(master_bgr: np.ndarray, tint_color=(0,215,255), dim: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2HSV)
    mask_dark = cv2.inRange(hsv, (0,0,0), (180,120,170))
    mask_blue = cv2.inRange(hsv, (95,60,60), (130,255,255))
    mask = cv2.bitwise_or(mask_dark, mask_blue)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    tint_layer = np.full_like(master_bgr, tint_color, dtype=np.uint8)
    ghost = cv2.addWeighted(master_bgr, dim, tint_layer, 1.0-dim, 0)
    ghost_masked = cv2.bitwise_and(ghost, ghost, mask=mask)
    return ghost_masked, (mask>0).astype(np.uint8)

def draw_metrics_hud(frame: np.ndarray, M: dict, T: dict):
    info = compute_metrics(M, T)
    H, W = frame.shape[:2]
    hud_h = 140
    y0 = H - hud_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (W, H), (30,30,30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)
    y = y0 + 36
    def put(text, color):
        nonlocal y
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        y += 36
    put(f"Sharpness: {info['sharp_pct']:+.1f}%   [{info['sev_sharp']}]", info['col_sharp'])
    put(f"Lighting:  mean {info['mean_delta']:+.1f}, contrast {info['std_pct']:+.1f}%, clipping {info['clip_pp']:+.2f} pp   [{info['sev_light']}]", info['col_light'])
    cv2.putText(frame, f"SUGGESTION: {info['suggestion']}", (16, H-12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255) if info['suggestion'] != "—" else (255,255,255), 3, cv2.LINE_AA)


class AutoExposureController:
    def __init__(self, *, camera=None, cap=None, use_basler=False,
                 min_exposure=-100000.0, max_exposure=100000.0,
                 step_pct=0.1, min_interval_s=1.0):
        self.camera = camera
        self.cap = cap
        self.use_basler = use_basler
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.step_pct = step_pct
        self.min_interval_s = min_interval_s
        self.last_change = 0.0

    def _get_exposure_basler(self):
        try:
            if not self.camera: return None
            if hasattr(self.camera, 'ExposureTime'):
                return float(self.camera.ExposureTime.GetValue())
            if hasattr(self.camera, 'ExposureTimeAbs'):
                return float(self.camera.ExposureTimeAbs.GetValue())
        except Exception:
            return None

    def _set_exposure_basler(self, val):
        try:
            if not self.camera: return False
            if hasattr(self.camera, 'ExposureTime'):
                self.camera.ExposureTime.SetValue(int(val)); return True
            if hasattr(self.camera, 'ExposureTimeAbs'):
                self.camera.ExposureTimeAbs.SetValue(float(val)); return True
        except Exception:
            return False
        return False

    def _get_exposure_opencv(self):
        try:
            if not self.cap: return None
            val = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            return float(val) if val is not None and val != -1 else None
        except Exception:
            return None

    def _set_exposure_opencv(self, val):
        try:
            if not self.cap: return False
            return self.cap.set(cv2.CAP_PROP_EXPOSURE, float(val))
        except Exception:
            return False

    def get_exposure(self):
        return self._get_exposure_basler() if self.use_basler else self._get_exposure_opencv()

    def set_exposure(self, val):
        val = float(np.clip(val, self.min_exposure, self.max_exposure))
        if self.use_basler:
            return self._set_exposure_basler(val)
        return self._set_exposure_opencv(val)

    def step_exposure(self, direction=1):
        now = time.time()
        if now - self.last_change < self.min_interval_s:
            return False
        cur = self.get_exposure()
        if cur is None:
            return False
        if cur == 0:
            new = cur + direction * 1.0
        else:
            new = cur * (1.0 + direction * self.step_pct)
        new = float(np.clip(new, self.min_exposure, self.max_exposure))
        ok = self.set_exposure(new)
        if ok:
            self.last_change = now
        return ok