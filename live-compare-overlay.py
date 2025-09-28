# live_overlay_focus_contrast.py
# Single VIEW: MASTER overlaid on LIVE. Only Sharpness + Lighting/Contrast metrics with clear suggestions.
# MASTER stays at original scale. LIVE is resized to match MASTER for visual line-up.
#
# pip install opencv-python numpy
# For Basler: pip install pypylon

import argparse, time
import cv2, numpy as np
from typing import Tuple

# -------------------- optional Basler import --------------------
try:
    from pypylon import pylon
except ImportError:
    pylon = None

# -------------------- thresholds (tune if needed) --------------------
THRESH = {
    "sharp_warn": -10.0, "sharp_crit": -20.0,   # % drop vs MASTER
    "mean_warn":  10.0,  "mean_crit":  15.0,    # brightness delta (0..255)
    "std_warn":   -10.0, "std_crit":   -20.0,   # % contrast change (neg is worse)
    "clip_warn":   1.0,  "clip_crit":   2.0,    # +percentage points clipped at 0/255
}

# -------------------- metrics --------------------
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

# -------------------- severity + suggestions --------------------
def sev_label(value, warn, crit, invert=False):
    # Returns ("OK"|"WARN"|"CRIT", color BGR)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "CRIT", (66,66,255)
    if invert:
        level = abs(value)
        if level >= crit: return "CRIT", (66,66,255)     # red
        if level >= warn: return "WARN", (32,176,255)     # orange
        return "OK", (107,191,32)                         # green
    else:
        if value <= crit: return "CRIT", (66,66,255)
        if value <= warn: return "WARN", (32,176,255)
        return "OK", (107,191,32)

def compute_metrics(master, live):
    # deltas (LIVE - MASTER)
    sharp_pct  = 100.0*(live["sharp"]/master["sharp"] - 1.0) if master["sharp"] > 1e-9 else float('nan')
    mean_delta = live["mean"] - master["mean"]
    std_pct    = 100.0*(live["std"]/master["std"] - 1.0) if master["std"] > 1e-9 else float('nan')
    clip_pp    = (live["clip"] - master["clip"]) * 100.0

    sev_sharp, col_sharp = sev_label(sharp_pct, THRESH["sharp_warn"], THRESH["sharp_crit"])
    s_mean,_ = sev_label(mean_delta, THRESH["mean_warn"], THRESH["mean_crit"], invert=True)
    s_std,_  = sev_label(std_pct,    THRESH["std_warn"],  THRESH["std_crit"])
    s_clip,_ = sev_label(clip_pp,    THRESH["clip_warn"], THRESH["clip_crit"], invert=True)
    sev_map = {"OK":1,"WARN":2,"CRIT":3}
    lighting_lvl = max(sev_map[s_mean], sev_map[s_std], sev_map[s_clip])
    sev_light = {1:"OK",2:"WARN",3:"CRIT"}[lighting_lvl]
    col_light = {1:(107,191,32),2:(32,176,255),3:(66,66,255)}[lighting_lvl]

    # Suggestions (clear human directions)
    suggestions = []
    if sev_sharp in ("WARN","CRIT"):
        suggestions.append("FOCUS CAMERA")
    if sev_light in ("WARN","CRIT"):
        if mean_delta < -THRESH["mean_warn"]:
            suggestions.append("Increase brightness/exposure")
        elif mean_delta > THRESH["mean_warn"]:
            suggestions.append("Decrease brightness/exposure")
        if std_pct < THRESH["std_warn"]:
            suggestions.append("Increase contrast")
        if clip_pp > THRESH["clip_warn"]:
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

# -------------------- visuals --------------------
from typing import Tuple

def make_tinted_master(master_bgr: np.ndarray,
                       tint_color=(0, 215, 255),   # gold-ish tint (BGR)
                       dim: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a gold-tinted 'ghost' of MASTER and a 0/1 mask that keeps:
      - dark gray/black (now looser to include more blacks),
      - blue tape.
    Returns (ghost_masked_bgr, mask_0or1).
    """
    hsv = cv2.cvtColor(master_bgr, cv2.COLOR_BGR2HSV)

    # Looser dark mask → keeps more black/dark-gray
    # (raise V max and S max to include slightly lighter, low-sat grays)
    mask_dark = cv2.inRange(
        hsv,
        (0,   0,   0),       # H,  S,  V  mins
        (180, 120, 170)      # S<=120, V<=170  ← widened
    )

    # Blue mask (unchanged)
    mask_blue = cv2.inRange(
        hsv,
        (95,  60,  60),
        (130, 255, 255)
    )

    # Combine + clean
    mask = cv2.bitwise_or(mask_dark, mask_blue)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Gold-tinted ghost (kept from your favorite look)
    tint_layer = np.full_like(master_bgr, tint_color, dtype=np.uint8)
    ghost = cv2.addWeighted(master_bgr, dim, tint_layer, 1.0 - dim, 0)

    # Apply mask (only dark/blue regions survive)
    ghost_masked = cv2.bitwise_and(ghost, ghost, mask=mask)

    return ghost_masked, (mask > 0).astype(np.uint8)



def draw_metrics_hud(frame: np.ndarray, M: dict, T: dict):
    info = compute_metrics(M, T)

    H, W = frame.shape[:2]
    hud_h = 140
    y0 = H - hud_h

    # Semi-opaque background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (W, H), (30,30,30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

    # Text lines
    y = y0 + 36
    def put(text, color):
        nonlocal y
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        y += 36

    put(f"Sharpness: {info['sharp_pct']:+.1f}%   [{info['sev_sharp']}]", info["col_sharp"])
    put(f"Lighting:  mean {info['mean_delta']:+.1f}, contrast {info['std_pct']:+.1f}%, clipping {info['clip_pp']:+.2f} pp   [{info['sev_light']}]",
        info["col_light"])

    # Suggestion line
    cv2.putText(frame, f"SUGGESTION: {info['suggestion']}",
                (16, H-12), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,0,255) if info['suggestion'] != "—" else (255,255,255), 3, cv2.LINE_AA)

# -------------------- camera helpers --------------------
def get_frame_opencv(cap):
    ok, frame = cap.read()
    return frame if ok else None

def get_frame_basler(camera, converter):
    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab.GrabSucceeded():
        frame = converter.Convert(grab).GetArray()
        grab.Release()
        return frame
    grab.Release()
    return None

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Single-view overlay with Sharpness + Lighting metrics")
    ap.add_argument("--master", required=True, help="Path to MASTER image")
    ap.add_argument("--basler", action="store_true", help="Use Basler via pypylon")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--alpha",  type=float, default=0.40, help="Overlay alpha (0..1)")
    args = ap.parse_args()

    # Load MASTER (keep original scale)
    master = cv2.imread(args.master, cv2.IMREAD_COLOR)
    if master is None:
        print(f"[ERR] Cannot read MASTER '{args.master}'")
        return
    Hm, Wm = master.shape[:2]
    M = analyze(master)

    # Build tinted/trimmed MASTER and its mask (remove bright background)
    master_ghost, master_mask = make_tinted_master(master)


    # Camera init
    use_basler = False
    cap = None; camera = None; converter = None
    if args.basler and pylon is not None:
        try:
            tlf = pylon.TlFactory.GetInstance()
            devs = tlf.EnumerateDevices()
            if not devs: raise RuntimeError("No Basler devices")
            camera = pylon.InstantCamera(tlf.CreateDevice(devs[0]))
            camera.Open()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = cv2.pylab = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            use_basler = True
            print("[CAM] Basler connected")
        except Exception as e:
            print(f"[WARN] Basler init failed ({e}); falling back to OpenCV")

    if not use_basler:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("[ERR] Cannot open OpenCV camera")
            return
        print(f"[CAM] OpenCV index {args.camera}")

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    show_overlay = True
    win = "LIVE + MASTER overlay (Sharpness & Lighting)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        frame = get_frame_basler(camera, converter) if use_basler else get_frame_opencv(cap)
        if frame is None:
            continue

        # Resize LIVE to MASTER size (MASTER remains untouched)
        if frame.shape[:2] != (Hm, Wm):
            frame = cv2.resize(frame, (Wm, Hm), interpolation=cv2.INTER_AREA)

        # Compute metrics on full frame
        T = analyze(frame)

        # Compose single-view: LIVE base + optional MASTER ghost overlay (masked)
        out = frame.copy()
        if show_overlay:
            fg = cv2.addWeighted(out, 1.0, master_ghost, alpha, 0)
            out = np.where(master_mask[:, :, None] == 1, fg, out)

        # HUD: metrics and suggestion
        draw_metrics_hud(out, M, T)

        # Small help line
        cv2.putText(out, f"[o] overlay {'ON' if show_overlay else 'OFF'}   alpha={alpha:.2f}   [+/-] adjust   [s] save   [q] quit",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('o'):
            show_overlay = not show_overlay
        elif key in (ord('+'), ord('=')):
            alpha = float(np.clip(alpha + 0.05, 0.0, 1.0))
        elif key == ord('-'):
            alpha = float(np.clip(alpha - 0.05, 0.0, 1.0))
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"overlay_metrics_{ts}.jpg", out)
            print(f"[SAVE] overlay_metrics_{ts}.jpg")

    if use_basler and camera:
        camera.StopGrabbing(); camera.Close()
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
