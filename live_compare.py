# python .\live_compare.py --master .\captures\master.jpg --ref .\reference.jpg --basler

import argparse, math, time
import cv2, numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from pypylon import pylon

# -------------------- thresholds --------------------
THRESH = {
    "sharp_warn": -10.0, "sharp_crit": -20.0,       # % drop vs MASTER
    "mean_warn":  10.0,  "mean_crit":  15.0,        # brightness levels (0..255)
    "std_warn":  -10.0,  "std_crit":   -20.0,       # % contrast change (neg is worse)
    "clip_warn":   1.0,  "clip_crit":   2.0,        # +percentage points
    "off_warn":  0.015,  "off_crit":   0.030,       # norm diag
    "roll_warn": 1.0,    "roll_crit":   2.0,        # degrees
    "zoom_warn": 0.020,  "zoom_crit":  0.035,       # area fraction
}

# -------------------- reference prep --------------------
ref_gray = None
ref_h = ref_w = 0
ref_keypoints = ref_descriptors = None
ref_templates = []
reference_image_loaded = False

def load_reference_image(ref_path: str) -> bool:
    global ref_gray, ref_h, ref_w, ref_keypoints, ref_descriptors, ref_templates, reference_image_loaded
    ref_image = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref_image is None:
        print(f"[ERR] Cannot load reference '{ref_path}'"); return False
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = ref_gray.shape
    orb = cv2.ORB_create(nfeatures=1000)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
    ref_templates = []
    for scale in [0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.6]:
        sw, sh = int(ref_w*scale), int(ref_h*scale)
        if 10 < sw < 3000 and 10 < sh < 3000:
            ref_templates.append((cv2.resize(ref_gray, (sw, sh)), scale))
    reference_image_loaded = True
    print(f"[REF] {ref_path}  size={ref_w}x{ref_h}  features={len(ref_keypoints) if ref_keypoints else 0}")
    return True

# -------------------- scoring vs reference --------------------
def compare_rect(candidate_gray: np.ndarray) -> float:
    if not reference_image_loaded or candidate_gray.size == 0:
        return 0.0
    best_template = 0.0
    for templ,_ in ref_templates:
        if candidate_gray.shape[0] >= templ.shape[0] and candidate_gray.shape[1] >= templ.shape[1]:
            res = cv2.matchTemplate(candidate_gray, templ, cv2.TM_CCOEFF_NORMED)
            if res.size: best_template = max(best_template, float(np.max(res)))
    if candidate_gray.shape[0] > 20 and candidate_gray.shape[1] > 20:
        resized_ref = cv2.resize(ref_gray, (candidate_gray.shape[1], candidate_gray.shape[0]))
        res = cv2.matchTemplate(candidate_gray, resized_ref, cv2.TM_CCOEFF_NORMED)
        if res.size: best_template = max(best_template, float(res[0,0]))
    # features
    feature_score = 0.0
    orb = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(candidate_gray, None)
    if ref_descriptors is not None and desc is not None and len(kp) > 3:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ref_descriptors, desc)
        if matches:
            good = [m for m in matches if m.distance < 60]
            feature_score = len(good) / max(1, len(ref_keypoints))
    # histogram
    ref_hist = cv2.calcHist([ref_gray],[0],None,[256],[0,256])
    cand_hist= cv2.calcHist([candidate_gray],[0],None,[256],[0,256])
    hist_score = cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_CORREL)
    hist_score = max(0.0, float(hist_score))
    return 0.6*best_template + 0.3*feature_score + 0.1*hist_score

# -------------------- rectangle search --------------------
def find_best_rectangle(img: np.ndarray) -> Optional[Dict[str, Any]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    e1 = cv2.Canny(blur, 20, 60)
    e2 = cv2.Canny(blur, 50,150)
    e3 = cv2.Canny(blur, 80,200)
    edges = cv2.bitwise_or(cv2.bitwise_or(e1,e2), e3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
    contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    H,W = gray.shape; img_area = H*W
    best, best_score = None, -1.0
    for cnt in contours:
        per = cv2.arcLength(cnt, True)
        if per < 16: continue
        for eps in [0.02,0.03,0.04,0.05]:
            approx = cv2.approxPolyDP(cnt, eps*per, True)
            if len(approx) != 4: continue
            pts = approx.reshape(4,2)
            x,y,w,h = cv2.boundingRect(pts)
            if w < 12 or h < 12: continue
            contour_area = cv2.contourArea(approx)
            bbox_area = w*h
            area_ratio = contour_area / max(1,bbox_area)
            aspect = max(w,h) / max(1, min(w,h))
            if area_ratio < 0.3 or aspect > 5: continue
            # angles (loose)
            def ang(a,b,c):
                ba,bc = a-b, c-b
                cos = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
                return np.degrees(np.arccos(np.clip(cos,-1,1)))
            if not all(45 <= ang(pts[(i-1)%4], pts[i], pts[(i+1)%4]) <= 135 for i in range(4)):
                continue
            cand = gray[y:y+h, x:x+w]
            sim = compare_rect(cand)
            rect = cv2.minAreaRect(approx)
            (cx,cy), (rw,rh), a = rect
            if rw >= rh: a += 90.0
            center_cost = math.hypot(cx - W/2, cy - H/2) / math.hypot(W,H)
            size_cost = 0.0 if (0.02*img_area) < (w*h) < (0.6*img_area) else 0.5
            final = sim - 0.4*center_cost - size_cost
            if final > best_score:
                best_score = final
                best = {"rect": rect, "center": (float(cx),float(cy)),
                        "size": (float(rw),float(rh)), "angle": float(a)}
            break
    return best

# -------------------- metrics --------------------
def tenengrad(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1,0,3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0,1,3)
    return float(np.mean(gx**2 + gy**2))

def exposure(gray):
    mean = float(np.mean(gray))
    std  = float(np.std(gray))
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    total = gray.size
    clip = float(hist[:2].sum() + hist[254:].sum())/total
    return mean, std, clip

def analyze_np(img: np.ndarray) -> Dict[str, Any]:
    H,W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = find_best_rectangle(img)
    if det:
        cx,cy = det["center"]; rw,rh = det["size"]; ang = det["angle"]
        area = (rw*rh)/(W*H)
        offset = math.hypot(cx - W/2, cy - H/2) / math.hypot(W,H)
    else:
        cx = cy = ang = area = offset = float("nan")
    mean,std,clip = exposure(gray)
    sharp = tenengrad(gray)
    return dict(image=img, gray=gray, det=det, center=(cx,cy), roll=ang,
                area=area, offset=offset, mean=mean, std=std, clip=clip, sharp=sharp, size=(W,H))

# -------------------- visuals --------------------
def overlay(img: np.ndarray, info: Dict[str,Any]) -> np.ndarray:
    out = img.copy()
    H,W = out.shape[:2]
    cv2.circle(out, (W//2, H//2), 6, (255,0,0), -1)       # blue = image center
    if info["det"] is not None:
        box = cv2.boxPoints(info["det"]["rect"]).astype(int)
        cv2.drawContours(out, [box], 0, (0,255,0), 2)
        cx,cy = map(int, info["center"])
        cv2.circle(out, (cx,cy), 6, (0,0,255), -1)        # red  = rect center
    return out

def severity(value, warn, crit, invert=False):
    if np.isnan(value): return "CRIT", (66,66,255) # red in BGR
    if invert:
        level = abs(value)
        if level >= crit: return "CRIT", (66,66,255)
        if level >= warn: return "WARN", (32,176,255)   # orange
        return "OK", (107,191,32)                      # green
    else:
        if value <= crit: return "CRIT", (66,66,255)
        if value <= warn: return "WARN", (32,176,255)
        return "OK", (107,191,32)

def draw_text_panel(canvas: np.ndarray, M: Dict[str,Any], T: Dict[str,Any]) -> np.ndarray:
    # -------- deltas (TARGET - MASTER) --------
    sharp_pct  = 100.0*(T["sharp"]/M["sharp"] - 1.0) if M["sharp"] else float('nan')
    mean_delta = T["mean"]   - M["mean"]
    std_pct    = 100.0*(T["std"]/M["std"] - 1.0)     if M["std"]   else float('nan')
    clip_pp    = (T["clip"]  - M["clip"]) * 100.0
    off_delta  = T["offset"] - M["offset"]
    roll_delta = T["roll"]   - M["roll"]
    zoom_delta = T["area"]   - M["area"]

    # -------- severities --------
    sev_sharp, col_sharp = severity(sharp_pct, THRESH["sharp_warn"], THRESH["sharp_crit"], invert=False)

    s_mean, _ = severity(mean_delta, THRESH["mean_warn"], THRESH["mean_crit"], invert=True)
    s_std,  _ = severity(std_pct,     THRESH["std_warn"],  THRESH["std_crit"],  invert=False)
    s_clip, _ = severity(clip_pp,     THRESH["clip_warn"], THRESH["clip_crit"], invert=True)
    sev_map   = {"OK":1, "WARN":2, "CRIT":3}
    lighting_lvl = max(sev_map[s_mean], sev_map[s_std], sev_map[s_clip])
    sev_light    = {1:"OK", 2:"WARN", 3:"CRIT"}[lighting_lvl]
    col_light    = {1:(107,191,32), 2:(32,176,255), 3:(66,66,255)}[lighting_lvl]

    sev_pos_off,  _ = severity(off_delta,  THRESH["off_warn"],  THRESH["off_crit"],  invert=True)
    sev_pos_roll, _ = severity(roll_delta, THRESH["roll_warn"], THRESH["roll_crit"], invert=True)
    pos_lvl  = max(sev_map[sev_pos_off], sev_map[sev_pos_roll])
    sev_pos  = {1:"OK", 2:"WARN", 3:"CRIT"}[pos_lvl]
    col_pos  = {1:(107,191,32), 2:(32,176,255), 3:(66,66,255)}[pos_lvl]

    sev_zoom, col_zoom = severity(zoom_delta, THRESH["zoom_warn"], THRESH["zoom_crit"], invert=True)

    # -------- panel drawing (bigger, spaced, readable) --------
    margin_top = 28
    line_gap   = 36      # vertical spacing between lines
    lines      = 4
    panel_h    = margin_top + lines*line_gap + 32  # +space for NUDGE
    panel = np.zeros((panel_h, canvas.shape[1], 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # dark background

    y = margin_top
    def put(text, color):
        nonlocal y
        cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        y += line_gap

    # one clean line per metric
    put(f"Sharpness: {sharp_pct:+.1f}%   [{sev_sharp}]", col_sharp)
    put(f"Lighting:  mean {mean_delta:+.1f}, contrast {std_pct:+.1f}%, clipping {clip_pp:+.2f} pp   [{sev_light}]", col_light)
    put(f"Position:  offset {off_delta:+.4f} (norm), roll {roll_delta:+.2f}°   [{sev_pos}]", col_pos)
    put(f"Zoom:      Δarea {zoom_delta:+.4f}   [{sev_zoom}]", col_zoom)

    # -------- NUDGE (bold, at bottom) --------
    nudge = []
    if M["det"] and T["det"]:
        Wt, Ht = T["size"]
        (cx1, cy1) = M["center"]; (cx2, cy2) = T["center"]
        dx = (cx1 - cx2)/Wt; dy = (cy1 - cy2)/Ht
        if abs(dx) > 0.002: nudge.append(("RIGHT" if dx>0 else "LEFT") + f" {abs(dx)*100:.1f}%")
        if abs(dy) > 0.002: nudge.append(("DOWN"  if dy>0 else "UP") + f" {abs(dy)*100:.1f}%")
        dtheta = (M["roll"] - T["roll"])
        if np.isfinite(dtheta) and abs(dtheta) > 0.1:
            nudge.append(("ROTATE CW" if dtheta < 0 else "ROTATE CCW") + f" {abs(dtheta):.2f}°")
        if np.isfinite(zoom_delta) and abs(zoom_delta) > 0.002:
            nudge.append(("MOVE FARTHER" if zoom_delta > 0 else "MOVE CLOSER") + f" |Δarea| {abs(zoom_delta):.3f}")

    any_crit = ("CRIT" in [sev_sharp, sev_light, sev_pos, sev_zoom])
    nudge_text = "NUDGE: " + (" | ".join(nudge) if nudge else "—")
    cv2.putText(panel, nudge_text, (20, panel_h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,0,255) if any_crit else (255,255,255), 3, cv2.LINE_AA)

    return panel


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

# -------------------- main loop --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="Path to MASTER image")
    ap.add_argument("--ref", default="reference.jpg", help="Path to reference template")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index (ignored with --basler)")
    ap.add_argument("--basler", action="store_true", help="Use Basler via pypylon if available")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    if not load_reference_image(args.ref):
        return

    # load MASTER and analyze once
    master_img = cv2.imread(args.master, cv2.IMREAD_COLOR)
    if master_img is None:
        print(f"[ERR] cannot read MASTER '{args.master}'"); return
    master_img = cv2.resize(master_img, (args.width, args.height), interpolation=cv2.INTER_AREA)
    M = analyze_np(master_img)

    # init camera (Basler or OpenCV)
    use_basler = False
    cap = None
    camera = converter = None
    if args.basler:
        try:
            from pypylon import pylon
            tlf = pylon.TlFactory.GetInstance()
            devices = tlf.EnumerateDevices()
            if not devices: raise RuntimeError("No Basler devices")
            camera = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
            camera.Open()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            use_basler = True
            print("[CAM] Basler connected")
        except Exception as e:
            print(f"[WARN] Basler not available ({e}); falling back to OpenCV")
    if not use_basler:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print("[ERR] Cannot open OpenCV camera"); return
        print(f"[CAM] OpenCV camera index {args.camera}")

    win = "Master vs Live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        # grab frame
        if use_basler:
            frame = get_frame_basler(camera, converter)
        else:
            frame = get_frame_opencv(cap)
        if frame is None: continue

        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        T = analyze_np(frame)

        left = overlay(M["image"], M)
        right = overlay(T["image"], T)

        # side-by-side
        H = max(left.shape[0], right.shape[0])
        canvas = np.zeros((H, left.shape[1]+right.shape[1], 3), dtype=np.uint8)
        canvas[:left.shape[0], :left.shape[1]] = left
        canvas[:right.shape[0], left.shape[1]:] = right

        panel = draw_text_panel(canvas, M, T)
        out = np.vstack([canvas, panel])

        cv2.putText(out, "MASTER", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(out, "LIVE", (left.shape[1]+20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"live_compare_{ts}.jpg", out)
            print(f"[SAVE] live_compare_{ts}.jpg")
        elif key == ord('m'):
            # Set current LIVE as new master (quick re-baseline)
            M = T
            print("[MASTER] re-baselined from live frame")

    # cleanup
    if use_basler and camera:
        camera.StopGrabbing(); camera.Close()
    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
