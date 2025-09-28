# align_guard_ui.py
# MASTER vs TARGET visual comparator with clear GRADES (as deltas) and a bold NUDGE.
# Uses your reference-based rectangle detection (multi-scale template + ORB).
#
# pip install opencv-python numpy pillow

import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageTk

# ===================== TUNABLE THRESHOLDS (set for "will break model") =====================
THRESH = {
    # Sharpness (Tenengrad) — % drop vs MASTER
    "sharp_warn": -10.0,   # below this % is yellow
    "sharp_crit": -20.0,   # below this % is RED

    # Lighting — absolute mean shift & contrast % change; extra clipping pp (percentage points)
    "mean_warn":  10.0,    # levels (0..255)
    "mean_crit":  15.0,
    "std_warn":  -10.0,    # % change in contrast; negative is worse
    "std_crit":  -20.0,
    "clip_warn":   1.0,    # +pp
    "clip_crit":   2.0,

    # Position — normalized center offset delta (image diagonal) & roll delta (deg)
    "off_warn":  0.015,
    "off_crit":  0.030,
    "roll_warn": 1.0,      # degrees
    "roll_crit": 2.0,

    # Zoom — area ratio delta (fraction of frame)
    "zoom_warn": 0.020,
    "zoom_crit": 0.035,
}

# ===================== Reference image prep =====================
ref_gray = None
ref_h = ref_w = 0
ref_keypoints = None
ref_descriptors = None
ref_templates = []
reference_image_loaded = False

def load_reference_image(ref_path: str) -> bool:
    global ref_gray, ref_h, ref_w, ref_keypoints, ref_descriptors, ref_templates, reference_image_loaded
    ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        messagebox.showerror("Reference", f"Could not load {ref_path}")
        return False
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = ref_gray.shape
    orb = cv2.ORB_create(nfeatures=1000)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
    ref_templates = []
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]:
        sw, sh = int(ref_w*scale), int(ref_h*scale)
        if 10 < sw < 3000 and 10 < sh < 3000:
            ref_templates.append((cv2.resize(ref_gray, (sw, sh)), scale))
    reference_image_loaded = True
    return True

# ===================== Detection (your approach) =====================
def compare_rect(candidate_gray: np.ndarray) -> float:
    if not reference_image_loaded or candidate_gray.size == 0:
        return 0.0

    best_template = 0.0
    for templ, _ in ref_templates:
        if candidate_gray.shape[0] >= templ.shape[0] and candidate_gray.shape[1] >= templ.shape[1]:
            res = cv2.matchTemplate(candidate_gray, templ, cv2.TM_CCOEFF_NORMED)
            if res.size:
                best_template = max(best_template, float(np.max(res)))

    if candidate_gray.shape[0] > 20 and candidate_gray.shape[1] > 20:
        resized_ref = cv2.resize(ref_gray, (candidate_gray.shape[1], candidate_gray.shape[0]))
        res = cv2.matchTemplate(candidate_gray, resized_ref, cv2.TM_CCOEFF_NORMED)
        if res.size:
            best_template = max(best_template, float(res[0, 0]))

    feature_score = 0.0
    orb = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(candidate_gray, None)
    if ref_descriptors is not None and desc is not None and len(kp) > 3:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ref_descriptors, desc)
        if matches:
            good = [m for m in matches if m.distance < 60]
            feature_score = len(good) / max(1, len(ref_keypoints))

    ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0,256])
    cand_hist = cv2.calcHist([candidate_gray], [0], None, [256], [0,256])
    hist_score = cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_CORREL)
    hist_score = max(0.0, float(hist_score))

    return 0.6*best_template + 0.3*feature_score + 0.1*hist_score

def find_best_rectangle(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    e1 = cv2.Canny(blur, 20, 60)
    e2 = cv2.Canny(blur, 50, 150)
    e3 = cv2.Canny(blur, 80, 200)
    edges = cv2.bitwise_or(cv2.bitwise_or(e1, e2), e3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    img_area = H*W
    best, best_score = None, -1.0

    for cnt in contours:
        per = cv2.arcLength(cnt, True)
        if per < 16: continue
        for eps in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps*per, True)
            if len(approx) != 4: continue
            pts = approx.reshape(4,2)
            x,y,w,h = cv2.boundingRect(pts)
            if w < 12 or h < 12: continue
            contour_area = cv2.contourArea(approx)
            bbox_area = w*h
            area_ratio = contour_area / max(1, bbox_area)
            aspect = max(w,h) / max(1, min(w,h))
            if area_ratio < 0.3 or aspect > 5: continue

            def angle(a,b,c):
                ba, bc = a-b, c-b
                cosang = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
                return np.degrees(np.arccos(np.clip(cosang,-1,1)))
            angs = [angle(pts[(i-1)%4], pts[i], pts[(i+1)%4]) for i in range(4)]
            if not all(45 <= a <= 135 for a in angs): continue

            cand = gray[y:y+h, x:x+w]
            sim = compare_rect(cand)

            rect = cv2.minAreaRect(approx)
            (cx,cy), (rw,rh), ang = rect
            if rw >= rh: ang += 90.0

            center_cost = math.hypot(cx - W/2, cy - H/2) / math.hypot(W, H)
            size_cost = 0.0 if (0.02*img_area) < (w*h) < (0.6*img_area) else 0.5
            final = sim - 0.4*center_cost - size_cost

            if final > best_score:
                best_score = final
                best = {"rect": rect, "center": (float(cx), float(cy)),
                        "size": (float(rw), float(rh)), "angle": float(ang)}
            break
    return best

# ===================== Metrics =====================
def tenengrad(gray): 
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
    return float(np.mean(gx**2 + gy**2))

def exposure(gray):
    mean = float(np.mean(gray))
    std  = float(np.std(gray))
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    total = gray.size
    clip = float(hist[:2].sum() + hist[254:].sum())/total
    return mean, std, clip

def analyze(path: str) -> Dict[str, Any]:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = find_best_rectangle(img)
    if det:
        cx, cy = det["center"]; rw, rh = det["size"]; ang = det["angle"]
        area_ratio = (rw*rh)/(W*H)
        offset = math.hypot(cx - W/2, cy - H/2) / math.hypot(W, H)
    else:
        cx = cy = ang = area_ratio = offset = float("nan")
    mean, std, clip = exposure(gray)
    sharp = tenengrad(gray)
    return dict(image=img, gray=gray, det=det, center=(cx,cy), roll=ang,
                area=area_ratio, offset=offset, mean=mean, std=std, clip=clip,
                sharp=sharp, size=(W,H))

# ===================== UI helpers =====================
def severity(value, warn, crit, invert=False):
    """
    Returns ('OK'|'WARN'|'CRIT', color) given thresholds.
    invert=True means higher is worse (we compare value vs +limits);
    default means more negative is worse (e.g., % drop).
    """
    if np.isnan(value): return "CRIT", "#ff4242"
    if invert:
        if abs(value) >= crit:  return "CRIT", "#ff4242"
        if abs(value) >= warn:  return "WARN", "#ffb020"
        return "OK", "#20bf6b"
    else:
        if value <= crit:  return "CRIT", "#ff4242"
        if value <= warn:  return "WARN", "#ffb020"
        return "OK", "#20bf6b"

def overlay(img: np.ndarray, det: Optional[Dict[str,Any]]) -> np.ndarray:
    out = img.copy()
    H,W = out.shape[:2]
    cv2.circle(out, (W//2, H//2), 6, (255,0,0), -1)   # blue: image center
    if det:
        box = cv2.boxPoints(det["rect"]).astype(int)
        cv2.drawContours(out, [box], 0, (0,255,0), 2)
        cx,cy = map(int, det["center"])
        cv2.circle(out, (cx,cy), 6, (0,0,255), -1)    # red: rect center
    return out

# ===================== Tkinter App =====================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Alignment Guard — MASTER vs TARGET")
        self.geometry("1200x740")

        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        self.ref_path = tk.StringVar(value="reference.jpg")
        ttk.Label(top, text="Reference:").pack(side="left")
        ttk.Entry(top, textvariable=self.ref_path, width=36).pack(side="left", padx=4)
        ttk.Button(top, text="Load REF", command=self.load_ref).pack(side="left", padx=(2,8))
        ttk.Button(top, text="Load MASTER", command=self.load_master).pack(side="left", padx=4)
        ttk.Button(top, text="Load TARGET", command=self.load_target).pack(side="left", padx=4)
        ttk.Button(top, text="Compare", command=self.compare).pack(side="left", padx=10)

        mid = ttk.Frame(self, padding=8); mid.pack(fill="both", expand=True)
        self.cL = tk.Canvas(mid, bg="#111"); self.cR = tk.Canvas(mid, bg="#111")
        self.cL.pack(side="left", fill="both", expand=True, padx=(0,4))
        self.cR.pack(side="left", fill="both", expand=True, padx=(4,0))
        self.tkL = self.tkR = None

        bottom = ttk.Frame(self, padding=10); bottom.pack(fill="x")
        font_big = ("Segoe UI", 12, "bold")
        self.lbl_sharp = ttk.Label(bottom, text="Sharpness Δ: —", font=font_big)
        self.lbl_light = ttk.Label(bottom, text="Contrast/Lighting Δ: —", font=font_big)
        self.lbl_pos   = ttk.Label(bottom, text="Position Δ: —", font=font_big)
        self.lbl_zoom  = ttk.Label(bottom, text="Zoom Δ: —", font=font_big)
        self.lbl_nudge = ttk.Label(bottom, text="NUDGE: —",  font=("Segoe UI", 13, "bold"))
        self.lbl_sharp.pack(anchor="w"); self.lbl_light.pack(anchor="w")
        self.lbl_pos.pack(anchor="w");   self.lbl_zoom.pack(anchor="w")
        ttk.Separator(bottom, orient="horizontal").pack(fill="x", pady=6)
        self.lbl_nudge.pack(anchor="w")

        self.M = None; self.T = None

    def load_ref(self):
        if load_reference_image(self.ref_path.get()):
            messagebox.showinfo("Reference", "Reference loaded.")
        else:
            messagebox.showerror("Reference", "Failed to load reference image.")

    def load_master(self):
        p = filedialog.askopenfilename(title="Select MASTER",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if p:
            self.M = analyze(p); self._draw(self.cL, self.M, side="L")

    def load_target(self):
        p = filedialog.askopenfilename(title="Select TARGET",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if p:
            self.T = analyze(p); self._draw(self.cR, self.T, side="R")

    def _draw(self, canvas, info, side="L"):
        canvas.delete("all")
        if not info: return
        img = overlay(info["image"], info["det"])
        cw, ch = canvas.winfo_width() or 580, canvas.winfo_height() or 480
        ih, iw = img.shape[:2]
        s = min(cw/iw, ch/ih); new = (max(1,int(iw*s)), max(1,int(ih*s)))
        rgb = cv2.cvtColor(cv2.resize(img, new, cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        if side=="L": self.tkL = tkimg
        else: self.tkR = tkimg
        canvas.create_image((cw-new[0])//2, (ch-new[1])//2, anchor="nw", image=tkimg)

    def compare(self):
        if self.M is None or self.T is None:
            messagebox.showinfo("Compare", "Load MASTER and TARGET first.")
            return

        M, T = self.M, self.T

        # ----- Sharpness (% change) -----
        sharp_pct = 100.0*(T["sharp"]/M["sharp"] - 1.0) if M["sharp"] else float('nan')
        sev, col = severity(sharp_pct, THRESH["sharp_warn"], THRESH["sharp_crit"], invert=False)
        self.lbl_sharp.config(foreground=col,
            text=f"Sharpness Δ = {sharp_pct:+.1f}% vs MASTER   [{sev}]")

        # ----- Contrast/Lighting -----
        mean_delta = T["mean"] - M["mean"]
        std_pct = 100.0*(T["std"]/M["std"] - 1.0) if M["std"] else float('nan')
        clip_pp = (T["clip"] - M["clip"])*100.0
        # Overall severity = worst of three
        sev_mean,_ = severity(abs(mean_delta), THRESH["mean_warn"], THRESH["mean_crit"], invert=True)
        sev_std, _ = severity(std_pct, THRESH["std_warn"], THRESH["std_crit"], invert=False)
        sev_clip,_ = severity(abs(clip_pp), THRESH["clip_warn"], THRESH["clip_crit"], invert=True)
        sev_map = {"OK":1, "WARN":2, "CRIT":3}
        lighting_sev = max(sev_map[sev_mean], sev_map[sev_std], sev_map[sev_clip])
        lighting_col = {1:"#20bf6b", 2:"#ffb020", 3:"#ff4242"}[lighting_sev]
        sev_label = {1:"OK", 2:"WARN", 3:"CRIT"}[lighting_sev]
        self.lbl_light.config(foreground=lighting_col,
            text=f"Contrast/Lighting Δ = mean {mean_delta:+.1f} levels, contrast {std_pct:+.1f}%, clipping {clip_pp:+.2f} pp   [{sev_label}]")

        # ----- Position -----
        off_delta = T["offset"] - M["offset"]
        roll_delta = T["roll"] - M["roll"]
        sev_off, _  = severity(abs(off_delta), THRESH["off_warn"], THRESH["off_crit"], invert=True)
        sev_roll,_  = severity(abs(roll_delta), THRESH["roll_warn"], THRESH["roll_crit"], invert=True)
        pos_sev = max(sev_map[sev_off], sev_map[sev_roll])
        pos_col = {1:"#20bf6b", 2:"#ffb020", 3:"#ff4242"}[pos_sev]
        pos_label = {1:"OK", 2:"WARN", 3:"CRIT"}[pos_sev]
        self.lbl_pos.config(foreground=pos_col,
            text=f"Position Δ = offset {off_delta:+.4f} (norm), roll {roll_delta:+.2f}°   [{pos_label}]")

        # ----- Zoom -----
        zoom_delta = T["area"] - M["area"]
        sev_zoom,_ = severity(abs(zoom_delta), THRESH["zoom_warn"], THRESH["zoom_crit"], invert=True)
        self.lbl_zoom.config(foreground=sev_zoom=="CRIT" and "#ff4242" or (sev_zoom=="WARN" and "#ffb020" or "#20bf6b"),
            text=f"Zoom Δ = area ratio {zoom_delta:+.4f}   [{sev_zoom}]")

        # ----- NUDGE (clear operator guidance) -----
        nudge = []
        if self.T["det"] and self.M["det"]:
            Wt, Ht = self.T["size"]
            (cx1, cy1) = self.M["center"]; (cx2, cy2) = self.T["center"]
            dx = (cx1 - cx2)/Wt; dy = (cy1 - cy2)/Ht
            if abs(dx) > 0.002: nudge.append(("RIGHT" if dx>0 else "LEFT")+f" {abs(dx)*100:.1f}%")
            if abs(dy) > 0.002: nudge.append(("DOWN"  if dy>0 else "UP")+f" {abs(dy)*100:.1f}%")
            dθ = (self.M["roll"] - self.T["roll"])
            if np.isfinite(dθ) and abs(dθ) > 0.1: nudge.append(("ROTATE CW" if dθ<0 else "ROTATE CCW")+f" {abs(dθ):.2f}°")
            dz = zoom_delta
            if np.isfinite(dz) and abs(dz) > 0.002:
                nudge.append(("MOVE FARTHER" if dz>0 else "MOVE CLOSER")+f" |Δarea| {abs(dz):.3f}")
        self.lbl_nudge.config(foreground="#ffffff" if lighting_sev<3 and pos_sev<3 else "#ff4242",
                              background="#333333" if lighting_sev<3 and pos_sev<3 else "#330000",
                              text="NUDGE: " + (" | ".join(nudge) if nudge else "—"))

        # refresh overlays
        self._draw(self.cL, self.M, "L")
        self._draw(self.cR, self.T, "R")

# ===================== main =====================
if __name__ == "__main__":
    app = App()
    # Try to load default reference on startup (optional)
    try:
        load_reference_image("reference.jpg")
    except Exception:
        pass
    app.mainloop()
