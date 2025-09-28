import cv2, numpy as np, math, argparse
from skimage.metrics import structural_similarity as ssim

def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.max(s)==s][0]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def detect_outer_rect(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ed = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000: continue
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) == 4:
            # check right-ish angles
            p = approx.reshape(-1,2)
            def ang(a,b,c):
                ab = a-b; cb = c-b
                cos = np.dot(ab,cb)/(np.linalg.norm(ab)*np.linalg.norm(cb)+1e-9)
                return abs(math.degrees(math.acos(np.clip(cos,-1,1))) - 90)
            err = sum(ang(p[(i-1)%4], p[i], p[(i+1)%4]) for i in range(4))/4
            if err < 12 and area > best_area:  # fairly rectangular + largest
                best = p; best_area = area
    return best

def warp_to_canonical(img, corners, W=600, H=300):
    src = order_corners(corners)
    dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Hm, (W,H), flags=cv2.INTER_LINEAR)
    return warped, Hm

def quality_metrics(gray):
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())
    brightness = float(gray.mean())
    return focus, contrast, brightness

def grid_cell_stats(gray, grid=(3,3), roi=None):
    # roi: (x0,y0,x1,y1) inside warped image; else whole
    h,w = gray.shape
    x0,y0,x1,y1 = roi if roi else (0,0,w,h)
    cells = []
    gw, gh = grid
    cw = (x1-x0)//gw; ch = (y1-y0)//gh
    for gy in range(gh):
        for gx in range(gw):
            xs = x0 + gx*cw; ys = y0 + gy*ch
            cell = gray[ys:ys+ch, xs:xs+cw]
            cells.append((cell.mean(), cell.std()))
    return cells  # list of (mean, std) per cell

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--inner_margin", type=int, default=12,
                    help="pixels trimmed inside border after warp")
    args = ap.parse_args()

    m = cv2.imread(args.master, cv2.IMREAD_COLOR)
    c = cv2.imread(args.current, cv2.IMREAD_COLOR)
    if m is None or c is None:
        raise SystemExit("Failed to load images")

    W,H = 600,300  # canonical size; tune to your template

    for name, img in [("master", m), ("current", c)]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = detect_outer_rect(gray)
        if rect is None:
            raise SystemExit(f"{name}: outer rectangle not found")
        warped, Hm = warp_to_canonical(img, rect, W, H)
        if name=="master":
            mw, mH = warped, Hm
        else:
            cw, cH = warped, Hm

    # Trim inside border so we analyze the PART only
    x0 = y0 = args.inner_margin
    x1 = W - args.inner_margin
    y1 = H - args.inner_margin
    m_part = cv2.cvtColor(mw[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    c_part = cv2.cvtColor(cw[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    # Alignment score after normalization
    s = ssim(m_part, c_part, data_range=255)

    # Quality metrics
    mf, mc, mb = quality_metrics(m_part)
    cf, cc, cb = quality_metrics(c_part)

    print("=== Alignment (normalized by outer rectangle) ===")
    print(f"SSIM (part ROI): {s*100:.1f}%")
    print("--- Quality ---")
    print(f"Focus: current {cf:.1f}  vs master {mf:.1f}  (Δ {cf/mf*100:.0f}% of master)")
    print(f"Contrast (std): current {cc:.1f} vs master {mc:.1f}")
    print(f"Brightness: current {cb:.1f}  vs master {mb:.1f}")

    # Optional: per-cell stats over the right/left blocks (split ROI in halves, then 3x3 each)
    h, w = c_part.shape
    left_roi  = (0, 0, w//2, h)
    right_roi = (w//2, 0, w, h)
    left_cells_cur  = grid_cell_stats(c_part, grid=(3,3), roi=left_roi)
    right_cells_cur = grid_cell_stats(c_part, grid=(3,3), roi=right_roi)
    # You can compare these to the master’s cell stats the same way for heatmaps or thresholds.

    # Save visual overlay (green edges after warp)
    vis = cv2.addWeighted(mw, 0.5, cw, 0.5, 0)
    cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.imwrite("normalized_overlay.png", vis)
    print("Saved: normalized_overlay.png")