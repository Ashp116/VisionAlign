# compare_ref_based.py
# Master vs Target image comparison using your reference-guided rectangle detection.
# - Uses multi-scale template matching + ORB feature matching against reference.jpg
# - Finds the "best" rectangle in each image and computes deltas (TARGET - MASTER)
# - Outputs: four grades + "nudges" and a side-by-side overlay image (compare_out.jpg)
#
# pip install opencv-python numpy

import argparse
import math
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

# ---------- Reference data ----------
ref_gray = None
ref_h = ref_w = 0
ref_keypoints = None
ref_descriptors = None
ref_templates = []
reference_image_loaded = False

def load_reference_image(ref_path: str) -> bool:
    """Load reference.jpg and prepare templates + ORB features."""
    global ref_gray, ref_h, ref_w, ref_keypoints, ref_descriptors, ref_templates, reference_image_loaded

    ref_image = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref_image is None:
        print(f"ERROR: Could not load '{ref_path}'")
        return False

    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = ref_gray.shape
    orb = cv2.ORB_create(nfeatures=1000)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)

    # Build multi-scale templates (tuned to common webcam frames)
    ref_templates = []
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]:
        sw, sh = int(ref_w * scale), int(ref_h * scale)
        if 10 < sw < 2000 and 10 < sh < 2000:
            ref_templates.append((cv2.resize(ref_gray, (sw, sh)), scale))

    reference_image_loaded = True
    print(f"[REF] loaded {ref_path} | size={ref_w}x{ref_h} | features={len(ref_keypoints) if ref_keypoints else 0} | templates={len(ref_templates)}")
    return True

# ---------- Scoring vs reference (your method condensed) ----------
def compare_rect(candidate_gray: np.ndarray) -> float:
    """Multi-method similarity score between candidate region and reference."""
    if ref_gray is None or candidate_gray.size == 0 or not reference_image_loaded:
        return 0.0

    best_template_score = 0.0
    # Method 1: multi-scale template matching
    for template, _ in ref_templates:
        if candidate_gray.shape[0] >= template.shape[0] and candidate_gray.shape[1] >= template.shape[1]:
            res = cv2.matchTemplate(candidate_gray, template, cv2.TM_CCOEFF_NORMED)
            if res.size:
                best_template_score = max(best_template_score, float(np.max(res)))

    # Method 2: direct resized match
    if candidate_gray.shape[0] > 20 and candidate_gray.shape[1] > 20:
        resized_ref = cv2.resize(ref_gray, (candidate_gray.shape[1], candidate_gray.shape[0]))
        res = cv2.matchTemplate(candidate_gray, resized_ref, cv2.TM_CCOEFF_NORMED)
        if res.size:
            best_template_score = max(best_template_score, float(res[0, 0]))

    # Method 3: ORB feature ratio
    feature_score = 0.0
    orb = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(candidate_gray, None)
    if ref_descriptors is not None and desc is not None and len(kp) > 3:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ref_descriptors, desc)
        if matches:
            good = [m for m in matches if m.distance < 60]
            feature_score = len(good) / max(1, len(ref_keypoints))

    # Method 4: histogram correlation
    ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
    cand_hist = cv2.calcHist([candidate_gray], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_CORREL)
    hist_score = max(0.0, float(hist_score))  # clamp negatives

    # Weighted blend (favor template/feature)
    return 0.6 * best_template_score + 0.3 * feature_score + 0.1 * hist_score

# ---------- Rectangle search in a static image ----------
def find_best_rectangle(img: np.ndarray) -> Optional[Dict[str, Any]]:
    """Search whole frame for 4-point rectangles; score each vs reference."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Multi-sensitivity edges, then combine
    e1 = cv2.Canny(blur, 20, 60)
    e2 = cv2.Canny(blur, 50, 150)
    e3 = cv2.Canny(blur, 80, 200)
    edges = cv2.bitwise_or(cv2.bitwise_or(e1, e2), e3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    img_area = H * W

    best = None
    best_score = -1.0

    for cnt in contours:
        per = cv2.arcLength(cnt, True)
        if per < 16:
            continue
        # try several approximation levels
        for eps in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps * per, True)
            if len(approx) != 4:
                continue
            pts = approx.reshape(4, 2)
            x, y, w, h = cv2.boundingRect(pts)
            if w < 12 or h < 12:
                continue
            contour_area = cv2.contourArea(approx)
            bbox_area = w * h
            area_ratio = contour_area / max(1, bbox_area)
            aspect_ratio = max(w, h) / max(1, min(w, h))
            if area_ratio < 0.3 or aspect_ratio > 5:
                continue

            # Angle sanity (loose)
            def angle(a, b, c):
                ba, bc = a - b, c - b
                cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
                return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            angs = [angle(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]) for i in range(4)]
            if not all(45 <= a <= 135 for a in angs):
                continue

            # Candidate region & similarity score
            cand = gray[y:y + h, x:x + w]
            score = compare_rect(cand)

            # Rect geometry
            rect = cv2.minAreaRect(approx)
            (cx, cy), (rw, rh), ang = rect
            if rw >= rh:
                ang += 90.0

            # prefer not-too-small/not-huge & fairly centered
            center_cost = math.hypot(cx - W / 2, cy - H / 2) / math.hypot(W, H)
            size_cost = 0.0 if (0.02 * img_area) < (w * h) < (0.6 * img_area) else 0.5
            final_score = score - 0.4 * center_cost - size_cost

            if final_score > best_score:
                best_score = final_score
                best = {
                    "bbox": (x, y, w, h),
                    "center": (float(cx), float(cy)),
                    "size": (float(rw), float(rh)),
                    "angle": float(ang),
                    "rect": rect,
                    "score": float(score),
                }
            break  # move to next contour after a valid 4-pt approx
    return best

# ---------- Metrics ----------
def tenengrad_focus(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))

def exposure(gray: np.ndarray) -> Tuple[float, float, float]:
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = gray.size
    clip_low = float(hist[:2].sum()) / total
    clip_high = float(hist[254:].sum()) / total
    return mean, std, (clip_low + clip_high)

def analyze_image(path: str) -> Dict[str, Any]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = find_best_rectangle(img)

    if det:
        cx, cy = det["center"]
        rw, rh = det["size"]
        angle = det["angle"]
        area_ratio = (rw * rh) / (W * H)
        offset = math.hypot(cx - W / 2, cy - H / 2) / math.hypot(W, H)
    else:
        cx = cy = angle = area_ratio = offset = float("nan")

    mean, std, clip = exposure(gray)
    sharp = tenengrad_focus(gray)

    return dict(
        image=img,
        gray=gray,
        det=det,
        center=(cx, cy),
        roll=angle,
        area_ratio=area_ratio,
        offset=offset,
        sharp=sharp,
        mean=mean,
        std=std,
        clip=clip,
        size=(W, H)
    )

# ---------- Visualization & grading ----------
def overlay(img: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    # centers
    cv2.circle(out, (W // 2, H // 2), 6, (255, 0, 0), -1)  # blue: image center
    if info["det"] is not None:
        box = cv2.boxPoints(info["det"]["rect"]).astype(int)
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)
        cx, cy = map(int, info["center"])
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)      # red: rect center
    return out

def grade_and_nudge(M: Dict[str, Any], T: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    # Sharpness grade
    s1, s2 = M["sharp"], T["sharp"]
    sharp_grade = "MASTER" if s1 >= s2 else "TARGET"

    # Lighting grade (same scoring used previously)
    def light_score(m):
        exp_err = abs(m["mean"] - 127.5) / 127.5
        return (m["std"] / 64.0) - 0.6 * exp_err - 0.8 * m["clip"]
    light_grade = "MASTER" if light_score(M) >= light_score(T) else "TARGET"

    # Position grade
    def pos_score(m):
        if not (np.isfinite(m["offset"]) and np.isfinite(m["roll"])):
            return -1e9
        return -(2.0 * m["offset"] + 0.02 * abs(m["roll"]))
    pos_grade = "MASTER" if pos_score(M) >= pos_score(T) else "TARGET"

    # Zoom grade (closer to MASTER area wins)
    a1, a2 = M["area_ratio"], T["area_ratio"]
    zoom_grade = "MASTER" if (np.isfinite(a1) and np.isfinite(a2) and abs(a2 - a1) >= 0) else "TARGET"

    # Nudge
    hint = "—"
    if T["det"] is not None and M["det"] is not None:
        Wt, Ht = T["size"]
        (cx1, cy1) = M["center"]
        (cx2, cy2) = T["center"]
        dx = (cx1 - cx2) / Wt
        dy = (cy1 - cy2) / Ht
        moves = []
        if abs(dx) > 0.002:
            moves.append(("right" if dx > 0 else "left") + f" {abs(dx)*100:.1f}%")
        if abs(dy) > 0.002:
            moves.append(("down" if dy > 0 else "up") + f" {abs(dy)*100:.1f}%")
        if np.isfinite(M["roll"]) and np.isfinite(T["roll"]):
            dtheta = M["roll"] - T["roll"]
            if abs(dtheta) > 0.1:
                moves.append(("rotate CW" if dtheta < 0 else "rotate CCW") + f" {abs(dtheta):.2f}°")
        if np.isfinite(M["area_ratio"]) and np.isfinite(T["area_ratio"]):
            da = T["area_ratio"] - M["area_ratio"]
            if abs(da) > 0.002:
                # positive da → target larger (move farther), negative → move closer
                moves.append(("move farther" if da > 0 else "move closer") + f" ({abs(da)/max(1e-9,M['area_ratio'])*100:.1f}%)")
        if moves:
            hint = " | ".join(moves)

    grades = {
        "Sharpness grade": sharp_grade,
        "Contrast/lighting grade": light_grade,
        "Position grade": pos_grade,
        "Zoom grade": zoom_grade
    }
    return grades, hint

def make_side_by_side(M: Dict[str, Any], T: Dict[str, Any], grades: Dict[str, str], hint: str, out="compare_out.jpg"):
    left  = overlay(M["image"], M)
    right = overlay(T["image"], T)
    H = max(left.shape[0], right.shape[0])
    W = left.shape[1] + right.shape[1]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:left.shape[1]+right.shape[1]] = right

    # annotate
    def put(y, text, color=(255,255,255)):
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    y = 24
    put(y, "MASTER (left) vs TARGET (right)"); y += 26
    for k in ["Sharpness grade", "Contrast/lighting grade", "Position grade", "Zoom grade"]:
        put(y, f"{k}: {grades[k]}"); y += 24
    put(y, f"Nudge: {hint}", (0,255,255))

    cv2.imwrite(out, canvas)
    print(f"[OUT] {out} saved")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Master vs Target comparison using reference-based rectangle detection")
    ap.add_argument("--master", required=True, help="Path to master (image 1)")
    ap.add_argument("--target", required=True, help="Path to target (image 2)")
    ap.add_argument("--ref",    default="reference.jpg", help="Path to reference template (default: reference.jpg)")
    args = ap.parse_args()

    if not load_reference_image(args.ref):
        return

    M = analyze_image(args.master)
    T = analyze_image(args.target)
    grades, hint = grade_and_nudge(M, T)

    # Print the four metrics ONLY (as requested)
    print("=== GRADES (TARGET vs MASTER) ===")
    for k,v in grades.items():
        print(f"{k}: {v}")

    # Save visual compare
    make_side_by_side(M, T, grades, hint, out="compare_out.jpg")
    print("Done.")

if __name__ == "__main__":
    main()
