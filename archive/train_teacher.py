import argparse, os, glob, math, json, csv
import numpy as np, cv2

# ---- knobs (geometry) ----
W, H = 800, 400              # canonical size after warp
INNER_MARGIN = 14            # trim inside border
TAB_BAND_PX  = 50            # ignore top strip (blue tab zone)

def angle_err_90(a,b,c):
    ab, cb = a-b, c-b
    den = (np.linalg.norm(ab)*np.linalg.norm(cb)+1e-9)
    cos = np.clip(np.dot(ab,cb)/den, -1, 1)
    return abs(math.degrees(math.acos(cos))-90)

def order_corners(pts):
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)

def detect_outer_rect(gray):
    lab = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    Lc = cv2.createCLAHE(2.0,(8,8)).apply(L)
    normg = cv2.cvtColor(cv2.merge([Lc,A,B]), cv2.COLOR_LAB2BGR)
    normg = cv2.cvtColor(normg, cv2.COLOR_BGR2GRAY)

    bin_adapt = cv2.adaptiveThreshold(normg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,41,5)
    bin_adapt = cv2.morphologyEx(bin_adapt, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),1)
    cnts,_ = cv2.findContours(bin_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, score = None, -1
    Hh, Ww = gray.shape
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.01*Hh*Ww: continue
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx)!=4: continue
        p = approx.reshape(-1,2).astype(np.float32)
        q = order_corners(p)
        err = np.mean([angle_err_90(q[(i-1)%4], q[i], q[(i+1)%4]) for i in range(4)])
        if err>20: continue
        sc = (1-min(1,err/20))*math.log(1+area)
        if sc>score: score, best = sc, q
    return best

def warp_perspective(img, corners):
    src = order_corners(corners)
    dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, Hm, (W,H)), Hm

def build_part_mask():
    mask = np.zeros((H,W), np.uint8)
    cv2.rectangle(mask, (INNER_MARGIN, INNER_MARGIN+TAB_BAND_PX),
                  (W-INNER_MARGIN, H-INNER_MARGIN), 255, -1)
    return mask

def affine_to_params(M):
    # M: 2x3 mapping current -> canonical master
    a, b, tx = M[0,0], M[0,1], M[0,2]
    c, d, ty = M[1,0], M[1,1], M[1,2]
    # similarity assumption: [sR|t]; s = sqrt(a^2+c^2); theta = atan2(c,a)
    s = math.sqrt(a*a + c*c)
    theta = math.atan2(c, a)
    rho = math.log(max(1e-8, s))
    return tx, ty, theta, rho

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--currents_glob", default="captures/current_*.jpg")
    ap.add_argument("--out_csv", default="align/train_labels.csv")
    ap.add_argument("--config_json", default="align/train_config.json")
    args = ap.parse_args()

    m_bgr = cv2.imread(args.master, cv2.IMREAD_COLOR)
    if m_bgr is None: raise SystemExit("Failed to read master")
    m_gray = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2GRAY)
    quad = detect_outer_rect(m_gray)
    if quad is None: raise SystemExit("Master: outer rectangle not found")
    master_can, Hm = warp_perspective(m_bgr, quad)

    # ORB on masked master
    mask_part = build_part_mask()
    orb = cv2.ORB_create(nfeatures=3000)
    mkp, mdesc = orb.detectAndCompute(master_can, mask_part)
    if mdesc is None or len(mkp)<10: raise SystemExit("Not enough master features")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    paths = sorted(glob.glob(args.currents_glob))
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","dx","dy","theta","rho","W","H","inner_margin","tab_band_px"])
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None: continue
            # Features on current (unmasked; RANSAC will reject tab)
            ckp, cdesc = orb.detectAndCompute(img, None)
            if cdesc is None or len(ckp)<10: 
                print("[skip] few features:", p); continue
            # Match + ratio
            matches = bf.knnMatch(mdesc, cdesc, k=2)
            good = [m for m,n in matches if m.distance < 0.75*n.distance]
            if len(good) < 12: 
                print("[skip] few good matches:", p); continue
            src = np.float32([mkp[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # master_can
            dst = np.float32([ckp[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # current
            # Robust affine (partial2D ~ similarity+shear) mapping current -> canonical
            M, inl = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is None: 
                print("[skip] affine fail:", p); continue
            dx, dy, th, rho = affine_to_params(M)
            w.writerow([p, dx, dy, th, rho, W, H, INNER_MARGIN, TAB_BAND_PX])

    # Save config (so training/inference know geometry)
    with open(args.config_json, "w") as jf:
        json.dump({"W":W,"H":H,"inner_margin":INNER_MARGIN,"tab_band_px":TAB_BAND_PX,
                   "master_canonical_path":"align/master_canonical.jpg"}, jf, indent=2)
    cv2.imwrite("align/master_canonical.jpg", master_can)
    print("Wrote:", args.out_csv, "and align/master_canonical.jpg")

if __name__ == "__main__":
    main()
