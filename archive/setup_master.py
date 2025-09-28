import argparse, json, os, cv2, numpy as np

pts = []  # clicked points

def order_corners(p):
    p = np.array(p, np.float32)
    s = p.sum(axis=1); d = np.diff(p, axis=1).ravel()
    tl = p[np.argmin(s)]; br = p[np.argmax(s)]
    tr = p[np.argmin(d)]; bl = p[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)

def on_mouse(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])

def main():
    ap = argparse.ArgumentParser(description="Pick master rectangle corners + tune mask margins.")
    ap.add_argument("--master", required=True, help="Path to master image")
    ap.add_argument("--W", type=int, default=920)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--out_dir", default="align")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.master)
    if img is None:
        raise SystemExit("Failed to read master image")

    disp = img.copy()
    cv2.namedWindow("click 4 corners (outer border). Press R to reset, S to save")
    cv2.setMouseCallback("click 4 corners (outer border). Press R to reset, S to save", on_mouse)

    inner_margin = 14
    tab_band_px  = 50

    while True:
        vis = disp.copy()
        # draw clicked points
        for i, (x,y) in enumerate(pts):
            cv2.circle(vis, (x,y), 6, (0,0,255), -1)
            cv2.putText(vis, f"{i+1}", (x+8,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("click 4 corners (outer border). Press R to reset, S to save", vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            pts.clear()
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        if key == ord('s'):
            if len(pts) != 4:
                print("Please click exactly 4 corners before saving.")
                continue
            break

    # Warp to canonical W×H
    src = order_corners(pts)
    dst = np.float32([[0,0],[args.W-1,0],[args.W-1,args.H-1],[0,args.H-1]])
    Hm  = cv2.getPerspectiveTransform(src, dst)
    master_can = cv2.warpPerspective(img, Hm, (args.W, args.H))

    # Mask tuning preview window
    def nothing(_): pass
    cv2.namedWindow("tune mask (Tab: top band, Inner: border trim). Press S to finalize")
    cv2.createTrackbar("inner_margin", "tune mask (Tab: top band, Inner: border trim). Press S to finalize", inner_margin, 80, nothing)
    cv2.createTrackbar("tab_band_px", "tune mask (Tab: top band, Inner: border trim). Press S to finalize", tab_band_px, 150, nothing)

    while True:
        inner_margin = cv2.getTrackbarPos("inner_margin", "tune mask (Tab: top band, Inner: border trim). Press S to finalize")
        tab_band_px  = cv2.getTrackbarPos("tab_band_px", "tune mask (Tab: top band, Inner: border trim). Press S to finalize")
        overlay = master_can.copy()
        x0 = inner_margin
        y0 = inner_margin + tab_band_px
        x1 = args.W - inner_margin
        y1 = args.H - inner_margin
        x0 = max(0, min(args.W-1, x0))
        y0 = max(0, min(args.H-1, y0))
        x1 = max(1, min(args.W, x1))
        y1 = max(1, min(args.H, y1))
        cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,255,0), 2)
        view = cv2.addWeighted(master_can, 0.7, overlay, 0.3, 0)
        cv2.imshow("tune mask (Tab: top band, Inner: border trim). Press S to finalize", view)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            break
        if k == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # Save outputs
    cv2.imwrite(os.path.join(args.out_dir, "master_canonical.jpg"), master_can)
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump({
            "W": args.W, "H": args.H,
            "inner_margin": int(inner_margin),
            "tab_band_px": int(tab_band_px),
            "master_canonical_path": os.path.join(args.out_dir, "master_canonical.jpg"),
            "master_corners": src.tolist()
        }, f, indent=2)
    with open(os.path.join(args.out_dir, "master_corners.json"), "w") as f:
        json.dump({"corners_src_image_space": src.tolist()}, f, indent=2)

    print("Saved:")
    print(" -", os.path.join(args.out_dir, "master_canonical.jpg"))
    print(" -", os.path.join(args.out_dir, "train_config.json"))
    print(" -", os.path.join(args.out_dir, "master_corners.json"))

if __name__ == "__main__":
    main()
