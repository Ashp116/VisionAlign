import cv2, json, argparse, numpy as np, os

POINTS = []  # in click order

def order_corners(pts):
    """Return corners ordered as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]

def on_mouse(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN and len(POINTS) < 4:
        POINTS.append([float(x), float(y)])

def main():
    ap = argparse.ArgumentParser(description="Click exactly 4 corners and press 's' to save.")
    ap.add_argument("--image", required=True, help="Path to the master image")
    ap.add_argument("--out", default="align/corners.json", help="Where to save the points JSON")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    win = "Click 4 corners (press r=reset, s=save, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse, img)

    while True:
        vis = img.copy()
        # draw current points
        for i, (x, y) in enumerate(POINTS):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"{i+1}", (int(x)+8, int(y)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            POINTS.clear()
        elif key == ord('s'):
            if len(POINTS) != 4:
                print("Please click exactly 4 points before saving.")
                continue

            # also save an ordered version (tl, tr, br, bl) for convenience
            ordered = order_corners(POINTS)

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({
                    "image_path": os.path.abspath(args.image),
                    "image_size": {"width": img.shape[1], "height": img.shape[0]},
                    "points_clicked": POINTS,          # in the order you clicked
                    "points_ordered_tl_tr_br_bl": ordered
                }, f, indent=2)
            print("Saved:", os.path.abspath(args.out))
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
