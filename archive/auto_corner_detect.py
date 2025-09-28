import cv2, numpy as np, argparse, json, os

def main():
    ap = argparse.ArgumentParser(description="Auto-find outer 4 corners with Harris (simple).")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_json", default="align/auto_corners.json")
    ap.add_argument("--out_debug", default="align/auto_corners_debug.png")
    ap.add_argument("--thr", type=float, default=0.01, help="threshold on Harris response")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read {args.image}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)

    # Harris corner response
    harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    # take strong responses
    pts = np.argwhere(harris > args.thr * harris.max())
    if len(pts) < 4:
        raise SystemExit("not enough Harris corners found, try lowering --thr")

    # choose outermost corners
    pts = np.array(pts)[:, [1,0]]  # (x,y)
    left  = pts[np.argmin(pts[:,0])]
    right = pts[np.argmax(pts[:,0])]
    top   = pts[np.argmin(pts[:,1])]
    bot   = pts[np.argmax(pts[:,1])]
    corners = [left.tolist(), top.tolist(), right.tolist(), bot.tolist()]

    # save JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json,"w") as f:
        json.dump({"corners":[[int(x),int(y)] for x,y in corners]}, f, indent=2)
    print("saved:", args.out_json)

    # draw debug
    vis = img.copy()
    for i,(x,y) in enumerate(corners):
        cv2.circle(vis,(int(x),int(y)),6,(0,0,255),-1)
        cv2.putText(vis,str(i+1),(int(x)+8,int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.imwrite(args.out_debug, vis)
    print("saved:", args.out_debug)

if __name__=="__main__":
    main()