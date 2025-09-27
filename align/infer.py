import argparse, os, time, math
import numpy as np
import cv2
import tensorflow as tf
from utils import warp_similarity, variance_of_laplacian, rms_contrast, brightness, ssim_score

def load_and_resize(path, size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def preprocess(img):
    return img.astype(np.float32) / 255.0

def guidance_text(dx, dy, theta, rho, px_per_mm):
    # Convert pixels -> mm
    mmx = dx / px_per_mm
    mmy = dy / px_per_mm
    deg = np.degrees(theta)
    scale = math.exp(rho)
    dist_hint = "closer" if scale > 1.0 else "farther" if scale < 1.0 else "same"
    return mmx, mmy, deg, scale, dist_hint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', required=True, help='Path to master/reference image')
    ap.add_argument('--current', default=None, help='Path to current image (if omitted and --webcam is set, will capture a frame)')
    ap.add_argument('--webcam', type=int, default=None, help='Webcam index (e.g., 0)')
    ap.add_argument('--model', required=True, help='Path to saved keras model')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--px_per_mm', type=float, default=4.0, help='Pixels per millimeter at master pose')
    ap.add_argument('--out', default='aligned_overlay.png')
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)

    master = load_and_resize(args.master, args.img_size)
    master_in = preprocess(master)[None, ...]

    if args.current is None and args.webcam is None:
        raise SystemExit("Provide --current image or --webcam index.")

    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise SystemExit("Failed to open webcam")
        time.sleep(0.3)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise SystemExit("Failed to capture frame from webcam")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current = cv2.resize(frame_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
    else:
        current = load_and_resize(args.current, args.img_size)

    current_in = preprocess(current)[None, ...]

    # Predict inverse similarity (current->master): dx, dy, theta(rad), rho
    pred = model.predict([master_in, current_in], verbose=0)[0]
    dx, dy, theta, rho = [float(x) for x in pred]

    # Build overlay: warp current with predicted inverse to align with master
    aligned = warp_similarity(current, dx, dy, theta, rho)
    # Compute SSIM alignment score
    gray_m = cv2.cvtColor(master, cv2.COLOR_RGB2GRAY)
    gray_a = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    score = ssim_score(gray_m, gray_a)

    # Quality metrics on current
    gray_c = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    focus = variance_of_laplacian(gray_c)
    contrast = rms_contrast(gray_c)
    bright = brightness(gray_c)

    # Prepare overlay visualization (blend 50/50)
    blend = cv2.addWeighted(master, 0.5, aligned, 0.5, 0)
    cv2.imwrite(args.out, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))

    mmx, mmy, deg, scale, dist_hint = guidance_text(dx, dy, theta, rho, args.px_per_mm)

    print("=== Alignment Prediction (current -> master) ===")
    print(f"dx = {dx:.2f} px  ({mmx:+.2f} mm)")
    print(f"dy = {dy:.2f} px  ({mmy:+.2f} mm)")
    print(f"theta = {theta:+.4f} rad  ({deg:+.2f} deg)")
    print(f"scale = e^{rho:+.4f} = {scale:.4f}  => you are {dist_hint}")
    print(f"SSIM (after warp) = {score*100:.1f}%")
    print("--- Quality (current) ---")
    print(f"Focus (Var Laplacian): {focus:.1f}")
    print(f"Contrast (std gray):   {contrast:.1f}")
    print(f"Brightness (mean):     {bright:.1f}")
    print(f"Saved overlay preview to: {args.out}")

if __name__ == '__main__':
    main()
