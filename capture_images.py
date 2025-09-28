import os
import cv2
import time
import math
import argparse
import numpy as np
from datetime import datetime
from pypylon import pylon

def timestamp():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{int(now.microsecond/1000):03d}ms"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_jpeg(path, img_bgr, quality=95):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok = cv2.imwrite(path, img_bgr, params)
    if ok:
        print(f"[saved] {path}")
    else:
        print(f"[warn] failed to save {path}")

def main():
    ap = argparse.ArgumentParser(description="Basler image capture tool (pypylon + OpenCV).")
    ap.add_argument("--save-dir", default="captures", help="Directory to save images")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100)")
    args = ap.parse_args()

    ensure_dir(args.save_dir)

    # --- Setup Basler camera ---
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    if not devices:
        print("No Basler cameras found!")
        return

    camera = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    print("[info] Controls: m=save master | c=save current | q=quit")

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                frame = converter.Convert(grab).GetArray()
                frame_resized = cv2.resize(frame, (960, 720))
                cv2.imshow("Basler Capture", frame_resized)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    save_jpeg(os.path.join(args.save_dir, "master.jpg"), frame_resized, args.jpeg_quality)
                elif key == ord('c'):
                    fname = f"example_{timestamp()}.jpg"
                    save_jpeg(os.path.join(args.save_dir, fname), frame_resized, args.jpeg_quality)

            grab.Release()
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
