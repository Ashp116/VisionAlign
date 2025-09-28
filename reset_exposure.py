"""
reset_exposure.py

Attempt to restore camera exposure settings to defaults.

For Basler (pypylon): tries to set ExposureAuto to Continuous (if available) and reset ExposureTime to a safe default.
For OpenCV (DirectShow/VideoCapture): attempts to enable auto exposure via common property toggles and nudges exposure to a driver-default value.

Usage:
  python reset_exposure.py         # tries both Basler and OpenCV cameras
  python reset_exposure.py --opencv 0   # operate on OpenCV camera index 0 only
  python reset_exposure.py --basler      # operate on Basler device if available

Notes:
- Camera drivers vary widely: this script tries safe, best-effort operations and prints what it changes.
- Run as the same user that normally uses the cameras so drivers allow property changes.
"""

import argparse
import time
import cv2
import sys

try:
    from pypylon import pylon
except Exception:
    pylon = None

def reset_basler():
    if pylon is None:
        print('[Basler] pypylon not installed or unavailable')
        return False
    try:
        tlf = pylon.TlFactory.GetInstance()
        devs = tlf.EnumerateDevices()
        if not devs:
            print('[Basler] No devices found')
            return False
        cam = pylon.InstantCamera(tlf.CreateDevice(devs[0]))
        cam.Open()
        node_map = cam.GetNodeMap()
        print(f"[Basler] Connected to {cam.GetDeviceInfo().GetModelName()}")
        # Try set ExposureAuto to Continuous or Auto
        for name, val in [('ExposureAuto', 'Continuous'), ('ExposureAuto', 'Continuous'), ('ExposureAuto', 'On')]:
            try:
                if hasattr(cam, name):
                    node = getattr(cam, name)
                    # for pypylon enumerations, try to set by string value
                    if hasattr(node, 'GetValue') or hasattr(node, 'ToString'):
                        try:
                            node.SetValue(val)
                            print(f"[Basler] Set {name} -> {val}")
                            break
                        except Exception:
                            pass
            except Exception:
                pass
        # If ExposureTime is writable, try resetting to a mid-range value
        try:
            if hasattr(cam, 'ExposureTime') and hasattr(cam, 'ExposureTimeMin') and hasattr(cam, 'ExposureTimeMax'):
                minv = float(cam.ExposureTimeMin.GetValue())
                maxv = float(cam.ExposureTimeMax.GetValue())
                mid = int((minv + maxv) / 2)
                cam.ExposureTime.SetValue(mid)
                print(f"[Basler] Set ExposureTime -> {mid} (mid of {minv}-{maxv})")
        except Exception:
            try:
                if hasattr(cam, 'ExposureTimeAbs'):
                    # best-effort
                    cam.ExposureTimeAbs.SetValue(cam.ExposureTimeAbs.GetValue())
            except Exception:
                pass
        cam.Close()
        return True
    except Exception as e:
        print(f"[Basler] Error: {e}")
        return False

def reset_opencv(index=0):
    print(f"[OpenCV] Opening camera index {index}")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"[OpenCV] Cannot open camera {index}")
            return False
    # Try to enable auto exposure in a few common ways
    changed = False
    try:
        # Many Windows drivers use CAP_PROP_AUTO_EXPOSURE with 0.75/0.25 semantics
        cur = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"[OpenCV] Current CAP_PROP_AUTO_EXPOSURE = {cur}")
        # Try setting to auto: for DirectShow drivers 0.75 often means 'auto'
        for val in [0.75, 0.5, 1.0, 0]:
            ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(val))
            if ok:
                print(f"[OpenCV] Tried CAP_PROP_AUTO_EXPOSURE={val} -> ok")
                changed = True
                break
        # Try CAP_PROP_EXPOSURE = -6 (common default) then let auto take over
        if not changed:
            if cap.set(cv2.CAP_PROP_EXPOSURE, -6):
                print('[OpenCV] Set CAP_PROP_EXPOSURE to -6 (best-effort)')
                changed = True
    except Exception as e:
        print(f"[OpenCV] Error toggling properties: {e}")

    # Wait a bit for driver to apply changes
    time.sleep(0.5)
    # Read one frame to warm camera
    ret, frame = cap.read()
    if ret:
        print('[OpenCV] Grabbed a frame after reset')
    else:
        print('[OpenCV] Failed to grab frame after reset')

    cap.release()
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opencv', type=int, help='OpenCV camera index to reset (default: try 0)')
    ap.add_argument('--basler', action='store_true', help='Reset Basler camera if available')
    args = ap.parse_args()

    ok_any = False
    if args.basler:
        ok = reset_basler()
        ok_any = ok_any or ok
    if args.opencv is not None:
        ok = reset_opencv(args.opencv)
        ok_any = ok_any or ok
    if not args.basler and args.opencv is None:
        # try both: Basler then OpenCV index 0
        ok_any = reset_basler() or reset_opencv(0)

    if ok_any:
        print('Done (best-effort). If camera still dark restart the camera driver or reboot the PC.')
    else:
        print('No changes made. Please run with --basler or --opencv <index> or check camera drivers.')

if __name__ == '__main__':
    main()
