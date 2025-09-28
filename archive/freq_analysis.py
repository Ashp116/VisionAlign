from pypylon import pylon
import cv2
import numpy as np

# =========================
# Utility functions
# =========================

def preprocess(img):
    """Convert to grayscale float32 and normalize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    return gray

def frequency_similarity(imgA, imgB):
    """Quick similarity using normalized cross correlation."""
    sim = cv2.matchTemplate(imgA, imgB, cv2.TM_CCOEFF_NORMED)
    return float(sim[0][0]) * 100.0  # percent

def translation_offset(imgA, imgB):
    """Find dx, dy using phase correlation."""
    (shift_y, shift_x), _ = cv2.phaseCorrelate(imgA, imgB)
    return float(shift_x), float(shift_y)

def focus_measure(img):
    """Variance of Laplacian as focus metric."""
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return float(lap.var())

# =========================
# Main analyzer
# =========================

def analyze(master, current):
    master_p = preprocess(master)
    current_p = preprocess(current)

    sim = frequency_similarity(master_p, current_p)
    dx, dy = translation_offset(master_p, current_p)
    focus = focus_measure(current_p)

    return {
        "similarity": sim,
        "dx": dx,
        "dy": dy,
        "focus": focus
    }

# =========================
# Camera setup
# =========================

def setup_camera():
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    if not devices:
        print("No Basler cameras found!")
        exit(1)

    cam = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
    cam.Open()
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_BGR8packed
    conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return cam, conv

# =========================
# Display overlay
# =========================

def draw_overlay(frame, results):
    display = frame.copy()

    # Similarity
    sim = results["similarity"]
    color = (0,255,0) if sim > 90 else (0,255,255) if sim > 75 else (0,0,255)
    cv2.putText(display, f"Similarity: {sim:.1f}%", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Offsets
    dx, dy = results["dx"], results["dy"]
    cv2.putText(display, f"dx={dx:.1f}px dy={dy:.1f}px", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Focus
    cv2.putText(display, f"Focus={results['focus']:.1f}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

    # Arrow for offset
    h,w = display.shape[:2]
    center = (w//2, h//2)
    tip = (int(center[0]+dx), int(center[1]+dy))
    cv2.arrowedLine(display, center, tip, (0,0,255), 3, tipLength=0.3)

    return display

# =========================
# Main loop
# =========================

def main():
    # Load reference
    master = cv2.imread("refrence.jpg")
    if master is None:
        raise FileNotFoundError("refrence.jpg not found!")

    cam, conv = setup_camera()
    cv2.namedWindow("Alignment", cv2.WINDOW_NORMAL)

    print("Press 'q' to quit")

    while cam.IsGrabbing():
        grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            frame = conv.Convert(grab).GetArray()
            frame = cv2.resize(frame, (960, 720))

            results = analyze(master, frame)
            display = draw_overlay(frame, results)

            cv2.imshow("Alignment", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        grab.Release()

    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
