from pypylon import pylon
import cv2
import numpy as np

# --- Load reference image ---
ref_img = cv2.imread("reference.jpg")
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
ref_h, ref_w = ref_gray.shape

# --- Function to compute template matching score ---
def compare_rect(ref_gray, candidate_gray):
    # Resize candidate to reference size
    candidate_resized = cv2.resize(candidate_gray, (ref_w, ref_h))
    # Use normalized correlation coefficient
    res = cv2.matchTemplate(candidate_resized, ref_gray, cv2.TM_CCOEFF_NORMED)
    return res[0][0]  # single value

# --- Setup Basler camera ---
tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
if not devices:
    print("No Basler cameras found!")
    exit(1)

camera = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# --- Real-time rectangle detection and matching ---
while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab_result.GrabSucceeded():
        frame = converter.Convert(grab_result).GetArray()
        frame_resized = cv2.resize(frame, (640, 480))
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Preprocess: blur + edge detection
        blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = -1
        best_pts = None

        # --- Check each 4-corner polygon ---
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                # Ensure roughly right angles
                pts = approx.reshape(4, 2)
                def angle(pt1, pt2, pt0):
                    dx1 = pt1[0] - pt0[0]
                    dy1 = pt1[1] - pt0[1]
                    dx2 = pt2[0] - pt0[0]
                    dy2 = pt2[1] - pt0[1]
                    ang = np.arccos((dx1*dx2 + dy1*dy2) / (np.sqrt(dx1*dx1+dy1*dy1) * np.sqrt(dx2*dx2+dy2*dy2)+1e-10))
                    return np.degrees(ang)
                angles = [angle(pts[(i-1)%4], pts[(i+1)%4], pts[i]) for i in range(4)]
                if all(80 <= a <= 100 for a in angles):
                    # Extract bounding rect for comparison
                    x, y, w, h = cv2.boundingRect(pts)
                    candidate = gray_frame[y:y+h, x:x+w]
                    try:
                        score = compare_rect(ref_gray, candidate)
                    except cv2.error:
                        continue
                    if score > best_score:
                        best_score = score
                        best_pts = pts

        # --- Draw best-matching rectangle ---
        if best_pts is not None:
            cv2.polylines(frame_resized, [best_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            for x, y in best_pts:
                cv2.circle(frame_resized, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame_resized, f"Score: {best_score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Rectangle Detection & Matching", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
