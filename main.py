from pypylon import pylon
import cv2
import numpy as np

# Global variables for point selection
selected_points = []
ref_gray = None
ref_h, ref_w = 0, 0
setup_complete = False

def mouse_callback(event, x, y, flags, param):
    global selected_points, setup_complete
    
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append((x, y))
        print(f"Point {len(selected_points)} selected: ({x}, {y})")
        
        if len(selected_points) == 4:
            print("4 points selected! Press 'c' to confirm and start tracking, or 'r' to reset points")

def extract_reference_region(frame, points):
    """Extract the region defined by 4 points and create reference template"""
    global ref_gray, ref_h, ref_w
    
    # Convert points to numpy array
    pts = np.array(points, dtype=np.float32)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(pts)
    
    # Extract the region
    ref_region = frame[y:y+h, x:x+w]
    ref_gray = cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY) if len(ref_region.shape) == 3 else ref_region
    ref_h, ref_w = ref_gray.shape
    
    print(f"Reference template created: {ref_w}x{ref_h}")

def compare_rect(ref_gray, candidate_gray):
    if ref_gray is None or candidate_gray.size == 0:
        return 0
    candidate_resized = cv2.resize(candidate_gray, (ref_w, ref_h))
    res = cv2.matchTemplate(candidate_resized, ref_gray, cv2.TM_CCOEFF_NORMED)
    return res[0][0]

def is_rect_in_polygon(rect_points, polygon_points):
    """Check if rectangle is within the polygon defined by selected points"""
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # Check if all 4 corners of the rectangle are inside the polygon
    for point in rect_points:
        result = cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False)
        if result < 0:  # Point is outside polygon
            return False
    return True

MATCH_THRESHOLD = 0.7
DETECTION_SKIP = 10  # run detection every 10 frames

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

frame_counter = 0
best_pts = None
best_score = -1

# Create window and set mouse callback for point selection
cv2.namedWindow("Rectangle Detection & Matching", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Rectangle Detection & Matching", mouse_callback)

print("=== SETUP MODE ===")
print("Click 4 points on the image to define the reference rectangle")
print("Press 'c' to confirm points and start tracking")
print("Press 'r' to reset points")
print("Press 'q' to quit")

while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab_result.GrabSucceeded():
        frame = converter.Convert(grab_result).GetArray()
        frame_resized = cv2.resize(frame, (640, 480))
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Setup mode - allow point selection
        if not setup_complete:
            display_frame = frame_resized.copy()
            
            # Draw filled polygon if we have 4 points
            if len(selected_points) == 4:
                pts = np.array(selected_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # Draw filled polygon with transparency
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0, 100))
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                # Draw polygon outline
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            
            # Draw selected points
            for i, point in enumerate(selected_points):
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw lines between points if we have more than 1
            if len(selected_points) > 1 and len(selected_points) < 4:
                for i in range(len(selected_points) - 1):
                    cv2.line(display_frame, selected_points[i], selected_points[i+1], (255, 0, 0), 2)
            
            # Instructions
            cv2.putText(display_frame, f"Points: {len(selected_points)}/4", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if len(selected_points) == 4:
                cv2.putText(display_frame, "Press 'c' to confirm, 'r' to reset", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Rectangle Detection & Matching", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(selected_points) == 4:
                # Confirm points and create reference template
                extract_reference_region(frame_resized, selected_points)
                setup_complete = True
                print("=== TRACKING MODE ===")
                print("Reference template created. Now tracking rectangles...")
                print("Press 'q' to quit")
            elif key == ord('r'):
                # Reset points
                selected_points = []
                print("Points reset. Click 4 new points.")
                
        else:
            # Tracking mode - original detection logic
            frame_counter += 1

            # --- Only run detection every DETECTION_SKIP frames ---
            if frame_counter % DETECTION_SKIP == 0:
                blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
                edges = cv2.Canny(blur, 50, 150)

                contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                best_score = -1
                best_pts = None

                for cnt in contours:
                    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                    if len(approx) == 4:
                        pts = approx.reshape(4, 2)
                        def angle(pt1, pt2, pt0):
                            dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
                            dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
                            ang = np.arccos((dx1*dx2 + dy1*dy2) /
                                            (np.sqrt(dx1*dx1+dy1*dy1) * np.sqrt(dx2*dx2+dy2*dy2)+1e-10))
                            return np.degrees(ang)
                        angles = [angle(pts[(i-1)%4], pts[(i+1)%4], pts[i]) for i in range(4)]
                        if all(70 <= a <= 110 for a in angles):
                            # Check if the detected rectangle is within our defined polygon region
                            if is_rect_in_polygon(pts, selected_points):
                                x, y, w, h = cv2.boundingRect(pts)
                                candidate = gray_frame[y:y+h, x:x+w]
                                try:
                                    score = compare_rect(ref_gray, candidate)
                                except (cv2.error, ZeroDivisionError):
                                    continue
                                if score > best_score:
                                    best_score = score
                                    best_pts = pts

            # --- Draw only if above threshold ---
            display_frame = frame_resized.copy()
            
            # Always show the defined polygon region
            if len(selected_points) == 4:
                pts = np.array(selected_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # Draw filled polygon with transparency
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts], (255, 255, 0, 50))  # Yellow region
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
                # Draw polygon outline
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)
            
            if best_pts is not None and best_score >= MATCH_THRESHOLD:
                cv2.polylines(display_frame, [best_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for x, y in best_pts:
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(display_frame, f"Score: {best_score:.2f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(display_frame, "DETECTED IN REGION", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Rectangle Detection & Matching", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
