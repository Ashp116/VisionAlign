from pypylon import pylon
import cv2
import numpy as np

# Load reference image
print("Loading reference.jpg...")
ref_image = cv2.imread("reference.jpg")
if ref_image is None:
    print("ERROR: Could not load reference.jpg!")
    exit(1)

ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
ref_h, ref_w = ref_gray.shape
print(f"Reference loaded: {ref_w}x{ref_h}")

# Create multiple scale templates
templates = []
for scale in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    w = int(ref_w * scale)
    h = int(ref_h * scale)
    if 15 <= w <= 400 and 15 <= h <= 300:  # Reasonable size limits
        scaled = cv2.resize(ref_gray, (w, h))
        templates.append((scaled, scale))

print(f"Created {len(templates)} template scales")

# Initialize camera
tlf = pylon.TlFactory.GetInstance()
devices = tlf.EnumerateDevices()
if not devices:
    print("No cameras found!")
    exit(1)

camera = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Detection parameters
detection_threshold = 0.6  # Higher threshold to prevent false positives
show_all_detections = False
setup_complete = False
selected_points = []

def mouse_callback(event, x, y, flags, param):
    global selected_points, setup_complete
    
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4 and not setup_complete:
        selected_points.append((x, y))
        print(f"Point {len(selected_points)} selected: ({x}, {y})")
        
        if len(selected_points) == 4:
            print("4 points selected! Press 'c' to confirm region and start detection")

def point_in_polygon(point, polygon_points):
    """Check if a point is inside the polygon defined by 4 points"""
    if len(polygon_points) != 4:
        return True  # If no region defined, consider all points valid
    
    polygon = np.array(polygon_points, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0  # Point is inside or on boundary

def is_detection_in_region(detection_rect, region_points):
    """Check if detection rectangle overlaps significantly with the defined region"""
    if len(region_points) != 4:
        return True  # No region defined, accept all detections
    
    # Get center and corners of detection
    x, y, w, h = detection_rect
    center = (x + w//2, y + h//2)
    corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    
    # Check if center is in region
    center_in = point_in_polygon(center, region_points)
    
    # Check how many corners are in region
    corners_in = sum(1 for corner in corners if point_in_polygon(corner, region_points))
    
    # Consider it "in region" if center is in OR at least 2 corners are in
    return center_in or corners_in >= 2

print("\n=== Reference Detection with Region Setup ===")
print("STEP 1: Click 4 points to define the detection region")
print("STEP 2: Press 'c' to confirm region and start detection")
print("During detection:")
print("  - Press 'a' to toggle showing all detections")
print("  - Press '+'/'-' to adjust threshold")
print("  - Press 's' to save current frame")
print("  - Press 'r' to reset region")
print("  - Press 'q' to quit")

frame_count = 0
best_match = None
best_score = 0
all_matches = []

# Set up mouse callback
cv2.namedWindow("Reference Detection with Region", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Reference Detection with Region", mouse_callback)

while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab_result.GrabSucceeded():
        frame = converter.Convert(grab_result).GetArray()
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1
        
        # Only process detection if setup is complete
        if setup_complete and frame_count % 2 == 0:
            best_match = None
            best_score = 0
            all_matches = []
            
            # Try each template scale
            for template, scale in templates:
                if template.shape[0] <= gray.shape[0] and template.shape[1] <= gray.shape[1]:
                    # Template matching
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    # Only consider matches above a minimum threshold
                    if max_val > 0.4:  # Minimum threshold to reduce false positives
                        match_info = {
                            'score': max_val,
                            'location': max_loc,
                            'size': (template.shape[1], template.shape[0]),
                            'scale': scale
                        }
                        
                        # Check if detection is in the defined region
                        detection_rect = (max_loc[0], max_loc[1], template.shape[1], template.shape[0])
                        if is_detection_in_region(detection_rect, selected_points):
                            all_matches.append(match_info)
                            
                            if max_val > best_score:
                                best_score = max_val
                                best_match = match_info
        
        # Draw results
        display_frame = frame.copy()
        
        # Draw region setup or detection results
        if not setup_complete:
            # SETUP MODE: Draw selected points and region
            for i, point in enumerate(selected_points):
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw lines between points
            if len(selected_points) > 1:
                for i in range(len(selected_points)):
                    start_point = selected_points[i]
                    end_point = selected_points[(i+1) % len(selected_points)] if len(selected_points) == 4 else selected_points[i+1] if i < len(selected_points)-1 else selected_points[0]
                    if i < len(selected_points)-1 or len(selected_points) == 4:
                        cv2.line(display_frame, start_point, end_point, (255, 0, 0), 2)
            
            # Draw filled polygon if we have 4 points
            if len(selected_points) == 4:
                pts = np.array(selected_points, np.int32).reshape((-1, 1, 2))
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            
            # Setup instructions
            cv2.putText(display_frame, f"SETUP: Click points {len(selected_points)}/4", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if len(selected_points) == 4:
                cv2.putText(display_frame, "Press 'c' to confirm region", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        else:
            # DETECTION MODE: Show region and detection results
            # Draw the defined region
            if len(selected_points) == 4:
                pts = np.array(selected_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts], (255, 255, 0))
                cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)
            
            # Draw all matches if enabled (only weak ones)
            if show_all_detections:
                for match in all_matches:
                    if match['score'] < detection_threshold:  # Only show matches below main threshold
                        x, y = match['location']
                        w, h = match['size']
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                        cv2.putText(display_frame, f"{match['score']:.2f}", (x, y-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw best match if it meets the threshold
            if best_match and best_score >= detection_threshold:
                x, y = best_match['location']
                w, h = best_match['size']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Draw corner circles
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(display_frame, (x + w, y), 5, (0, 255, 0), -1)
                cv2.circle(display_frame, (x + w, y + h), 5, (0, 255, 0), -1)
                cv2.circle(display_frame, (x, y + h), 5, (0, 255, 0), -1)
                
                # Show "GOOD SET" message
                cv2.putText(display_frame, "GOOD SET", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(display_frame, f"Score: {best_score:.3f} (Scale: {best_match['scale']:.1f}x)", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif best_match and best_score > 0.3:  # Show weak detection
                x, y = best_match['location']
                w, h = best_match['size']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(display_frame, f"Weak: {best_score:.3f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Status info for detection mode
            status_color = (0, 255, 0) if (best_match and best_score >= detection_threshold) else (255, 255, 255)
            cv2.putText(display_frame, f"Threshold: {detection_threshold:.2f} | Matches: {len(all_matches)}", 
                       (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(display_frame, f"Best Score: {best_score:.3f} | Debug: {'ON' if show_all_detections else 'OFF'}", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        cv2.imshow("Reference Detection with Region", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and len(selected_points) == 4 and not setup_complete:
            setup_complete = True
            print("Region confirmed! Starting object detection...")
            print("Looking for reference object inside the defined region.")
        elif key == ord('r'):
            # Reset everything
            selected_points = []
            setup_complete = False
            best_match = None
            best_score = 0
            all_matches = []
            print("Region reset. Click 4 new points.")
        elif key == ord('a') and setup_complete:
            show_all_detections = not show_all_detections
            print(f"Show all detections: {'ON' if show_all_detections else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            detection_threshold = min(1.0, detection_threshold + 0.05)
            print(f"Threshold: {detection_threshold:.2f}")
        elif key == ord('-'):
            detection_threshold = max(0.1, detection_threshold - 0.05)
            print(f"Threshold: {detection_threshold:.2f}")
        elif key == ord('s') and setup_complete:
            timestamp = cv2.getTickCount()
            filename = f"detection_frame_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved: {filename}")
            if best_match and best_score >= detection_threshold:
                # Save the detected region
                x, y = best_match['location']
                w, h = best_match['size']
                detected_region = frame[y:y+h, x:x+w]
                region_filename = f"detected_region_{timestamp}.jpg"
                cv2.imwrite(region_filename, detected_region)
                print(f"Saved detected region: {region_filename}")
    
    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()