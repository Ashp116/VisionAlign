from pypylon import pylon
import cv2
import numpy as np

# Global variables for point selection
selected_points = []
ref_gray = None
ref_h, ref_w = 0, 0
ref_template = None
ref_keypoints = None
ref_descriptors = None
ref_templates = []
setup_complete = False
reference_image_loaded = False

def mouse_callback(event, x, y, flags, param):
    global selected_points, setup_complete
    
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append((x, y))
        print(f"Point {len(selected_points)} selected: ({x}, {y})")
        
        if len(selected_points) == 4:
            print("4 points selected! Press 'c' to confirm and start tracking, or 'r' to reset points")

def load_reference_image():
    """Load the reference.jpg image and prepare it for detection"""
    global ref_gray, ref_h, ref_w, ref_template, ref_keypoints, ref_descriptors, reference_image_loaded
    
    try:
        # Load the reference image
        ref_image = cv2.imread("reference.jpg")
        if ref_image is None:
            print("ERROR: Could not load reference.jpg!")
            return False
        
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        ref_h, ref_w = ref_gray.shape
        ref_template = ref_gray.copy()
        
        # Create ORB detector for feature-based matching
        orb = cv2.ORB_create(nfeatures=1000)
        ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
        
        print(f"Reference image loaded: {ref_w}x{ref_h}")
        print(f"Reference features detected: {len(ref_keypoints) if ref_keypoints else 0}")
        
        # Create template matching versions at multiple scales
        global ref_templates
        ref_templates = []
        for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]:
            scaled_w = int(ref_w * scale)
            scaled_h = int(ref_h * scale)
            if scaled_w > 10 and scaled_h > 10 and scaled_w < 800 and scaled_h < 600:
                scaled_template = cv2.resize(ref_gray, (scaled_w, scaled_h))
                ref_templates.append((scaled_template, scale))
        
        print(f"Created {len(ref_templates)} scaled templates")
        reference_image_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading reference image: {e}")
        return False

def extract_reference_region(frame, points):
    """This function is now just for defining the search region"""
    global setup_complete
    print(f"Search region defined with {len(points)} points")
    setup_complete = True

def compare_rect(ref_gray, candidate_gray):
    """Enhanced comparison using multiple methods with reference image"""
    if ref_gray is None or candidate_gray.size == 0 or not reference_image_loaded:
        return 0
    
    try:
        best_score = 0
        
        # Method 1: Multi-scale template matching
        best_template_score = 0
        for template, scale in ref_templates:
            if candidate_gray.shape[0] >= template.shape[0] and candidate_gray.shape[1] >= template.shape[1]:
                result = cv2.matchTemplate(candidate_gray, template, cv2.TM_CCOEFF_NORMED)
                max_val = np.max(result)
                best_template_score = max(best_template_score, max_val)
        
        # Method 2: Direct template matching at candidate size
        if candidate_gray.shape[0] > 20 and candidate_gray.shape[1] > 20:
            # Resize reference to match candidate size for comparison
            resized_ref = cv2.resize(ref_gray, (candidate_gray.shape[1], candidate_gray.shape[0]))
            direct_score = cv2.matchTemplate(candidate_gray, resized_ref, cv2.TM_CCOEFF_NORMED)[0,0]
            best_template_score = max(best_template_score, direct_score)
        
        # Method 3: Feature matching if available
        feature_score = 0
        if ref_descriptors is not None and len(ref_keypoints) > 5:
            orb = cv2.ORB_create(nfeatures=500)
            kp, desc = orb.detectAndCompute(candidate_gray, None)
            
            if desc is not None and len(kp) > 3:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(ref_descriptors, desc)
                if len(matches) > 3:
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = [m for m in matches if m.distance < 60]
                    if len(good_matches) > 0:
                        feature_score = len(good_matches) / len(ref_keypoints)
        
        # Method 4: Histogram comparison
        hist_score = 0
        ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
        cand_hist = cv2.calcHist([candidate_gray], [0], None, [256], [0, 256])
        hist_score = cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_CORREL)
        
        # Combine scores with weights favoring template matching
        final_score = (0.6 * best_template_score + 0.3 * feature_score + 0.1 * max(0, hist_score))
        
        return final_score
        
    except Exception as e:
        print(f"Comparison error: {e}")
        return 0

def is_rect_in_polygon(rect_points, polygon_points):
    """Check if rectangle overlaps with the polygon defined by selected points"""
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # Check if at least 2 corners of the rectangle are inside the polygon
    # This is more flexible than requiring all corners to be inside
    inside_count = 0
    for point in rect_points:
        result = cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False)
        if result >= 0:  # Point is inside or on boundary
            inside_count += 1
    
    # Return True if at least half the points are inside
    return inside_count >= 2

MATCH_THRESHOLD = 0.15  # Lower threshold for reference-based detection
DETECTION_SKIP = 2  # Very frequent detection for better responsiveness

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

print("=== REFERENCE-BASED OBJECT DETECTION ===")
print("Loading reference.jpg...")

# Load reference image first
if not load_reference_image():
    print("Failed to load reference image. Exiting.")
    exit(1)

print("\n=== SETUP MODE ===")
print("Optional: Click 4 points on the image to define search region")
print("Press 'c' to confirm points and start tracking (or skip region selection)")
print("Press 'r' to reset points")
print("Press 'q' to quit")
print("\n=== TRACKING MODE CONTROLS (after setup) ===")
print("Press '+' to increase threshold")
print("Press '-' to decrease threshold") 
print("Press 'd' to toggle debug mode")
print("Press 's' to save debug images (when object detected)")
print("Press 'q' to quit")

debug_mode = False

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
            cv2.putText(display_frame, f"Points: {len(selected_points)}/4 (Optional)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, "Define search region or press 'c' to skip", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if len(selected_points) == 4:
                cv2.putText(display_frame, "Press 'c' to confirm, 'r' to reset", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Reference loaded: {ref_w}x{ref_h}", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Rectangle Detection & Matching", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):  # Allow skipping region selection
                if len(selected_points) == 4:
                    # Confirm points and start tracking
                    extract_reference_region(frame_resized, selected_points)
                    print("=== TRACKING MODE ===")
                    print("Search region defined. Now tracking the reference object...")
                else:
                    # Skip region selection and track in full frame
                    setup_complete = True
                    print("=== TRACKING MODE ===")
                    print("Tracking reference object in full frame...")
                print("Press 'q' to quit")
            elif key == ord('r'):
                # Reset points
                selected_points = []
                print("Points reset. Click 4 new points or press 'c' to skip region.")
                
        else:
            # Tracking mode - original detection logic
            frame_counter += 1

            # --- Only run detection every DETECTION_SKIP frames ---
            if frame_counter % DETECTION_SKIP == 0:
                # Enhanced preprocessing for better edge detection
                blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
                
                # Try multiple edge detection approaches
                edges1 = cv2.Canny(blur, 20, 60)  # More sensitive
                edges2 = cv2.Canny(blur, 50, 150)  # Original
                edges3 = cv2.Canny(blur, 80, 200)  # Less sensitive
                
                # Combine edges with different sensitivities
                edges_combined = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
                
                # Morphological operations to connect nearby edges
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)

                contours, hierarchy = cv2.findContours(edges_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                best_score = -1
                best_pts = None
                rectangles_found = 0
                rectangles_in_region = 0
                all_candidates = []  # Store all candidates for debugging

                for cnt in contours:
                    # Try different approximation levels
                    for epsilon_factor in [0.02, 0.03, 0.04, 0.05]:
                        approx = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
                        if len(approx) == 4:
                            rectangles_found += 1
                            pts = approx.reshape(4, 2)
                            
                            # Check minimum size (made more flexible)
                            x, y, w, h = cv2.boundingRect(pts)
                            if w < 12 or h < 12:  # Even smaller minimum size
                                continue
                            
                            # More flexible shape validation
                            # Calculate area ratio and aspect ratio
                            contour_area = cv2.contourArea(approx)
                            bbox_area = w * h
                            area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
                            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                            
                            # Skip if too thin or area ratio is too small
                            if area_ratio < 0.3 or aspect_ratio > 5:
                                continue
                                
                            # More lenient angle checking
                            def angle(pt1, pt2, pt0):
                                dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
                                dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
                                ang = np.arccos((dx1*dx2 + dy1*dy2) /
                                                (np.sqrt(dx1*dx1+dy1*dy1) * np.sqrt(dx2*dx2+dy2*dy2)+1e-10))
                                return np.degrees(ang)
                            
                            angles = [angle(pts[(i-1)%4], pts[(i+1)%4], pts[i]) for i in range(4)]
                            
                            # More flexible angle requirement (allow for slight distortions)
                            angle_ok = all(45 <= a <= 135 for a in angles)
                            
                            if angle_ok:
                                # Check if the detected rectangle is within our defined polygon region
                                if is_rect_in_polygon(pts, selected_points):
                                    rectangles_in_region += 1
                                    candidate = gray_frame[y:y+h, x:x+w]
                                    
                                    try:
                                        score = compare_rect(ref_gray, candidate)
                                        all_candidates.append((pts, score, w, h))
                                        
                                        if debug_mode:
                                            print(f"Rectangle {rectangles_in_region}: Score = {score:.3f}, Size: {w}x{h}")
                                    except (cv2.error, ZeroDivisionError, Exception) as e:
                                        if debug_mode:
                                            print(f"Error processing candidate: {e}")
                                        continue
                                        
                                    if score > best_score:
                                        best_score = score
                                        best_pts = pts
                            break  # Found a valid 4-sided approximation, move to next contour
                
                # Post-process candidates to find the best match
                if all_candidates:
                    # Sort by score and take the best
                    all_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_pts, best_score = all_candidates[0][0], all_candidates[0][1]
                
                if debug_mode and frame_counter % (DETECTION_SKIP * 4) == 0:  # Print every 4th detection cycle
                    print(f"Found {rectangles_found} rectangles, {rectangles_in_region} in region")
                    print(f"Best score: {best_score:.3f}, Total candidates: {len(all_candidates)}")
                    if len(all_candidates) > 0:
                        print(f"Top 3 candidates: {[(c[1], c[2], c[3]) for c in all_candidates[:3]]}")

            # --- Draw detection results ---
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
            
            # Show current best score for debugging (always show if we have any score)
            if best_score > 0:
                color = (0, 255, 0) if best_score >= MATCH_THRESHOLD else (0, 165, 255)  # Green if above threshold, Orange if below
                cv2.putText(display_frame, f"Best Score: {best_score:.3f} (Threshold: {MATCH_THRESHOLD})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show additional info
            cv2.putText(display_frame, f"Frame: {frame_counter}", (10, display_frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Debug: {'ON' if debug_mode else 'OFF'}", (10, display_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Features: {len(ref_keypoints) if ref_keypoints else 0}", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if best_pts is not None and best_score >= MATCH_THRESHOLD:
                cv2.polylines(display_frame, [best_pts], isClosed=True, color=(0, 255, 0), thickness=3)
                for x, y in best_pts:
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(display_frame, "DETECTED IN REGION", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif best_pts is not None:  # Show detection but below threshold
                cv2.polylines(display_frame, [best_pts], isClosed=True, color=(0, 165, 255), thickness=2)  # Orange outline
                cv2.putText(display_frame, "DETECTED (Below Threshold)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Show candidate rectangles in debug mode
            if debug_mode and 'all_candidates' in locals() and len(all_candidates) > 1:
                for i, (pts, score, w, h) in enumerate(all_candidates[1:4]):  # Show top 3 additional candidates
                    if pts is not None:
                        color = (128, 128, 128)  # Gray for other candidates
                        cv2.polylines(display_frame, [pts], isClosed=True, color=color, thickness=1)
                        # Show score near the rectangle
                        x, y, _, _ = cv2.boundingRect(pts)
                        cv2.putText(display_frame, f"{score:.2f}", (x, y-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imshow("Rectangle Detection & Matching", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):  # Increase threshold
                MATCH_THRESHOLD = min(1.0, MATCH_THRESHOLD + 0.05)
                print(f"Threshold increased to: {MATCH_THRESHOLD:.2f}")
            elif key == ord('-'):  # Decrease threshold
                MATCH_THRESHOLD = max(0.0, MATCH_THRESHOLD - 0.05)
                print(f"Threshold decreased to: {MATCH_THRESHOLD:.2f}")
            elif key == ord('d'):  # Toggle debug mode
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('s') and best_pts is not None:  # Save current detection
                timestamp = cv2.getTickCount()
                debug_filename = f"debug_detection_{timestamp}.jpg"
                cv2.imwrite(debug_filename, display_frame)
                print(f"Saved debug image: {debug_filename}")
                if best_score >= MATCH_THRESHOLD:
                    # Also save the detected region
                    x, y, w, h = cv2.boundingRect(best_pts)
                    detected_region = frame_resized[y:y+h, x:x+w]
                    region_filename = f"debug_detected_region_{timestamp}.jpg"
                    cv2.imwrite(region_filename, detected_region)
                    print(f"Saved detected region: {region_filename}")

    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
