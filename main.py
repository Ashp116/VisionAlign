from pypylon import pylon
import cv2
import numpy as np
import time

# --- thresholds for AE/metrics (copied from live-compare-overlay)
THRESH = {
    "sharp_warn": -10.0, "sharp_crit": -20.0,
    "mean_warn":  10.0,  "mean_crit":  15.0,
    "std_warn":   -10.0, "std_crit":   -20.0,
    "clip_warn":   1.0,  "clip_crit":   2.0,
}


# --- metrics helpers (tenengrad, exposure, analyze, compute_metrics)
def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
    return float(np.mean(gx**2 + gy**2))

def exposure(gray: np.ndarray):
    mean = float(np.mean(gray))
    std  = float(np.std(gray))
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    total = gray.size
    clip = float(hist[:2].sum() + hist[254:].sum()) / total
    return mean, std, clip

def analyze(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean, std, clip = exposure(gray)
    sharp = tenengrad(gray)
    return {"image": img_bgr, "mean": mean, "std": std, "clip": clip, "sharp": sharp}

def sev_label(value, warn, crit, invert=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "CRIT", (66,66,255)
    if invert:
        level = abs(value)
        if level >= crit: return "CRIT", (66,66,255)
        if level >= warn: return "WARN", (32,176,255)
        return "OK", (107,191,32)
    else:
        if value <= crit: return "CRIT", (66,66,255)
        if value <= warn: return "WARN", (32,176,255)
        return "OK", (107,191,32)

def compute_metrics(master, live):
    sharp_pct  = 100.0*(live['sharp']/master['sharp'] - 1.0) if master['sharp'] > 1e-9 else float('nan')
    mean_delta = live['mean'] - master['mean']
    std_pct    = 100.0*(live['std']/master['std'] - 1.0) if master['std'] > 1e-9 else float('nan')
    clip_pp    = (live['clip'] - master['clip']) * 100.0

    sev_sharp, col_sharp = sev_label(sharp_pct, THRESH['sharp_warn'], THRESH['sharp_crit'])
    s_mean,_ = sev_label(mean_delta, THRESH['mean_warn'], THRESH['mean_crit'], invert=True)
    s_std,_  = sev_label(std_pct,    THRESH['std_warn'],  THRESH['std_crit'])
    s_clip,_ = sev_label(clip_pp,    THRESH['clip_warn'], THRESH['clip_crit'], invert=True)
    sev_map = {"OK":1,"WARN":2,"CRIT":3}
    lighting_lvl = max(sev_map[s_mean], sev_map[s_std], sev_map[s_clip])
    sev_light = {1:"OK",2:"WARN",3:"CRIT"}[lighting_lvl]

    suggestions = []
    if sev_sharp in ("WARN","CRIT"):
        suggestions.append("FOCUS CAMERA")
    if sev_light in ("WARN","CRIT"):
        if mean_delta < -THRESH['mean_warn']:
            suggestions.append("Increase brightness/exposure")
        elif mean_delta > THRESH['mean_warn']:
            suggestions.append("Decrease brightness/exposure")
        if std_pct < THRESH['std_warn']:
            suggestions.append("Increase contrast")
        if clip_pp > THRESH['clip_warn']:
            suggestions.append("Reduce exposure to avoid clipping")

    return {
        "sharp_pct": sharp_pct,
        "mean_delta": mean_delta,
        "std_pct": std_pct,
        "clip_pp": clip_pp,
        "sev_sharp": sev_sharp,
        "sev_light": sev_light,
        "suggestion": " | ".join(suggestions) if suggestions else "—"
    }


class AutoExposureController:
    """Auto-exposure controller similar to live-compare-overlay."""
    def __init__(self, *, camera=None, cap=None, use_basler=False,
                 min_exposure=-100000.0, max_exposure=100000.0,
                 step_pct=0.1, min_interval_s=1.0):
        self.camera = camera
        self.cap = cap
        self.use_basler = use_basler
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.step_pct = step_pct
        self.min_interval_s = min_interval_s
        self.last_change = 0.0

    def _get_exposure_basler(self):
        try:
            if not self.camera:
                return None
            if hasattr(self.camera, 'ExposureTime'):
                return float(self.camera.ExposureTime.GetValue())
            if hasattr(self.camera, 'ExposureTimeAbs'):
                return float(self.camera.ExposureTimeAbs.GetValue())
        except Exception:
            return None

    def _set_exposure_basler(self, val):
        try:
            if not self.camera:
                return False
            if hasattr(self.camera, 'ExposureTime'):
                self.camera.ExposureTime.SetValue(int(val))
                return True
            if hasattr(self.camera, 'ExposureTimeAbs'):
                self.camera.ExposureTimeAbs.SetValue(float(val))
                return True
        except Exception:
            return False
        return False

    def _get_exposure_opencv(self):
        try:
            if not self.cap:
                return None
            val = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            return float(val) if val is not None and val != -1 else None
        except Exception:
            return None

    def _set_exposure_opencv(self, val):
        try:
            if not self.cap:
                return False
            return self.cap.set(cv2.CAP_PROP_EXPOSURE, float(val))
        except Exception:
            return False

    def get_exposure(self):
        return self._get_exposure_basler() if self.use_basler else self._get_exposure_opencv()

    def set_exposure(self, val):
        val = float(np.clip(val, self.min_exposure, self.max_exposure))
        if self.use_basler:
            return self._set_exposure_basler(val)
        return self._set_exposure_opencv(val)

    def step_exposure(self, direction=1):
        now = time.time()
        if now - self.last_change < self.min_interval_s:
            return False
        cur = self.get_exposure()
        if cur is None:
            return False
        if cur == 0:
            new = cur + direction * 1.0
        else:
            new = cur * (1.0 + direction * self.step_pct)
        new = float(np.clip(new, self.min_exposure, self.max_exposure))
        ok = self.set_exposure(new)
        if ok:
            self.last_change = now
        return ok


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

def is_rect_in_polygon(rect_points, polygon_points, margin_percent=5):
    """Check if rectangle overlaps with the polygon defined by selected points, with optional margin"""
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # Calculate the center of the polygon
    center_x = np.mean(polygon[:, 0])
    center_y = np.mean(polygon[:, 1])
    
    # Expand the polygon by the margin percentage
    expanded_polygon = []
    for point in polygon:
        # Calculate vector from center to point
        dx = point[0] - center_x
        dy = point[1] - center_y
        
        # Expand by margin percentage
        expanded_x = center_x + dx * (1 + margin_percent / 100.0)
        expanded_y = center_y + dy * (1 + margin_percent / 100.0)
        
        expanded_polygon.append([int(expanded_x), int(expanded_y)])
    
    expanded_polygon = np.array(expanded_polygon, dtype=np.int32)
    
    # Check if at least 2 corners of the rectangle are inside the expanded polygon
    # This is more flexible than requiring all corners to be inside
    inside_count = 0
    for point in rect_points:
        result = cv2.pointPolygonTest(expanded_polygon, (int(point[0]), int(point[1])), False)
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
best_outside_pts = None
best_outside_score = -1
best_bbox_pts = None
best_outside_bbox_pts = None
MASTER_METRICS = None

# Setup AutoExposureController for Basler camera (best-effort)
ae = None
try:
    min_exp = -1e9; max_exp = 1e9
    try:
        if hasattr(camera, 'ExposureTimeMin'):
            min_exp = float(camera.ExposureTimeMin.GetValue())
        if hasattr(camera, 'ExposureTimeMax'):
            max_exp = float(camera.ExposureTimeMax.GetValue())
    except Exception:
        pass
    ae = AutoExposureController(camera=camera, use_basler=True, min_exposure=min_exp, max_exposure=max_exp, step_pct=0.12, min_interval_s=1.0)
    print('[AE] Auto-exposure controller created for Basler')
except Exception:
    ae = None

# Create window and set mouse callback for point selection
cv2.namedWindow("Rectangle Detection & Matching", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Rectangle Detection & Matching", mouse_callback)

print("=== REFERENCE-BASED OBJECT DETECTION ===")
print("Loading reference.jpg...")

# Load reference image first
if not load_reference_image():
    print("Failed to load reference image. Exiting.")
    exit(1)

# Compute master metrics for AE comparisons
MASTER_METRICS = None
try:
    ref_img = cv2.imread('reference.jpg')
    if ref_img is not None:
        MASTER_METRICS = analyze(ref_img)
        print('[AE] Master metrics computed from reference.jpg')
except Exception:
    MASTER_METRICS = None

print("\n=== SETUP MODE ===")
print("Optional: Click 4 points on the image to define search region")
print("Press 'c' to confirm points and start tracking (or skip region selection)")
print("Press 'r' to reset points")
print("Press 'q' to quit")
print("\n=== DETECTION COLORS ===")
print("GREEN: Objects detected inside the region (above threshold)")
print("ORANGE: Objects detected inside the region (below threshold)")
print("MAGENTA: Objects detected outside the region (above threshold)")
print("PURPLE: Objects detected outside the region (below threshold)")
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
            cv2.putText(display_frame, f"Points: {len(selected_points)}/4 (REQUIRED)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if len(selected_points) < 4:
                cv2.putText(display_frame, "Click 4 points to define the search region", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, "All points selected! Press 'c' to confirm", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(selected_points) == 4:
                cv2.putText(display_frame, "Press 'c' to confirm, 'r' to reset", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Reference loaded: {ref_w}x{ref_h}", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Rectangle Detection & Matching", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):  # Require 4 points before confirming
                if len(selected_points) == 4:
                    # Confirm points and start tracking
                    extract_reference_region(frame_resized, selected_points)
                    print("=== TRACKING MODE ===")
                    print("Search region defined. Now tracking the reference object...")
                    print("Press 'q' to quit")
                else:
                    print(f"Please select all 4 points first. Currently selected: {len(selected_points)}/4")
            elif key == ord('r'):
                # Reset points
                selected_points = []
                print("Points reset. Click 4 new points to define the region.")
                
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

                # Reset detection variables for this frame
                current_best_score = -1
                current_best_pts = None
                current_best_outside_score = -1
                current_best_outside_pts = None
                current_best_bbox_pts = None
                current_best_outside_bbox_pts = None
                rectangles_found = 0
                rectangles_in_region = 0
                rectangles_outside_region = 0
                all_candidates = []  # Store all candidates for debugging
                outside_candidates = []  # Store candidates outside the region

                for cnt in contours:
                    # Try different approximation levels
                    for epsilon_factor in [0.02, 0.03, 0.04, 0.4]:
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
                                candidate = gray_frame[y:y+h, x:x+w]
                                
                                try:
                                    score = compare_rect(ref_gray, candidate)
                                    
                                    # Check if the detected rectangle is within our defined polygon region
                                    if len(selected_points) == 4 and is_rect_in_polygon(pts, selected_points):
                                        # Inside region
                                        rectangles_in_region += 1
                                        # Store both contour points and bounding rect for flexibility
                                        bbox_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                                        all_candidates.append((pts, bbox_pts, score, w, h, x, y))
                                        
                                        if debug_mode:
                                            print(f"Rectangle IN region {rectangles_in_region}: Score = {score:.3f}, Size: {w}x{h}")
                                        
                                        if score > current_best_score:
                                            current_best_score = score
                                            current_best_pts = pts
                                            current_best_bbox_pts = bbox_pts
                                    elif len(selected_points) == 4:
                                        # Outside region (only when region is defined)
                                        
                                        # Filter out small, low-confidence outside detections (likely noise)
                                        MIN_OUTSIDE_SIZE_FOR_LOW_CONFIDENCE = 100  # Minimum size for low-confidence outside detections
                                        
                                        # If detection is below threshold and small, ignore it
                                        if score < MATCH_THRESHOLD and (w < MIN_OUTSIDE_SIZE_FOR_LOW_CONFIDENCE or h < MIN_OUTSIDE_SIZE_FOR_LOW_CONFIDENCE):
                                            if debug_mode:
                                                print(f"Rectangle OUTSIDE region IGNORED (too small + low confidence): Score = {score:.3f}, Size: {w}x{h}")
                                            continue  # Skip this detection
                                        
                                        rectangles_outside_region += 1
                                        # Store both contour points and bounding rect for flexibility
                                        bbox_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                                        outside_candidates.append((pts, bbox_pts, score, w, h, x, y))
                                        
                                        if debug_mode:
                                            print(f"Rectangle OUTSIDE region {rectangles_outside_region}: Score = {score:.3f}, Size: {w}x{h}")
                                        
                                        if score > current_best_outside_score:
                                            current_best_outside_score = score
                                            current_best_outside_pts = pts
                                            current_best_outside_bbox_pts = bbox_pts
                                    else:
                                        # No region defined - treat all as "inside"
                                        rectangles_in_region += 1
                                        # Store both contour points and bounding rect for flexibility
                                        bbox_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                                        all_candidates.append((pts, bbox_pts, score, w, h, x, y))
                                        
                                        if debug_mode:
                                            print(f"Rectangle (no region) {rectangles_in_region}: Score = {score:.3f}, Size: {w}x{h}")
                                        
                                        if score > current_best_score:
                                            current_best_score = score
                                            current_best_pts = pts
                                            current_best_bbox_pts = bbox_pts
                                            
                                except (cv2.error, ZeroDivisionError, Exception) as e:
                                    if debug_mode:
                                        print(f"Error processing candidate: {e}")
                                    continue
                            break  # Found a valid 4-sided approximation, move to next contour
                
                # Post-process candidates to find the best match
                if all_candidates:
                    # Sort by score and take the best
                    all_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by score (index 2)
                    current_best_pts, current_best_bbox_pts, current_best_score = all_candidates[0][0], all_candidates[0][1], all_candidates[0][2]
                
                # Post-process outside candidates too
                if outside_candidates:
                    # Sort by score and take the best
                    outside_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by score (index 2)
                    current_best_outside_pts, current_best_outside_bbox_pts, current_best_outside_score = outside_candidates[0][0], outside_candidates[0][1], outside_candidates[0][2]
                
                # Update global variables with current frame's best detections
                best_score = current_best_score
                best_pts = current_best_pts
                best_outside_score = current_best_outside_score
                best_outside_pts = current_best_outside_pts
                best_bbox_pts = current_best_bbox_pts
                best_outside_bbox_pts = current_best_outside_bbox_pts
                
                if debug_mode and frame_counter % (DETECTION_SKIP * 4) == 0:  # Print every 4th detection cycle
                    print(f"Found {rectangles_found} rectangles, {rectangles_in_region} in region, {rectangles_outside_region} outside")
                    print(f"Best inside score: {current_best_score:.3f}, Best outside score: {current_best_outside_score:.3f}")
                    print(f"Inside candidates: {len(all_candidates)}, Outside candidates: {len(outside_candidates)}")
                    if len(all_candidates) > 0:
                        print(f"Top 3 inside candidates: {[(c[2], c[3], c[4]) for c in all_candidates[:3]]}")  # score, w, h
                    if len(outside_candidates) > 0:
                        print(f"Top 3 outside candidates: {[(c[2], c[3], c[4]) for c in outside_candidates[:3]]}")  # score, w, h

                # Auto-exposure logic (only when we have master metrics and AE controller)
                try:
                    if ae is not None and MASTER_METRICS is not None:
                        live_metrics = analyze(frame_resized)
                        info = compute_metrics(MASTER_METRICS, live_metrics)
                        if info['sev_light'] in ('WARN', 'CRIT'):
                            acted = False
                            # Reduce exposure if clipping
                            if info['clip_pp'] > THRESH['clip_warn']:
                                acted = ae.step_exposure(direction=-1)
                                if acted:
                                    print(f"[AE] Reduced exposure due to clipping (clip_pp={info['clip_pp']:.2f})")
                            # Increase exposure if too dark
                            if not acted and info['mean_delta'] < -THRESH['mean_warn']:
                                acted = ae.step_exposure(direction=1)
                                if acted:
                                    print(f"[AE] Increased exposure (mean_delta={info['mean_delta']:+.2f})")
                            # Try increasing exposure for very low contrast
                            if not acted and info['std_pct'] < THRESH['std_warn']:
                                acted = ae.step_exposure(direction=1)
                                if acted:
                                    print(f"[AE] Increased exposure to help low contrast (std_pct={info['std_pct']:+.2f}%)")
                except Exception:
                    pass

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
                
                # Show 5% margin area in debug mode
                if debug_mode:
                    # Calculate expanded polygon for visualization
                    center_x = np.mean([p[0] for p in selected_points])
                    center_y = np.mean([p[1] for p in selected_points])
                    
                    expanded_points = []
                    for point in selected_points:
                        dx = point[0] - center_x
                        dy = point[1] - center_y
                        expanded_x = center_x + dx * 1.05  # 5% expansion
                        expanded_y = center_y + dy * 1.05
                        expanded_points.append([int(expanded_x), int(expanded_y)])
                    
                    expanded_pts = np.array(expanded_points, np.int32)
                    expanded_pts = expanded_pts.reshape((-1, 1, 2))
                    # Draw expanded region with dotted line effect
                    cv2.polylines(display_frame, [expanded_pts], True, (0, 255, 255), 1)  # Cyan dashed outline
                    cv2.putText(display_frame, "5% Margin", (expanded_points[0][0], expanded_points[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Add center crosshair for alignment reference
                # Only show TARGET when guidance is actually needed
                show_target = False
                
                # Check if inside detection needs guidance
                if best_pts is not None and best_score < MATCH_THRESHOLD:
                    # Calculate if inside object needs guidance
                    if best_bbox_pts is not None:
                        bbox_center_x, bbox_center_y = cv2.boundingRect(best_bbox_pts)[:2]
                        bbox_center_x += cv2.boundingRect(best_bbox_pts)[2] // 2
                        bbox_center_y += cv2.boundingRect(best_bbox_pts)[3] // 2
                        region_center_x = sum(p[0] for p in selected_points) // 4
                        region_center_y = sum(p[1] for p in selected_points) // 4
                        dx = region_center_x - bbox_center_x
                        dy = region_center_y - bbox_center_y
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance > 40:  # Same PROXIMITY_THRESHOLD as in guidance logic
                            show_target = True
                
                # Check if outside detection needs guidance
                if best_outside_pts is not None and best_outside_score < MATCH_THRESHOLD:
                    show_target = True  # Outside detections always need guidance to move into region
                
                # Only draw crosshair if guidance is actually needed
                if show_target:
                    region_center_x = sum(p[0] for p in selected_points) // 4
                    region_center_y = sum(p[1] for p in selected_points) // 4
                    # Draw crosshair at region center
                    cv2.line(display_frame, (region_center_x - 15, region_center_y), (region_center_x + 15, region_center_y), (255, 255, 255), 2)
                    cv2.line(display_frame, (region_center_x, region_center_y - 15), (region_center_x, region_center_y + 15), (255, 255, 255), 2)
                    cv2.circle(display_frame, (region_center_x, region_center_y), 5, (255, 255, 255), 2)
                    cv2.putText(display_frame, "TARGET", (region_center_x + 20, region_center_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show current best scores for debugging
            score_y_offset = 30
            if best_score > 0:
                color = (0, 255, 0)  # Always green for inside score
                cv2.putText(display_frame, f"Inside Score: {best_score:.3f} (Threshold: {MATCH_THRESHOLD})", (10, score_y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                score_y_offset += 25
            
            if best_outside_score > 0:
                color = (255, 0, 255) if best_outside_score >= MATCH_THRESHOLD else (128, 0, 128)  # Magenta if above threshold, Purple if below
                cv2.putText(display_frame, f"Outside Score: {best_outside_score:.3f}", (10, score_y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                score_y_offset += 25
            
            # Show additional info
            cv2.putText(display_frame, f"Frame: {frame_counter}", (10, display_frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Debug: {'ON' if debug_mode else 'OFF'}", (10, display_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Features: {len(ref_keypoints) if ref_keypoints else 0}", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw detection results for objects inside the region
            detection_y_offset = 60
            if best_pts is not None and best_score >= MATCH_THRESHOLD:
                # High confidence: use original contour points for more precise shape
                cv2.polylines(display_frame, [best_pts], isClosed=True, color=(0, 255, 0), thickness=3)
                for x, y in best_pts:
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(display_frame, "DETECTED IN REGION", (10, detection_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                detection_y_offset += 30
                
                # Add arrow to target even for high confidence detections
                if len(selected_points) == 4 and best_bbox_pts is not None:
                    region_center_x = sum(p[0] for p in selected_points) // 4
                    region_center_y = sum(p[1] for p in selected_points) // 4
                    bbox_center_x, bbox_center_y = cv2.boundingRect(best_bbox_pts)[:2]
                    bbox_center_x += cv2.boundingRect(best_bbox_pts)[2] // 2
                    bbox_center_y += cv2.boundingRect(best_bbox_pts)[3] // 2
                    
                    # Draw arrow FROM object center TO region center (green for high confidence)
                    cv2.arrowedLine(display_frame, 
                                   (bbox_center_x, bbox_center_y), 
                                   (region_center_x, region_center_y), 
                                   (0, 255, 0), 2, tipLength=0.2)
                    
                    # Add a circle at object position
                    cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 6, (0, 255, 0), 2)
                    cv2.putText(display_frame, "OK", (bbox_center_x + 8, bbox_center_y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            elif best_pts is not None and best_bbox_pts is not None:  # Low confidence detection
                # Low confidence: use precise bounding rectangle but don't show visual outline
                # Only show guidance text if needed, no orange visuals
                
                # Add alignment guidance text with position information
                if len(selected_points) == 4:
                    # Calculate center of the defined region
                    region_center_x = sum(p[0] for p in selected_points) // 4
                    region_center_y = sum(p[1] for p in selected_points) // 4
                    
                    # Calculate center of detected object
                    bbox_center_x, bbox_center_y = cv2.boundingRect(best_bbox_pts)[:2]
                    bbox_center_x += cv2.boundingRect(best_bbox_pts)[2] // 2
                    bbox_center_y += cv2.boundingRect(best_bbox_pts)[3] // 2
                    
                    # Calculate distance from object center to region center
                    dx = region_center_x - bbox_center_x
                    dy = region_center_y - bbox_center_y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Define proximity threshold (adjust this value as needed)
                    PROXIMITY_THRESHOLD = 40  # pixels - only show guidance if object is more than 40 pixels from center
                    
                    # Only show guidance if object is significantly misaligned
                    if distance > PROXIMITY_THRESHOLD:
                        guidance = "Match camera to the "
                        arrow_directions = []  # Store arrow directions to draw
                        
                        if abs(dx) > 20 or abs(dy) > 20:  # If significantly off-center
                            if dx > 20:
                                guidance += "left side "
                                arrow_directions.append("LEFT")
                            elif dx < -20:
                                guidance += "right side "
                                arrow_directions.append("RIGHT")
                            if dy > 20:
                                guidance += "upper area "
                                arrow_directions.append("UP")
                            elif dy < -20:
                                guidance += "lower area "
                                arrow_directions.append("DOWN")
                            guidance += "of the reference region"
                        else:
                            guidance += "center of the reference region"
                            arrow_directions.append("CENTER")
                        
                        # Draw arrow FROM object center TO region center
                        cv2.arrowedLine(display_frame, 
                                       (bbox_center_x, bbox_center_y), 
                                       (region_center_x, region_center_y), 
                                       (255, 255, 0), 3, tipLength=0.2)
                        
                        # Add a circle at object position to show start point
                        cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 8, (255, 255, 0), 2)
                        cv2.putText(display_frame, "OBJ", (bbox_center_x + 10, bbox_center_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        # Draw directional arrows around the guidance text (smaller, supplementary)
                        text_x = 10
                        text_y = detection_y_offset
                        
                        # Draw small supplementary arrows based on direction
                        for i, direction in enumerate(arrow_directions):
                            offset_x = i * 15  # Spread arrows horizontally if multiple
                            if direction == "LEFT":
                                cv2.arrowedLine(display_frame, (text_x - 30 + offset_x, text_y - 5), (text_x - 15 + offset_x, text_y - 5), (255, 255, 0), 1, tipLength=0.5)
                            elif direction == "RIGHT":
                                cv2.arrowedLine(display_frame, (text_x - 30 + offset_x, text_y - 5), (text_x - 15 + offset_x, text_y - 5), (255, 255, 0), 1, tipLength=0.5)
                            elif direction == "UP":
                                cv2.arrowedLine(display_frame, (text_x - 22 + offset_x, text_y + 2), (text_x - 22 + offset_x, text_y - 12), (255, 255, 0), 1, tipLength=0.5)
                            elif direction == "DOWN":
                                cv2.arrowedLine(display_frame, (text_x - 22 + offset_x, text_y - 12), (text_x - 22 + offset_x, text_y + 2), (255, 255, 0), 1, tipLength=0.5)
                            elif direction == "CENTER":
                                cv2.circle(display_frame, (text_x - 22 + offset_x, text_y - 5), 4, (255, 255, 0), 1)
                                cv2.circle(display_frame, (text_x - 22 + offset_x, text_y - 5), 2, (255, 255, 0), -1)
                        
                        cv2.putText(display_frame, guidance, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Cyan text for visibility
                        detection_y_offset += 25
                        cv2.putText(display_frame, "Align object properly for better detection", (10, detection_y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        detection_y_offset += 25
                    # If object is close to center, don't show any guidance messages
                else:
                    # Fallback when no region is defined
                    cv2.putText(display_frame, "Match camera to the reference template position", (10, detection_y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    detection_y_offset += 25
                    cv2.putText(display_frame, "Align object properly for better detection", (10, detection_y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    detection_y_offset += 25
            
            # Draw detection results for objects outside the region
            if best_outside_pts is not None and best_outside_score >= MATCH_THRESHOLD:
                # High confidence: use original contour points for more precise shape
                cv2.polylines(display_frame, [best_outside_pts], isClosed=True, color=(255, 0, 255), thickness=3)  # Magenta for outside detection
                for x, y in best_outside_pts:
                    cv2.circle(display_frame, (x, y), 5, (255, 255, 0), -1)  # Cyan circles
                cv2.putText(display_frame, "DETECTED OUTSIDE REGION", (10, detection_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                detection_y_offset += 30
                
                # Add arrow to target for high confidence outside detections
                if len(selected_points) == 4 and best_outside_bbox_pts is not None:
                    region_center_x = sum(p[0] for p in selected_points) // 4
                    region_center_y = sum(p[1] for p in selected_points) // 4
                    bbox_center_x, bbox_center_y = cv2.boundingRect(best_outside_bbox_pts)[:2]
                    bbox_center_x += cv2.boundingRect(best_outside_bbox_pts)[2] // 2
                    bbox_center_y += cv2.boundingRect(best_outside_bbox_pts)[3] // 2
                    
                    # Draw arrow FROM outside object TO region center (magenta for outside)
                    cv2.arrowedLine(display_frame, 
                                   (bbox_center_x, bbox_center_y), 
                                   (region_center_x, region_center_y), 
                                   (255, 0, 255), 2, tipLength=0.2)
                    
                    # Add a circle at outside object position
                    cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 6, (255, 0, 255), 2)
                    cv2.putText(display_frame, "HIGH", (bbox_center_x + 8, bbox_center_y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
            elif best_outside_pts is not None and best_outside_bbox_pts is not None:  # Low confidence detection
                # Low confidence: use precise bounding rectangle for accurate size/position
                cv2.polylines(display_frame, [best_outside_bbox_pts], isClosed=True, color=(128, 0, 128), thickness=2)  # Purple outline
                for x, y in best_outside_bbox_pts:
                    cv2.circle(display_frame, (x, y), 3, (128, 0, 128), -1)
                cv2.putText(display_frame, "DETECTED OUTSIDE (Below Threshold)", (10, detection_y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
                detection_y_offset += 25
                
                # Add alignment guidance text with position information for outside detection
                if len(selected_points) == 4:
                    # Calculate center of the defined region
                    region_center_x = sum(p[0] for p in selected_points) // 4
                    region_center_y = sum(p[1] for p in selected_points) // 4
                    
                    # Calculate center of detected object
                    bbox_center_x, bbox_center_y = cv2.boundingRect(best_outside_bbox_pts)[:2]
                    bbox_center_x += cv2.boundingRect(best_outside_bbox_pts)[2] // 2
                    bbox_center_y += cv2.boundingRect(best_outside_bbox_pts)[3] // 2
                    
                    # Determine direction to move camera to bring object into region
                    dx = bbox_center_x - region_center_x  # Positive means object is to the right of region
                    dy = bbox_center_y - region_center_y  # Positive means object is below region
                    
                    guidance = "Move camera "
                    arrow_directions = []  # Store arrow directions for outside detection
                    
                    if abs(dx) > 20 or abs(dy) > 20:  # If significantly off-center
                        if dx > 20:
                            guidance += "left "  # Move camera left to bring right object into view
                            arrow_directions.append("LEFT")
                        elif dx < -20:
                            guidance += "right "  # Move camera right to bring left object into view
                            arrow_directions.append("RIGHT")
                        if dy > 20:
                            guidance += "up "  # Move camera up to bring lower object into view
                            arrow_directions.append("UP")
                        elif dy < -20:
                            guidance += "down "  # Move camera down to bring upper object into view
                            arrow_directions.append("DOWN")
                        guidance += "to align with reference region"
                    else:
                        guidance += "slightly to align with reference region"
                        arrow_directions.append("CENTER")
                else:
                    guidance = "Move camera to align with reference template"
                    arrow_directions.append("CENTER")
                
                # Draw directional arrows for outside detection
                text_x = 10
                text_y = detection_y_offset
                
                # Draw main arrow FROM outside object TO region center
                cv2.arrowedLine(display_frame, 
                               (bbox_center_x, bbox_center_y), 
                               (region_center_x, region_center_y), 
                               (255, 0, 255), 3, tipLength=0.2)
                
                # Add a circle at outside object position to show start point
                cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 8, (255, 0, 255), 2)
                cv2.putText(display_frame, "OUT", (bbox_center_x + 10, bbox_center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Draw small supplementary arrows based on direction (using magenta color for outside detection)
                for i, direction in enumerate(arrow_directions):
                    offset_x = i * 15  # Spread arrows horizontally if multiple
                    if direction == "LEFT":
                        cv2.arrowedLine(display_frame, (text_x - 30 + offset_x, text_y - 5), (text_x - 15 + offset_x, text_y - 5), (255, 0, 255), 1, tipLength=0.5)
                    elif direction == "RIGHT":
                        cv2.arrowedLine(display_frame, (text_x - 30 + offset_x, text_y - 5), (text_x - 15 + offset_x, text_y - 5), (255, 0, 255), 1, tipLength=0.5)
                    elif direction == "UP":
                        cv2.arrowedLine(display_frame, (text_x - 22 + offset_x, text_y + 2), (text_x - 22 + offset_x, text_y - 12), (255, 0, 255), 1, tipLength=0.5)
                    elif direction == "DOWN":
                        cv2.arrowedLine(display_frame, (text_x - 22 + offset_x, text_y - 12), (text_x - 22 + offset_x, text_y + 2), (255, 0, 255), 1, tipLength=0.5)
                    elif direction == "CENTER":
                        cv2.circle(display_frame, (text_x - 22 + offset_x, text_y - 5), 6, (255, 0, 255), 1)
                        cv2.circle(display_frame, (text_x - 22 + offset_x, text_y - 5), 3, (255, 0, 255), 1)
                        cv2.circle(display_frame, (text_x - 22 + offset_x, text_y - 5), 1, (255, 0, 255), -1)
                
                cv2.putText(display_frame, guidance, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)  # Magenta text for visibility
                detection_y_offset += 25
                cv2.putText(display_frame, "Bring object into the defined region", (10, detection_y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                detection_y_offset += 25
            
            # Show candidate rectangles in debug mode
            if debug_mode:
                # Show additional inside candidates
                if 'all_candidates' in locals() and len(all_candidates) > 1:
                    for i, candidate_data in enumerate(all_candidates[1:4]):  # Show top 3 additional candidates
                        pts, bbox_pts, score, w, h = candidate_data[0], candidate_data[1], candidate_data[2], candidate_data[3], candidate_data[4]
                        if pts is not None:
                            color = (128, 128, 128)  # Gray for other inside candidates
                            # Use bbox for consistency in debug mode
                            cv2.polylines(display_frame, [bbox_pts], isClosed=True, color=color, thickness=1)
                            # Show score near the rectangle
                            x, y, _, _ = cv2.boundingRect(bbox_pts)
                            cv2.putText(display_frame, f"IN:{score:.2f}", (x, y-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Show additional outside candidates
                if 'outside_candidates' in locals() and len(outside_candidates) > 1:
                    for i, candidate_data in enumerate(outside_candidates[1:4]):  # Show top 3 additional candidates
                        pts, bbox_pts, score, w, h = candidate_data[0], candidate_data[1], candidate_data[2], candidate_data[3], candidate_data[4]
                        if pts is not None:
                            color = (64, 0, 64)  # Dark purple for other outside candidates
                            # Use bbox for consistency in debug mode
                            cv2.polylines(display_frame, [bbox_pts], isClosed=True, color=color, thickness=1)
                            # Show score near the rectangle
                            x, y, _, _ = cv2.boundingRect(bbox_pts)
                            cv2.putText(display_frame, f"OUT:{score:.2f}", (x, y-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imshow("Rectangle Detection & Matching", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):  # Increase threshold
                MATCH_THRESHOLD = min(1.0, MATCH_THRESHOLD + 0.4)
                print(f"Threshold increased to: {MATCH_THRESHOLD:.2f}")
            elif key == ord('-'):  # Decrease threshold
                MATCH_THRESHOLD = max(0.0, MATCH_THRESHOLD - 0.4)
                print(f"Threshold decreased to: {MATCH_THRESHOLD:.2f}")
            elif key == ord('d'):  # Toggle debug mode
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('s') and (best_pts is not None or best_outside_pts is not None):  # Save current detection
                timestamp = cv2.getTickCount()
                debug_filename = f"debug_detection_{timestamp}.jpg"
                cv2.imwrite(debug_filename, display_frame)
                print(f"Saved debug image: {debug_filename}")
                
                # Save detected regions if they meet threshold
                if best_pts is not None and best_score >= MATCH_THRESHOLD:
                    # Use bbox_pts for consistent region extraction
                    if best_bbox_pts is not None:
                        x, y, w, h = cv2.boundingRect(best_bbox_pts)
                        detected_region = frame_resized[y:y+h, x:x+w]
                        region_filename = f"debug_detected_inside_{timestamp}.jpg"
                        cv2.imwrite(region_filename, detected_region)
                        print(f"Saved detected inside region: {region_filename}")
                
                if best_outside_pts is not None and best_outside_score >= MATCH_THRESHOLD:
                    # Use bbox_pts for consistent region extraction
                    if best_outside_bbox_pts is not None:
                        x, y, w, h = cv2.boundingRect(best_outside_bbox_pts)
                        detected_region = frame_resized[y:y+h, x:x+w]
                        region_filename = f"debug_detected_outside_{timestamp}.jpg"
                        cv2.imwrite(region_filename, detected_region)
                        print(f"Saved detected outside region: {region_filename}")

    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
