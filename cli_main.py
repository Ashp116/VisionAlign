#!/usr/bin/env python3
"""
AlignVision CLI Application

A command-line interface for the AlignVision system that supports both OpenCV window display
and web streaming via localhost.

Usage:
    python cli_main.py --mode opencv          # Display in OpenCV window (default)
    python cli_main.py --mode web             # Start web stream on localhost:5000
    python cli_main.py --mode web --port 8080 # Start web stream on localhost:8080
    python cli_main.py --help                 # Show help
"""

import argparse
import sys
import threading
import time
from pypylon import pylon
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import logging

# Suppress Flask's default logging to keep console clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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

# Global variables for streaming
current_frame = None
frame_lock = threading.Lock()
camera = None
streaming_active = False
global_threshold = 0.15

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for point selection in OpenCV mode"""
    global selected_points, setup_complete
    
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append((x, y))
        print(f"Point {len(selected_points)} selected: ({x}, {y})")
        
        if len(selected_points) == 4:
            print("4 points selected! Press 'c' to confirm and start tracking, or 'r' to reset points")

def load_reference_image():
    """Load the reference.jpg image and prepare it for detection"""
    global ref_gray, ref_h, ref_w, ref_template, ref_keypoints, ref_descriptors, reference_image_loaded, ref_templates
    
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
    inside_count = 0
    for point in rect_points:
        result = cv2.pointPolygonTest(expanded_polygon, (int(point[0]), int(point[1])), False)
        if result >= 0:  # Point is inside or on boundary
            inside_count += 1
    
    # Return True if at least half the points are inside
    return inside_count >= 2

def process_frame(frame, MATCH_THRESHOLD=0.15):
    """Process a single frame and return the processed frame with detections"""
    global best_pts, best_score, best_outside_pts, best_outside_score
    global best_bbox_pts, best_outside_bbox_pts
    
    frame_resized = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Detection logic (similar to main.py but optimized for streaming)
    DETECTION_SKIP = 2
    
    # Enhanced preprocessing for better edge detection
    blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    
    # Try multiple edge detection approaches
    edges1 = cv2.Canny(blur, 20, 60)
    edges2 = cv2.Canny(blur, 50, 150)
    edges3 = cv2.Canny(blur, 80, 200)
    
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
    all_candidates = []
    outside_candidates = []

    for cnt in contours:
        # Try different approximation levels
        for epsilon_factor in [0.02, 0.03, 0.04, 0.4]:
            approx = cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                
                # Check minimum size
                x, y, w, h = cv2.boundingRect(pts)
                if w < 12 or h < 12:
                    continue
                
                # Shape validation
                contour_area = cv2.contourArea(approx)
                bbox_area = w * h
                area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                
                if area_ratio < 0.3 or aspect_ratio > 5:
                    continue
                    
                # Angle checking
                def angle(pt1, pt2, pt0):
                    dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
                    dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
                    ang = np.arccos((dx1*dx2 + dy1*dy2) /
                                    (np.sqrt(dx1*dx1+dy1*dy1) * np.sqrt(dx2*dx2+dy2*dy2)+1e-10))
                    return np.degrees(ang)
                
                angles = [angle(pts[(i-1)%4], pts[(i+1)%4], pts[i]) for i in range(4)]
                angle_ok = all(45 <= a <= 135 for a in angles)
                
                if angle_ok:
                    candidate = gray_frame[y:y+h, x:x+w]
                    
                    try:
                        score = compare_rect(ref_gray, candidate)
                        bbox_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                        
                        # Check if the detected rectangle is within our defined polygon region
                        if len(selected_points) == 4 and is_rect_in_polygon(pts, selected_points):
                            # Inside region
                            all_candidates.append((pts, bbox_pts, score, w, h, x, y))
                            if score > current_best_score:
                                current_best_score = score
                                current_best_pts = pts
                                current_best_bbox_pts = bbox_pts
                        elif len(selected_points) == 4:
                            # Outside region
                            MIN_OUTSIDE_SIZE_FOR_LOW_CONFIDENCE = 100
                            if score < MATCH_THRESHOLD and (w < MIN_OUTSIDE_SIZE_FOR_LOW_CONFIDENCE or h < MIN_OUTSIDE_SIZE_FOR_LOW_CONFIDENCE):
                                continue
                            
                            outside_candidates.append((pts, bbox_pts, score, w, h, x, y))
                            if score > current_best_outside_score:
                                current_best_outside_score = score
                                current_best_outside_pts = pts
                                current_best_outside_bbox_pts = bbox_pts
                        else:
                            # No region defined
                            all_candidates.append((pts, bbox_pts, score, w, h, x, y))
                            if score > current_best_score:
                                current_best_score = score
                                current_best_pts = pts
                                current_best_bbox_pts = bbox_pts
                                
                    except Exception as e:
                        continue
                break
    
    # Update global variables
    best_score = current_best_score
    best_pts = current_best_pts
    best_outside_score = current_best_outside_score
    best_outside_pts = current_best_outside_pts
    best_bbox_pts = current_best_bbox_pts
    best_outside_bbox_pts = current_best_outside_bbox_pts
    
    # Draw detection results
    display_frame = frame_resized.copy()
    
    # Draw region if defined
    if len(selected_points) == 4:
        pts = np.array(selected_points, np.int32).reshape((-1, 1, 2))
        overlay = display_frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 255, 0, 50))
        cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
        cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)
        
        # Show target crosshair when needed
        if (best_pts is not None and best_score < MATCH_THRESHOLD) or \
           (best_outside_pts is not None and best_outside_score < MATCH_THRESHOLD):
            region_center_x = sum(p[0] for p in selected_points) // 4
            region_center_y = sum(p[1] for p in selected_points) // 4
            cv2.line(display_frame, (region_center_x - 15, region_center_y), 
                    (region_center_x + 15, region_center_y), (255, 255, 255), 2)
            cv2.line(display_frame, (region_center_x, region_center_y - 15), 
                    (region_center_x, region_center_y + 15), (255, 255, 255), 2)
            cv2.circle(display_frame, (region_center_x, region_center_y), 5, (255, 255, 255), 2)
            cv2.putText(display_frame, "TARGET", (region_center_x + 20, region_center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw detection results - simplified and clean
    if best_pts is not None and best_score >= MATCH_THRESHOLD:
        cv2.polylines(display_frame, [best_pts], isClosed=True, color=(0, 255, 0), thickness=3)
        # Add small status indicator
        if best_bbox_pts is not None:
            bbox_center_x, bbox_center_y = cv2.boundingRect(best_bbox_pts)[:2]
            bbox_center_x += cv2.boundingRect(best_bbox_pts)[2] // 2
            bbox_center_y += cv2.boundingRect(best_bbox_pts)[3] // 2
            cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 8, (0, 255, 0), 2)
            cv2.putText(display_frame, "OK", (bbox_center_x - 8, bbox_center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            
            # For high confidence detection, check if object needs to move to target
            if len(selected_points) == 4:
                region_center_x = sum(p[0] for p in selected_points) // 4
                region_center_y = sum(p[1] for p in selected_points) // 4
                dx = region_center_x - bbox_center_x
                dy = region_center_y - bbox_center_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 40:  # If object is not at target center
                    cv2.arrowedLine(display_frame, (bbox_center_x, bbox_center_y), 
                                   (region_center_x, region_center_y), (0, 255, 0), 2, tipLength=0.2)
                    cv2.putText(display_frame, "Move the object to the target", 
                               (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                       
    elif best_pts is not None and best_bbox_pts is not None:
        # Low confidence detection - show guidance arrow only
        if len(selected_points) == 4:
            region_center_x = sum(p[0] for p in selected_points) // 4
            region_center_y = sum(p[1] for p in selected_points) // 4
            bbox_center_x, bbox_center_y = cv2.boundingRect(best_bbox_pts)[:2]
            bbox_center_x += cv2.boundingRect(best_bbox_pts)[2] // 2
            bbox_center_y += cv2.boundingRect(best_bbox_pts)[3] // 2
            
            dx = region_center_x - bbox_center_x
            dy = region_center_y - bbox_center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 40:  # Only show guidance if significantly misaligned
                cv2.arrowedLine(display_frame, (bbox_center_x, bbox_center_y), 
                               (region_center_x, region_center_y), (255, 255, 0), 3, tipLength=0.2)
                cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 8, (255, 255, 0), 2)
                cv2.putText(display_frame, "ALIGN", (bbox_center_x - 15, bbox_center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 2)
                
                # For low confidence detection, instruct to move camera
                instruction = "Move camera "
                if dx > 20:
                    instruction += "LEFT "
                elif dx < -20:
                    instruction += "RIGHT "
                if dy > 20:
                    instruction += "UP"
                elif dy < -20:
                    instruction += "DOWN"
                if instruction != "Move camera ":
                    cv2.putText(display_frame, instruction.strip(), (10, display_frame.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Outside region detections - simplified
    if best_outside_pts is not None and best_outside_score >= MATCH_THRESHOLD:
        cv2.polylines(display_frame, [best_outside_pts], isClosed=True, color=(255, 0, 255), thickness=3)
        if best_outside_bbox_pts is not None:
            bbox_center_x, bbox_center_y = cv2.boundingRect(best_outside_bbox_pts)[:2]
            bbox_center_x += cv2.boundingRect(best_outside_bbox_pts)[2] // 2
            bbox_center_y += cv2.boundingRect(best_outside_bbox_pts)[3] // 2
            
            # Draw arrow to guide towards region
            region_center_x = sum(p[0] for p in selected_points) // 4
            region_center_y = sum(p[1] for p in selected_points) // 4
            cv2.arrowedLine(display_frame, (bbox_center_x, bbox_center_y), 
                           (region_center_x, region_center_y), (255, 0, 255), 2, tipLength=0.2)
            cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 8, (255, 0, 255), 2)
            cv2.putText(display_frame, "OUT", (bbox_center_x - 12, bbox_center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 2)
            
            # For high confidence outside detection, instruct to move the object
            cv2.putText(display_frame, "Move the object into the region", 
                       (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                       
    elif best_outside_pts is not None and best_outside_bbox_pts is not None:
        # Low confidence outside detection
        cv2.polylines(display_frame, [best_outside_bbox_pts], isClosed=True, color=(128, 0, 128), thickness=2)
        
        if len(selected_points) == 4:
            region_center_x = sum(p[0] for p in selected_points) // 4
            region_center_y = sum(p[1] for p in selected_points) // 4
            bbox_center_x, bbox_center_y = cv2.boundingRect(best_outside_bbox_pts)[:2]
            bbox_center_x += cv2.boundingRect(best_outside_bbox_pts)[2] // 2
            bbox_center_y += cv2.boundingRect(best_outside_bbox_pts)[3] // 2
            
            cv2.arrowedLine(display_frame, (bbox_center_x, bbox_center_y), 
                           (region_center_x, region_center_y), (255, 0, 255), 3, tipLength=0.2)
            cv2.circle(display_frame, (bbox_center_x, bbox_center_y), 8, (255, 0, 255), 2)
            cv2.putText(display_frame, "LOW", (bbox_center_x - 12, bbox_center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 2)
            
            # Calculate movement direction for outside object
            dx_out = bbox_center_x - region_center_x
            dy_out = bbox_center_y - region_center_y
            
            move_instruction = "Move camera "
            if dx_out > 20:
                move_instruction += "LEFT "
            elif dx_out < -20:
                move_instruction += "RIGHT "
            if dy_out > 20:
                move_instruction += "UP"
            elif dy_out < -20:
                move_instruction += "DOWN"
            move_instruction += " to align"
            
            cv2.putText(display_frame, move_instruction, (10, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Check if reference object is detected with high confidence inside the region
    if (best_score >= MATCH_THRESHOLD and best_pts is not None):
        cv2.putText(display_frame, "Aligned!", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Clean overlay panel in top-right corner
    overlay_x = display_frame.shape[1] - 250
    overlay_y = 20
    overlay_height = 120
    
    # Create semi-transparent background for overlay
    overlay_bg = display_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+240].copy()
    overlay_bg[:] = (0, 0, 0)
    cv2.addWeighted(display_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+240], 0.3, 
                   overlay_bg, 0.7, 0, display_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+240])
    
    # Add border to overlay
    cv2.rectangle(display_frame, (overlay_x-2, overlay_y-2), (overlay_x+242, overlay_y+overlay_height+2), (255, 255, 255), 2)
    
    text_y = overlay_y + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    
    # Show scores in overlay
    if best_score > 0:
        status = "DETECTED" if best_score >= MATCH_THRESHOLD else "ALIGNING"
        cv2.putText(display_frame, f"IN: {status} ({best_score:.2f})", 
                   (overlay_x + 5, text_y), font, font_scale, (0, 255, 0), 1)
        text_y += 20
        
    if best_outside_score > 0:
        status = "OUTSIDE" if best_outside_score >= MATCH_THRESHOLD else "OUT-LOW"
        color = (255, 0, 255) if best_outside_score >= MATCH_THRESHOLD else (128, 0, 128)
        cv2.putText(display_frame, f"OUT: {status} ({best_outside_score:.2f})", 
                   (overlay_x + 5, text_y), font, font_scale, color, 1)
        text_y += 20
    
    # Show threshold
    cv2.putText(display_frame, f"Threshold: {MATCH_THRESHOLD:.2f}", 
               (overlay_x + 5, text_y), font, font_scale, (255, 255, 255), 1)
    text_y += 20
    
    # Show detection status
    if len(selected_points) == 4:
        cv2.putText(display_frame, f"Region: Active ({len(selected_points)}/4)", 
                   (overlay_x + 5, text_y), font, font_scale, (0, 255, 255), 1)
    else:
        cv2.putText(display_frame, f"Setup: {len(selected_points)}/4 points", 
                   (overlay_x + 5, text_y), font, font_scale, (255, 255, 0), 1)
    
    return display_frame

def initialize_camera():
    """Initialize the Basler camera"""
    global camera
    
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    if not devices:
        print("No Basler cameras found!")
        return False

    camera = pylon.InstantCamera(tlf.CreateDevice(devices[0]))
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return True

def cleanup_camera():
    """Clean up camera resources"""
    global camera
    if camera:
        camera.StopGrabbing()
        camera.Close()
        camera = None

def run_opencv_mode():
    """Run the application in OpenCV window mode"""
    global current_frame, selected_points, setup_complete
    
    print("=== OPENCV WINDOW MODE ===")
    
    if not initialize_camera():
        return
    
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Create window and set mouse callback for point selection
    cv2.namedWindow("AlignVision - OpenCV Mode", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("AlignVision - OpenCV Mode", mouse_callback)

    print("\n=== SETUP MODE ===")
    print("Click 4 points on the image to define search region")
    print("Press 'c' to confirm points and start tracking")
    print("Press 'r' to reset points")
    print("Press 'q' to quit")
    
    MATCH_THRESHOLD = 0.15
    
    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = converter.Convert(grab_result).GetArray()
                
                if not setup_complete:
                    # Setup mode
                    display_frame = cv2.resize(frame, (640, 480))
                    
                    # Draw selected points
                    for i, point in enumerate(selected_points):
                        cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
                        cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw polygon if we have 4 points
                    if len(selected_points) == 4:
                        pts = np.array(selected_points, np.int32).reshape((-1, 1, 2))
                        overlay = display_frame.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0, 100))
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    
                    # Instructions
                    cv2.putText(display_frame, f"Points: {len(selected_points)}/4", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    if len(selected_points) < 4:
                        cv2.putText(display_frame, "Click 4 points to define search region", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Press 'c' to confirm, 'r' to reset", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow("AlignVision - OpenCV Mode", display_frame)
                else:
                    # Tracking mode
                    display_frame = process_frame(frame, MATCH_THRESHOLD)
                    cv2.imshow("AlignVision - OpenCV Mode", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and len(selected_points) == 4:
                    setup_complete = True
                    print("=== TRACKING MODE ===")
                elif key == ord('r'):
                    selected_points = []
                    print("Points reset. Click 4 new points.")
                elif key == ord('+') or key == ord('='):
                    MATCH_THRESHOLD = min(1.0, MATCH_THRESHOLD + 0.05)
                    print(f"Threshold: {MATCH_THRESHOLD:.2f}")
                elif key == ord('-'):
                    MATCH_THRESHOLD = max(0.0, MATCH_THRESHOLD - 0.05)
                    print(f"Threshold: {MATCH_THRESHOLD:.2f}")

            grab_result.Release()
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cleanup_camera()
        cv2.destroyAllWindows()

def camera_capture_thread():
    """Thread function for camera capture in web mode"""
    global current_frame, frame_lock, camera, streaming_active
    
    if not initialize_camera():
        return
    
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    streaming_active = True
    
    try:
        while streaming_active and camera.IsGrabbing():
            grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result.GrabSucceeded():
                frame = converter.Convert(grab_result).GetArray()
                
                with frame_lock:
                    if setup_complete:
                        current_frame = process_frame(frame, global_threshold)
                    else:
                        # Simple frame for web setup
                        current_frame = cv2.resize(frame, (640, 480))
                        
                        # Add setup instructions to frame
                        cv2.putText(current_frame, "Web Mode - Setup via browser controls", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(current_frame, f"Points selected: {len(selected_points)}/4", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw selected points
                        for i, point in enumerate(selected_points):
                            cv2.circle(current_frame, point, 5, (0, 0, 255), -1)
                            cv2.putText(current_frame, str(i+1), (point[0]+10, point[1]-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            grab_result.Release()
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        print(f"Camera capture error: {e}")
    finally:
        cleanup_camera()

def generate_frames():
    """Generator function for Flask streaming"""
    global current_frame, frame_lock
    
    while streaming_active:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

# Flask web application
app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AlignVision</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body, html {
            height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: #f8f9fa;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }
        
        .video-section {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border: 1px solid #e6e6e6;
        }
        
        .video-container {
            position: relative;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            background: #000;
            border: 2px solid #003478;
        }
        
        .video-feed {
            cursor: crosshair;
            display: block;
            width: 960px;
            height: 720px;
        }
        
        .controls-panel {
            width: 280px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border: 1px solid #e6e6e6;
        }
        
        .btn {
            padding: 14px 20px;
            border: 2px solid #003478;
            background: #003478;
            color: white;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            border-radius: 4px;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn:hover {
            background: #004599;
            border-color: #004599;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 52, 120, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
            transition: transform 0.1s;
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            background: #6c757d;
            border-color: #6c757d;
        }
        
        .btn:disabled:hover {
            transform: none;
            box-shadow: none;
            background: #6c757d;
            border-color: #6c757d;
        }
        
        .btn-primary {
            background: #0066cc;
            border-color: #0066cc;
        }
        
        .btn-primary:hover {
            background: #0052a3;
            border-color: #0052a3;
        }
        
        .btn-danger {
            background: #dc3545;
            border-color: #dc3545;
        }
        
        .btn-danger:hover {
            background: #c82333;
            border-color: #c82333;
        }
        
        .btn-secondary {
            background: white;
            color: #003478;
            border-color: #003478;
        }
        
        .btn-secondary:hover {
            background: #003478;
            color: white;
        }
        
        .threshold-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        
        .threshold-group .btn {
            padding: 12px 16px;
            font-size: 18px;
            font-weight: 700;
            background: white;
            color: #003478;
            border-color: #003478;
        }
        
        .threshold-group .btn:hover {
            background: #003478;
            color: white;
        }
        
        .separator {
            height: 1px;
            background: #dee2e6;
            margin: 15px 0;
        }
        
        .points-info {
            font-size: 14px;
            font-weight: 600;
            color: #003478;
            text-align: center;
            padding: 16px;
            background: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }
        
        .control-label {
            font-size: 11px;
            color: #6c757d;
            text-align: center;
            margin-top: 8px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-section">
            <div class="video-container">
                <img id="video-feed" class="video-feed" src="/video_feed" alt="Camera Feed">
            </div>
        </div>
        
        <div class="controls-panel">
            <div class="points-info" id="points-info">
                <div style="font-size: 16px; margin-bottom: 5px;">Region Setup</div>
                Points: <span id="points-count">{{ points_count }}</span>/4
            </div>
            
            {% if not setup_complete %}
            <button class="btn btn-primary" id="confirm-btn" onclick="confirmSetup()" {% if points_count < 4 %}disabled{% endif %}>
                Confirm Setup
            </button>
            <button class="btn btn-secondary" onclick="resetPoints()">Reset Points</button>
            {% else %}
            <button class="btn btn-danger" onclick="resetSetup()">Reset Setup</button>
            {% endif %}
            
            <div class="separator"></div>
            
            <div class="threshold-group">
                <button class="btn" onclick="increaseThreshold()">+</button>
                <button class="btn" onclick="decreaseThreshold()">−</button>
            </div>
            <div class="control-label">Detection Threshold</div>
        </div>
    </div>

    <script>
        // Auto-refresh status every 1 second
        setInterval(function() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('points-count').textContent = data.points_count;
                    
                    // Update button states
                    const confirmButton = document.getElementById('confirm-btn');
                    if (confirmButton) {
                        confirmButton.disabled = data.points_count < 4;
                    }
                })
                .catch(error => {
                    console.log('Status update failed:', error);
                });
        }, 1000);
        
        function confirmSetup() {
            fetch('/confirm_setup', { method: 'POST' })
                .then(() => setTimeout(() => location.reload(), 500));
        }
        
        function resetPoints() {
            fetch('/reset_points', { method: 'POST' })
                .then(() => setTimeout(() => location.reload(), 500));
        }
        
        function resetSetup() {
            fetch('/reset_setup', { method: 'POST' })
                .then(() => setTimeout(() => location.reload(), 500));
        }
        
        function increaseThreshold() {
            fetch('/threshold/increase', { method: 'POST' });
        }
        
        function decreaseThreshold() {
            fetch('/threshold/decrease', { method: 'POST' });
        }

        // Handle clicks on video feed (for point selection)
        document.getElementById('video-feed').addEventListener('click', function(e) {
            const rect = this.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) * (640 / rect.width));
            const y = Math.round((e.clientY - rect.top) * (480 / rect.height));
            
            fetch('/click_point', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: x, y: y })
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE, 
                                  points_count=len(selected_points),
                                  setup_status=setup_complete,
                                  ref_loaded=reference_image_loaded,
                                  points=selected_points,
                                  setup_complete=setup_complete)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint for current status"""
    return {
        'points_count': len(selected_points),
        'setup_complete': setup_complete,
        'reference_loaded': reference_image_loaded,
        'points': selected_points
    }

@app.route('/click_point', methods=['POST'])
def click_point():
    """Handle point selection from web interface clicks"""
    global selected_points
    from flask import request
    
    data = request.get_json()
    if len(selected_points) < 4:
        selected_points.append((data['x'], data['y']))
        print(f"Point {len(selected_points)} selected via web: ({data['x']}, {data['y']})")
    
    return {'status': 'ok'}

@app.route('/confirm_setup', methods=['POST'])
def confirm_setup():
    """Confirm setup and start tracking"""
    global setup_complete
    if len(selected_points) == 4:
        setup_complete = True
        extract_reference_region(None, selected_points)
        print("Setup confirmed via web interface")
    return {'status': 'ok'}

@app.route('/reset_points', methods=['POST'])
def reset_points():
    """Reset selected points"""
    global selected_points
    selected_points = []
    print("Points reset via web interface")
    return {'status': 'ok'}

@app.route('/reset_setup', methods=['POST'])
def reset_setup():
    """Reset entire setup"""
    global selected_points, setup_complete
    selected_points = []
    setup_complete = False
    print("Setup reset via web interface")
    return {'status': 'ok'}

@app.route('/threshold/<action>', methods=['POST'])
def adjust_threshold(action):
    """Adjust detection threshold"""
    global global_threshold
    
    if action == 'increase':
        global_threshold = min(1.0, global_threshold + 0.05)
        print(f"Threshold increased to: {global_threshold:.2f}")
    elif action == 'decrease':
        global_threshold = max(0.0, global_threshold - 0.05)
        print(f"Threshold decreased to: {global_threshold:.2f}")
    
    return {'status': 'ok', 'threshold': global_threshold}

def run_web_mode(port=5000):
    """Run the application in web streaming mode"""
    global streaming_active
    
    print(f"=== WEB STREAMING MODE ===")
    print(f"Starting web server on http://localhost:{port}")
    print(f"Open your browser and navigate to http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Start camera capture thread
    camera_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    camera_thread.start()
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down web server...")
    finally:
        streaming_active = False
        cleanup_camera()

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='AlignVision CLI Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python cli_main.py                    # Run with OpenCV window (default)
  python cli_main.py --mode opencv      # Run with OpenCV window
  python cli_main.py --mode web         # Run web stream on localhost:5000
  python cli_main.py --mode web --port 8080  # Run web stream on localhost:8080

Controls (OpenCV mode):
  - Click 4 points to define search region
  - 'c': Confirm setup and start tracking
  - 'r': Reset points
  - '+/-': Adjust detection threshold
  - 'd': Toggle debug mode
  - 'q': Quit

Web mode:
  - Navigate to http://localhost:PORT in your browser
  - Click on the video feed to select points
  - Use the web interface controls
        '''
    )
    
    parser.add_argument('--mode', choices=['opencv', 'web'], default='opencv',
                        help='Display mode: opencv window or web stream (default: opencv)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for web stream mode (default: 5000)')
    
    args = parser.parse_args()
    
    # Load reference image first
    print("Loading reference image...")
    if not load_reference_image():
        print("Failed to load reference.jpg. Make sure the file exists in the current directory.")
        return 1
    
    print("Reference image loaded successfully!")
    
    if args.mode == 'opencv':
        run_opencv_mode()
    elif args.mode == 'web':
        run_web_mode(args.port)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())