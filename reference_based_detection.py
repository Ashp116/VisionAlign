import cv2
import numpy as np
from pypylon import pylon

class ReferenceBasedDetector:
    def __init__(self, reference_image_path):
        self.reference_img = cv2.imread(reference_image_path)
        if self.reference_img is None:
            raise ValueError(f"Could not load reference image: {reference_image_path}")
        
        self.ref_gray = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2GRAY)
        self.ref_h, self.ref_w = self.ref_gray.shape
        
        # Create ORB detector for feature matching
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.ref_gray, None)
        
        # Create SIFT detector as backup (more robust but slower)
        try:
            self.sift = cv2.SIFT_create()
            self.ref_kp_sift, self.ref_desc_sift = self.sift.detectAndCompute(self.ref_gray, None)
        except:
            self.sift = None
            self.ref_kp_sift = None
            self.ref_desc_sift = None
        
        # Create multiple scaled versions of reference
        self.ref_templates = []
        for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]:
            scaled_w = int(self.ref_w * scale)
            scaled_h = int(self.ref_h * scale)
            if scaled_w > 10 and scaled_h > 10 and scaled_w < 800 and scaled_h < 600:
                scaled_template = cv2.resize(self.ref_gray, (scaled_w, scaled_h))
                self.ref_templates.append((scaled_template, scale))
        
        print(f"Reference image loaded: {self.ref_w}x{self.ref_h}")
        print(f"ORB features: {len(self.ref_keypoints) if self.ref_keypoints else 0}")
        print(f"SIFT features: {len(self.ref_kp_sift) if self.ref_kp_sift else 0}")
        print(f"Template scales: {len(self.ref_templates)}")
        
        # Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if self.sift:
            self.bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def detect_in_region(self, frame, region_points=None):
        """
        Detect the reference object in the given frame
        region_points: optional 4-point polygon to limit search area
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Create search mask if region specified
        mask = None
        if region_points is not None:
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
            pts = np.array(region_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        best_match = None
        best_score = 0
        
        # Method 1: Multi-scale template matching
        template_results = []
        for template, scale in self.ref_templates:
            if mask is not None:
                # Apply mask to frame for template matching
                masked_frame = cv2.bitwise_and(gray_frame, mask)
                result = cv2.matchTemplate(masked_frame, template, cv2.TM_CCOEFF_NORMED)
            else:
                result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.3:  # Threshold for template matching
                # Calculate the bounding box
                top_left = max_loc
                w, h = template.shape[1], template.shape[0]
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                # Create rectangle points
                rect_pts = np.array([
                    [top_left[0], top_left[1]],
                    [bottom_right[0], top_left[1]], 
                    [bottom_right[0], bottom_right[1]],
                    [top_left[0], bottom_right[1]]
                ], dtype=np.int32)
                
                template_results.append({
                    'score': max_val,
                    'method': 'template',
                    'scale': scale,
                    'points': rect_pts,
                    'center': ((top_left[0] + bottom_right[0])//2, (top_left[1] + bottom_right[1])//2)
                })
        
        # Method 2: ORB Feature matching
        if self.ref_descriptors is not None:
            frame_kp, frame_desc = self.orb.detectAndCompute(gray_frame, mask)
            if frame_desc is not None and len(frame_kp) > 10:
                matches = self.bf.match(self.ref_descriptors, frame_desc)
                if len(matches) > 8:  # Need at least 8 matches
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = [m for m in matches if m.distance < 50]
                    
                    if len(good_matches) > 6:
                        # Get matching points
                        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # Find homography
                        if len(good_matches) >= 8:
                            M, mask_homo = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if M is not None:
                                # Transform reference corners to frame
                                corners = np.float32([[0,0], [self.ref_w,0], [self.ref_w,self.ref_h], [0,self.ref_h]]).reshape(-1,1,2)
                                transformed_corners = cv2.perspectiveTransform(corners, M)
                                
                                # Calculate score based on number of good matches
                                orb_score = len(good_matches) / len(self.ref_keypoints) 
                                
                                template_results.append({
                                    'score': orb_score,
                                    'method': 'orb',
                                    'scale': 1.0,
                                    'points': transformed_corners.reshape(-1, 2).astype(np.int32),
                                    'matches': len(good_matches)
                                })
        
        # Method 3: SIFT Feature matching (if available)
        if self.sift and self.ref_desc_sift is not None:
            frame_kp_sift, frame_desc_sift = self.sift.detectAndCompute(gray_frame, mask)
            if frame_desc_sift is not None and len(frame_kp_sift) > 10:
                matches_sift = self.bf_sift.match(self.ref_desc_sift, frame_desc_sift)
                if len(matches_sift) > 8:
                    matches_sift = sorted(matches_sift, key=lambda x: x.distance)
                    good_matches_sift = [m for m in matches_sift if m.distance < 0.7 * matches_sift[-1].distance]
                    
                    if len(good_matches_sift) > 6:
                        src_pts = np.float32([self.ref_kp_sift[m.queryIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)
                        dst_pts = np.float32([frame_kp_sift[m.trainIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)
                        
                        if len(good_matches_sift) >= 8:
                            M, mask_homo = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if M is not None:
                                corners = np.float32([[0,0], [self.ref_w,0], [self.ref_w,self.ref_h], [0,self.ref_h]]).reshape(-1,1,2)
                                transformed_corners = cv2.perspectiveTransform(corners, M)
                                
                                sift_score = len(good_matches_sift) / len(self.ref_kp_sift)
                                
                                template_results.append({
                                    'score': sift_score * 1.2,  # SIFT is more reliable, give it higher weight
                                    'method': 'sift',
                                    'scale': 1.0,
                                    'points': transformed_corners.reshape(-1, 2).astype(np.int32),
                                    'matches': len(good_matches_sift)
                                })
        
        # Choose the best result
        if template_results:
            best_result = max(template_results, key=lambda x: x['score'])
            return best_result
        
        return None

def main():
    # Initialize detector with reference image
    detector = ReferenceBasedDetector("reference.jpg")
    
    # Setup Basler camera
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
    
    # UI variables
    selected_points = []
    region_defined = False
    detection_threshold = 0.3
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_points, region_defined
        if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
            selected_points.append((x, y))
            print(f"Point {len(selected_points)} selected: ({x}, {y})")
            if len(selected_points) == 4:
                region_defined = True
                print("Region defined! Detection will be limited to this area.")
    
    cv2.namedWindow("Reference-Based Detection", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Reference-Based Detection", mouse_callback)
    
    print("=== Reference-Based Object Detection ===")
    print("Optional: Click 4 points to define search region")
    print("Press 'r' to reset region")
    print("Press 'c' to clear region (search entire frame)")
    print("Press '+'/'-' to adjust detection threshold")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            frame = converter.Convert(grab_result).GetArray()
            frame_resized = cv2.resize(frame, (640, 480))
            frame_count += 1
            
            # Detect object
            search_region = selected_points if region_defined else None
            result = detector.detect_in_region(frame_resized, search_region)
            
            # Draw results
            display_frame = frame_resized.copy()
            
            # Draw search region if defined
            if region_defined:
                pts = np.array(selected_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts], (255, 255, 0))
                cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)
            
            # Draw points being selected
            for i, point in enumerate(selected_points):
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw detection result
            if result and result['score'] > detection_threshold:
                points = result['points']
                cv2.polylines(display_frame, [points], True, (0, 255, 0), 3)
                
                # Draw corner points
                for pt in points:
                    cv2.circle(display_frame, tuple(pt), 5, (0, 255, 0), -1)
                
                # Show detection info
                cv2.putText(display_frame, f"DETECTED: {result['method'].upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Score: {result['score']:.3f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if 'matches' in result:
                    cv2.putText(display_frame, f"Matches: {result['matches']}", (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show status info
            cv2.putText(display_frame, f"Threshold: {detection_threshold:.2f}", (10, display_frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Region: {'Defined' if region_defined else 'Full Frame'}", (10, display_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Reference-Based Detection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                selected_points = []
                region_defined = False
                print("Region reset")
            elif key == ord('c'):
                selected_points = []
                region_defined = False
                print("Cleared region - searching full frame")
            elif key == ord('+') or key == ord('='):
                detection_threshold = min(1.0, detection_threshold + 0.05)
                print(f"Threshold: {detection_threshold:.2f}")
            elif key == ord('-'):
                detection_threshold = max(0.0, detection_threshold - 0.05)
                print(f"Threshold: {detection_threshold:.2f}")
            elif key == ord('s') and result:
                # Save debug image
                timestamp = cv2.getTickCount()
                cv2.imwrite(f"detection_result_{timestamp}.jpg", display_frame)
                print(f"Saved: detection_result_{timestamp}.jpg")
        
        grab_result.Release()
    
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()