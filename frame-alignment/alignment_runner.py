"""
alignment_runner.py

This is the alignment guidance runner. It contains the detection loop and focuses on 
bounding box visualization without overlay functionality.
"""
import argparse
import time
import cv2
import numpy as np
from ae_hud import analyze, draw_metrics_hud, AutoExposureController, compute_metrics

try:
    from pypylon import pylon
except Exception:
    pylon = None

# Match thresholds to live-compare values
MATCH_THRESHOLD = 0.15  # Detection match threshold
DETECTION_SKIP = 2

# Removed overlay-related variables since we're not using them anymore

selected_points = []
setup_complete = False

def mouse_callback(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append((x,y))
        print(f"Point {len(selected_points)} selected: ({x},{y})")

def load_reference_image(path):
    ref_image = cv2.imread(path)
    if ref_image is None:
        print(f"ERROR: Could not load reference '{path}'")
        return None
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = ref_gray.shape
    orb = cv2.ORB_create(nfeatures=1000)
    kp, desc = orb.detectAndCompute(ref_gray, None)
    templates = []
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
        sw = int(ref_w * scale); sh = int(ref_h * scale)
        if sw>10 and sh>10:
            templates.append((cv2.resize(ref_gray, (sw, sh)), scale))
    return {'image': ref_image, 'gray': ref_gray, 'kp': kp, 'desc': desc, 'templates': templates}

def compare_rect(ref_gray, ref_kp, ref_desc, ref_templates, candidate_gray):
    try:
        best_template_score = 0.0
        for template, _ in ref_templates:
            if candidate_gray.shape[0] >= template.shape[0] and candidate_gray.shape[1] >= template.shape[1]:
                res = cv2.matchTemplate(candidate_gray, template, cv2.TM_CCOEFF_NORMED)
                maxv = float(np.max(res))
                best_template_score = max(best_template_score, maxv)
        if candidate_gray.shape[0] > 20 and candidate_gray.shape[1] > 20:
            resized_ref = cv2.resize(ref_gray, (candidate_gray.shape[1], candidate_gray.shape[0]))
            direct = float(cv2.matchTemplate(candidate_gray, resized_ref, cv2.TM_CCOEFF_NORMED)[0,0])
            best_template_score = max(best_template_score, direct)
        feature_score = 0.0
        if ref_desc is not None and len(ref_kp) > 5:
            orb = cv2.ORB_create(nfeatures=500)
            kp2, desc2 = orb.detectAndCompute(candidate_gray, None)
            if desc2 is not None and len(kp2) > 3:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(ref_desc, desc2)
                if matches:
                    matches = sorted(matches, key=lambda x: x.distance)
                    good = [m for m in matches if m.distance < 60]
                    if len(good) > 0:
                        feature_score = min(1.0, len(good) / 50.0)
        ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0,256])
        cand_hist = cv2.calcHist([candidate_gray], [0], None, [256], [0,256])
        hist_score = float(cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_CORREL))
        final = 0.6*best_template_score + 0.3*feature_score + 0.1*max(0.0, hist_score)
        return final
    except Exception as e:
        print(f"Comparison error: {e}")
        return 0.0

def is_rect_in_polygon(rect_points, polygon_points, margin_percent=5):
    polygon = np.array(polygon_points, dtype=np.int32)
    center_x = np.mean(polygon[:,0]); center_y = np.mean(polygon[:,1])
    expanded = []
    for p in polygon:
        dx = p[0]-center_x; dy = p[1]-center_y
        expanded.append([int(center_x + dx*(1+margin_percent/100.0)), int(center_y + dy*(1+margin_percent/100.0))])
    expanded = np.array(expanded, dtype=np.int32)
    inside = 0
    for pt in rect_points:
        if cv2.pointPolygonTest(expanded, (int(pt[0]), int(pt[1])), False) >= 0:
            inside += 1
    return inside >= 2

def compute_metrics_wrapper(master, live):
    """Wrapper around compute_metrics to handle potential errors"""
    try:
        return compute_metrics(master, live)
    except Exception as e:
        print(f"[WARN] Metrics computation failed: {e}")
        return {'sev_light': 'OK', 'mean_delta': 0.0, 'clip_pp': 0.0}
    
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', required=True)
    ap.add_argument('--basler', action='store_true')
    ap.add_argument('--auto-exposure', action='store_true')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--ae-step', type=float, default=0.12)
    ap.add_argument('--ae-interval', type=float, default=1.0)
    args = ap.parse_args(args=argv)

    global selected_points, setup_complete
    selected_points = []
    setup_complete = False
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    master = cv2.imread(args.master)
    if master is None:
        print(f"Cannot read MASTER '{args.master}'")
        return
    Hm, Wm = master.shape[:2]
    M = analyze(master)

    use_basler = False
    cap = None; camera = None; converter = None
    if args.basler and pylon is not None:
        try:
            tlf = pylon.TlFactory.GetInstance()
            devs = tlf.EnumerateDevices()
            if not devs: raise RuntimeError('No Basler devices')
            camera = pylon.InstantCamera(tlf.CreateDevice(devs[0]))
            camera.Open()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            use_basler = True
            print('[CAM] Basler connected')
        except Exception as e:
            print(f'[WARN] Basler init failed ({e}); falling back to OpenCV')

    if not use_basler:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print('[ERR] Cannot open OpenCV camera')
            return
        print(f'[CAM] OpenCV index {args.camera}')

    ae = None
    if args.auto_exposure:
        if use_basler and camera is not None:
            min_exp=-1e9; max_exp=1e9
            try:
                if hasattr(camera, 'ExposureTimeMin'): min_exp = float(camera.ExposureTimeMin.GetValue())
                if hasattr(camera, 'ExposureTimeMax'): max_exp = float(camera.ExposureTimeMax.GetValue())
            except Exception:
                pass
            ae = AutoExposureController(camera=camera, use_basler=True, min_exposure=min_exp, max_exposure=max_exp, 
                                      step_pct=float(args.ae_step), min_interval_s=float(args.ae_interval))
        else:
            ae = AutoExposureController(cap=cap, use_basler=False, step_pct=float(args.ae_step), 
                                      min_interval_s=float(args.ae_interval))
        print('[AE] Auto-exposure enabled')

    ref = load_reference_image(args.master)
    if ref is None:
        print('Failed to load reference - continuing but matching will be limited')

    win = 'Alignment Runner'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_callback)

    frame_counter = 0
    best_pts = None; best_score = -1.0
    best_outside_pts = None; best_outside_score = -1.0
    best_bbox_pts = None; best_outside_bbox_pts = None

    while True:
        if use_basler:
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                grab.Release(); continue
            frame = converter.Convert(grab).GetArray(); grab.Release()
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

        if frame.shape[:2] != (Hm, Wm):
            frame = cv2.resize(frame, (Wm, Hm), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_counter += 1

        if not setup_complete:
            display = frame.copy()
            if len(selected_points) == 4:
                pts = np.array(selected_points, np.int32).reshape((-1,1,2))
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (255,255,0))
                cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
                cv2.polylines(display, [pts], True, (255,255,0), 2)
            for i,p in enumerate(selected_points):
                cv2.circle(display, p, 5, (0,0,255), -1)
                cv2.putText(display, str(i+1), (p[0]+6,p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            cv2.putText(display, f'Points: {len(selected_points)}/4 (click to add)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
            cv2.putText(display, 'Press c to confirm region, r to reset', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),1)
            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c') and len(selected_points) == 4:
                print('Region confirmed, starting tracking')
                setup_complete = True
            if key == ord('r'):
                selected_points = []
                print('Points reset')
            continue

        if frame_counter % DETECTION_SKIP == 0:
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            edges = cv2.Canny(blur, 30, 120)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            all_candidates = []
            outside_candidates = []

            region_pts = selected_points if len(selected_points)==4 else None
            region_center = None
            if region_pts is not None:
                region_center = (int(np.mean([p[0] for p in region_pts])), int(np.mean([p[1] for p in region_pts])))

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200: continue
                peri = cv2.arcLength(cnt, True)
                for eps in [0.02, 0.03, 0.04]:
                    approx = cv2.approxPolyDP(cnt, eps*peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        rect = approx.reshape((4,2))
                        x,y,w,h = cv2.boundingRect(rect)
                        if w < 20 or h < 20: continue
                        cand = gray[y:y+h, x:x+w]
                        score = 0.0
                        if ref is not None:
                            score = compare_rect(ref['gray'], ref['kp'], ref['desc'], ref['templates'], cand)
                        inside = False
                        if region_pts is not None:
                            inside = is_rect_in_polygon(rect.tolist(), region_pts)
                        if inside:
                            all_candidates.append((rect.copy(), (x,y,w,h), score))
                        else:
                            outside_candidates.append((rect.copy(), (x,y,w,h), score))
                        break

            if all_candidates:
                all_candidates.sort(key=lambda x: x[2], reverse=True)
                best_pts, best_bbox, best_score = all_candidates[0]
                bx,by,bw,bh = best_bbox
                best_bbox_pts = np.array([[bx,by],[bx+bw,by],[bx+bw,by+bh],[bx,by+bh]])
            else:
                best_pts = None; best_score = -1.0; best_bbox_pts = None

            if outside_candidates:
                outside_candidates.sort(key=lambda x: x[2], reverse=True)
                best_outside_pts, best_outside_bbox, best_outside_score = outside_candidates[0]
                obx,oby,obw,obh = best_outside_bbox
                best_outside_bbox_pts = np.array([[obx,oby],[obx+obw,oby],[obx+obw,oby+obh],[obx,oby+obh]])
            else:
                best_outside_pts = None; best_outside_score = -1.0; best_outside_bbox_pts = None

            # AE logic
            if ae is not None and ref is not None:
                info = compute_metrics_wrapper(M, analyze(frame))
                if info['sev_light'] in ('WARN','CRIT'):
                    acted = False
                    if info['clip_pp'] > 0:
                        acted = ae.step_exposure(direction=-1)
                        if acted: print(f"[AE] Reduced exposure due to clipping (clip_pp={info['clip_pp']:.2f})")
                    if not acted and info['mean_delta'] < -10.0:
                        acted = ae.step_exposure(direction=1)
                        if acted: print(f"[AE] Increased exposure (mean_delta={info['mean_delta']:+.2f})")

        out = frame.copy()

        if len(selected_points)==4:
            pts = np.array(selected_points, np.int32).reshape((-1,1,2))
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], (255,255,0))
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
            cv2.polylines(out, [pts], True, (255,255,0), 2)

        if best_pts is not None and best_score >= MATCH_THRESHOLD:
            cv2.polylines(out, [best_pts.astype(np.int32)], True, (0,255,0), 3)
            cv2.putText(out, f"DETECTED IN REGION ({best_score:.3f})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        elif best_pts is not None and best_bbox_pts is not None:
            cv2.polylines(out, [best_bbox_pts.astype(np.int32)], True, (0,165,255), 2)
            cv2.putText(out, f"POSSIBLE IN REGION ({best_score:.3f})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255),2)

        if best_outside_pts is not None and best_outside_score >= MATCH_THRESHOLD:
            cv2.polylines(out, [best_outside_pts.astype(np.int32)], True, (255,0,255), 3)
            cv2.putText(out, f"DETECTED OUTSIDE ({best_outside_score:.3f})", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255),2)
            if region_center is not None and best_outside_bbox_pts is not None:
                bx = int(np.mean(best_outside_bbox_pts[:,0]))
                by = int(np.mean(best_outside_bbox_pts[:,1]))
                cv2.arrowedLine(out, (bx,by), region_center, (255,0,255), 3, tipLength=0.2)
        elif best_outside_pts is not None and best_outside_bbox_pts is not None:
            cv2.polylines(out, [best_outside_bbox_pts.astype(np.int32)], True, (128,0,128), 2)
            cv2.putText(out, f"POSSIBLE OUTSIDE ({best_outside_score:.3f})", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,0,128),2)

        # Draw metrics HUD
        draw_metrics_hud(out, M, analyze(out))

        cv2.putText(out, "[s] save   [q] quit", (16,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"alignment_snapshot_{ts}.jpg", out)
            print(f"Saved alignment_snapshot_{ts}.jpg")

    if use_basler and camera is not None:
        camera.StopGrabbing()
        camera.Close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()