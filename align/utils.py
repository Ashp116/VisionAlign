import math
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---------- Geometry helpers ----------

def build_similarity_affine(h, w, dx, dy, theta, rho):
    """
    Build a 2x3 affine matrix (for cv2.warpAffine) representing similarity transform:
    - rotation by theta (radians) and scale s = exp(rho) around the image center
    - then translation by (dx, dy) in pixels
    The matrix maps CURRENT -> MASTER (apply to current image to align to master).
    """
    s = math.exp(rho)
    c, s_ = math.cos(theta) * s, math.sin(theta) * s

    # rotate+scale about center
    cx, cy = w * 0.5, h * 0.5
    R = np.array([[c, -s_, (1 - c) * cx + s_ * cy],
                  [s_,  c, (1 - c) * cy - s_ * cx]], dtype=np.float32)
    # then translate
    R[:, 2] += np.array([dx, dy], dtype=np.float32)
    return R  # 2x3


def warp_similarity(img, dx, dy, theta, rho, out_size=None):
    if out_size is None:
        h, w = img.shape[:2]
    else:
        h, w = out_size[1], out_size[0]
    M = build_similarity_affine(h, w, dx, dy, theta, rho)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------- Nuisance / augmentation ----------

def add_nuisance(img_bgr):
    img = img_bgr.copy()
    # brightness, contrast
    alpha = np.random.uniform(0.8, 1.25)  # contrast
    beta  = np.random.uniform(-20, 20)    # brightness shift
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # gaussian blur (sometimes)
    if np.random.rand() < 0.35:
        k = np.random.choice([3,5,7])
        img = cv2.GaussianBlur(img, (k,k), 0)

    # JPEG-like noise via re-encode (simulate compression)
    if np.random.rand() < 0.4:
        q = np.random.randint(55, 90)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, enc = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # random occlusion
    if np.random.rand() < 0.25:
        h, w = img.shape[:2]
        for _ in range(np.random.randint(1, 3)):
            rw, rh = np.random.randint(w//10, w//5), np.random.randint(h//10, h//5)
            rx, ry = np.random.randint(0, w - rw), np.random.randint(0, h - rh)
            cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (0,0,0), thickness=-1)

    return img


# ---------- Quality metrics ----------

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def rms_contrast(gray):
    return float(gray.std())

def brightness(gray):
    return float(gray.mean())

def ssim_score(grayA, grayB):
    # clip to same size just in case
    h = min(grayA.shape[0], grayB.shape[0])
    w = min(grayA.shape[1], grayB.shape[1])
    if grayA.shape[:2] != (h,w):
        grayA = grayA[:h,:w]
    if grayB.shape[:2] != (h,w):
        grayB = grayB[:h,:w]
    return float(ssim(grayA, grayB, data_range=255))
