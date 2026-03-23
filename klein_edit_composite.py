"""
Klein Edit Composite Node 2bc3600 + Poisson Blending for seamless lighting transfers
====================================================================================
"""

import numpy as np
import torch
import cv2
from PIL import Image
import math


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def _create_color_wheel():
    """Create a color wheel for flow visualization."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    col = 0

    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY

    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC

    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM

    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


COLORWHEEL = _create_color_wheel()


def _flow_to_color(flow, max_flow=None):
    """Convert optical flow to RGB color image."""
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, -u) / np.pi

    if max_flow is None:
        max_flow = np.percentile(mag, 99)

    if max_flow > 0:
        mag = np.clip(mag * 8 / max_flow, 0, 8)

    angle = (angle + 1) / 2
    fk = (angle * (COLORWHEEL.shape[0] - 1) + 0.5).astype(np.int32)
    fk = np.clip(fk, 0, COLORWHEEL.shape[0] - 1)

    color = COLORWHEEL[fk]

    # Saturation based on magnitude
    mag = np.clip(mag, 0, 1)
    color = (1 - mag[..., np.newaxis]) * 255 + mag[..., np.newaxis] * color

    return color.astype(np.uint8)


def _apply_heatmap(img_float, mask_float, colormap=cv2.COLORMAP_JET):
    """Overlay a heatmap on top of an image."""
    mask_u8 = np.clip(mask_float * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha = 0.6
    return (1 - alpha) * img_float + alpha * heatmap


def _draw_sift_matches(gray_orig, gray_gen, kp1, kp2, matches, inlier_mask=None):
    """Draw SIFT feature matches between images."""
    if len(kp1) == 0 or len(kp2) == 0 or len(matches) == 0:
        h = max(gray_orig.shape[0], gray_gen.shape[0])
        w = gray_orig.shape[1] + gray_gen.shape[1]
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[:gray_orig.shape[0], :gray_orig.shape[1]] = gray_orig
        canvas[:gray_gen.shape[0], gray_orig.shape[1]:] = gray_gen
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    h1, w1 = gray_orig.shape[:2]
    h2, w2 = gray_gen.shape[:2]
    h_max = max(h1, h2)
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1, :] = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2RGB)
    canvas[:h2, w1:w1+w2, :] = cv2.cvtColor(gray_gen, cv2.COLOR_GRAY2RGB)

    kp2_shifted = [cv2.KeyPoint(p.pt[0] + w1, p.pt[1], p.size) for p in kp2]

    inlier_set = set()
    if inlier_mask is not None:
        inlier_set = set(np.where(inlier_mask.ravel())[0])

    for i, m in enumerate(matches):
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2_shifted[m.trainIdx].pt[0]), int(kp2_shifted[m.trainIdx].pt[1]))

        if inlier_mask is None:
            color = (128, 128, 128)
        else:
            color = (0, 255, 0) if i in inlier_set else (0, 0, 255)

        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)

    if inlier_mask is not None:
        n_inliers = int(inlier_mask.sum())
        text = f"Matches: {len(matches)} | Inliers: {n_inliers}"
        cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return canvas.astype(np.float32) / 255.0


def _create_side_by_side(img1, img2, labels=("Original", "Generated")):
    """Create side-by-side comparison with labels."""
    h, w = img1.shape[:2]
    canvas = np.zeros((h + 40, w * 2, 3), dtype=np.float32)
    canvas[40:, :w] = img1
    canvas[40:, w:] = img2

    cv2.putText(canvas, labels[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (1, 1, 1), 2)
    cv2.putText(canvas, labels[1], (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (1, 1, 1), 2)
    return canvas


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _diag(H: int, W: int) -> float:
    return math.sqrt(H * H + W * W)


def _pct_to_px(pct: float, diag: float) -> int:
    return max(0, round(abs(pct) * diag / 100.0))


def _blur_kernel_for_diag(diag: float) -> tuple:
    k = max(3, int(round(diag / 724.0 * 3)))
    if k % 2 == 0:
        k += 1
    return (k, k)


# ---------------------------------------------------------------------------
# LAB conversion & Advanced Diffing
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized D65 RGB to LAB conversion natively supporting any shape."""
    lin = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    M = np.array([[0.4124564, 0.3575761, 0.1804375],[0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    
    xyz = np.dot(lin, M.T) / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    def f(t):
        return np.where(t > (6 / 29) ** 3, t ** (1 / 3), t / (3 * (6 / 29) ** 2) + 4 / 29)

    fx = f(xyz[..., 0])
    fy = f(xyz[..., 1])
    fz = f(xyz[..., 2])
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1).astype(np.float32)

def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Perfectly invert D65 LAB back to standard sRGB."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (b / 200.0)

    delta = 6.0 / 29.0
    
    def f_inv(t):
        return np.where(t > delta, t ** 3, 3.0 * (delta ** 2) * (t - 4.0 / 29.0))

    # Convert back to XYZ
    x = f_inv(fx) * 0.95047
    y = f_inv(fy) * 1.00000
    z = f_inv(fz) * 1.08883
    xyz = np.stack([x, y, z], axis=-1)

    # Inverse transformation matrix for sRGB D65
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],[-0.9692660,  1.8760108,  0.0415560],[ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=np.float32)

    lin = np.dot(xyz, M_inv.T)
    
    # Clip negative linear values to avoid NaN during fractional power conversion
    lin = np.clip(lin, 0.0, None)
    
    # Back to standard sRGB Gamma
    rgb = np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * (lin ** (1.0 / 2.4)) - 0.055
    )
    
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _apply_color_match(orig_rgb: np.ndarray, gen_rgb: np.ndarray, 
                       composite_mask: np.ndarray, valid_mask: np.ndarray, 
                       blend_strength: float) -> tuple:
    """Matches the color/lighting of gen_rgb to orig_rgb strictly using the background."""
    if blend_strength <= 0.0:
        return gen_rgb, False
        
    # Isolate pixels that are purely background AND valid after homography warping
    bg_mask = (composite_mask < 0.05) & (valid_mask > 0.5)
    
    if bg_mask.sum() < 100:
        return gen_rgb, False  # Not enough background to get reliable statistics
        
    orig_lab = _rgb_to_lab(orig_rgb)
    gen_lab = _rgb_to_lab(gen_rgb)
    
    # Calculate Reinhard statistics (mean and std deviation)
    orig_mean = orig_lab[bg_mask].mean(axis=0)
    orig_std  = orig_lab[bg_mask].std(axis=0) + 1e-5
    
    gen_mean  = gen_lab[bg_mask].mean(axis=0)
    gen_std   = gen_lab[bg_mask].std(axis=0) + 1e-5
    
    # Apply standard deviation and mean shift to the ENTIRE image
    matched_lab = ((gen_lab - gen_mean) / gen_std) * orig_std + orig_mean
    matched_rgb = _lab_to_rgb(matched_lab)
    
    # Alpha blend based on user preference
    blended_rgb = (matched_rgb * blend_strength) + (gen_rgb * (1.0 - blend_strength))
    return np.clip(blended_rgb, 0.0, 1.0), True

def _compute_diff_map(orig_np: np.ndarray, gen_np: np.ndarray, blur_kernel: tuple) -> np.ndarray:
    """
    Computes a hybrid Color (Delta E) and Structural (Gradient) difference map.
    This suppresses global lighting shifts while severely penalizing shape/content changes.
    """
    o_blur = cv2.GaussianBlur(orig_np, blur_kernel, 0)
    g_blur = cv2.GaussianBlur(gen_np, blur_kernel, 0)

    # --- 1. Color Difference ---
    o_lab = _rgb_to_lab(o_blur)
    g_lab = _rgb_to_lab(g_blur)
    
    diff_lab = o_lab - g_lab
    diff_lab[..., 0] *= 0.5  # Ignore most global lighting (L) shifts
    diff_lab[..., 1] *= 1.2  # Slightly boost a, b channels to catch hue changes
    diff_lab[..., 2] *= 1.2
    color_diff = np.sqrt(np.sum(diff_lab**2, axis=-1))

    # --- 2. Structural Difference ---
    o_gray = cv2.cvtColor((o_blur * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g_gray = cv2.cvtColor((g_blur * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    o_gx = cv2.Sobel(o_gray, cv2.CV_32F, 1, 0, ksize=3)
    o_gy = cv2.Sobel(o_gray, cv2.CV_32F, 0, 1, ksize=3)
    g_gx = cv2.Sobel(g_gray, cv2.CV_32F, 1, 0, ksize=3)
    g_gy = cv2.Sobel(g_gray, cv2.CV_32F, 0, 1, ksize=3)

    o_mag = np.sqrt(o_gx**2 + o_gy**2)
    g_mag = np.sqrt(g_gx**2 + g_gy**2)

    # Scale gradients heavily to project structural changes into the Delta E value scale
    struct_diff = np.abs(o_mag - g_mag) * 40.0

    return color_diff + struct_diff


# ---------------------------------------------------------------------------
# Poisson Blending Helper
# ---------------------------------------------------------------------------

def _seamless_blend(orig_float: np.ndarray, gen_float: np.ndarray, mask_float: np.ndarray) -> np.ndarray:
    """
    Uses Poisson Blending (cv2.seamlessClone) to match lighting/color across boundaries,
    then applies an Alpha Blend using the smoothed mask for perfect edges.
    """
    orig_u8 = (np.clip(orig_float, 0, 1) * 255).astype(np.uint8)
    gen_u8  = (np.clip(gen_float, 0, 1) * 255).astype(np.uint8)

    # Create binary mask for Poisson (> 0.1 to encompass the internal feathering zone)
    binary_mask = (mask_float > 0.1).astype(np.uint8) * 255

    # Avoid OpenCV crash: the mask should never touch the exact 1px image boundaries
    binary_mask[0, :] = 0; binary_mask[-1, :] = 0
    binary_mask[:, 0] = 0; binary_mask[:, -1] = 0

    # FIX FOR PIXEL SHIFTING:
    # We must calculate the center using cv2.boundingRect exactly as cv2.seamlessClone
    # calculates it internally. Using numpy min/max math causes a 1-pixel shift mismatch.
    x, y, w, h = cv2.boundingRect(binary_mask)
    m3 = mask_float[..., np.newaxis]

    # If the mask is empty, return standard Alpha blend
    if w == 0 or h == 0:
        return np.clip(orig_float * (1.0 - m3) + gen_float * m3, 0, 1)

    # Calculate the exact mathematical center expected by OpenCV
    center = (x + w // 2, y + h // 2)

    try:
        # NORMAL_CLONE mathematically matches the internal gradients to the background (orig)
        cloned_u8 = cv2.seamlessClone(gen_u8, orig_u8, binary_mask, center, cv2.NORMAL_CLONE)
        cloned_float = cloned_u8.astype(np.float32) / 255.0

        # Final Fusion: Alpha blend between original and the lighting-corrected Poisson clone
        return np.clip(orig_float * (1.0 - m3) + cloned_float * m3, 0, 1)
    except Exception:
        # Safety fallback to classic Alpha Blend if Poisson solver fails
        return np.clip(orig_float * (1.0 - m3) + gen_float * m3, 0, 1)


# ---------------------------------------------------------------------------
# SIFT pre-alignment
# ---------------------------------------------------------------------------

def _sift_prealign(gray_orig: np.ndarray, gray_gen: np.ndarray,
                   gen_float: np.ndarray, orig_mask: np.ndarray = None,
                   debug: bool = False) -> tuple:
    """Full Perspective (Homography) alignment of gen → orig using SIFT + CLAHE + MAGSAC."""
    H, W = gray_orig.shape[:2]
    debug_dict = {} if debug else None

    # Apply CLAHE to dramatically boost SIFT robustness against heavy lighting/style edits
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_orig = clahe.apply(gray_orig)
    eq_gen = clahe.apply(gray_gen)

    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.03)
    kp1, des1 = sift.detectAndCompute(eq_orig, mask=orig_mask)
    kp2, des2 = sift.detectAndCompute(eq_gen, mask=None)

    if debug:
        debug_dict['orig_kp'] = len(kp1) if kp1 else 0
        debug_dict['gen_kp']  = len(kp2) if kp2 else 0

    fallback_valid = np.ones((H, W), dtype=np.float32)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        if debug:
            debug_dict['match_viz'] = _draw_sift_matches(eq_orig, eq_gen, kp1 or [], kp2 or[],[])
        return gen_float, None, 0, fallback_valid, debug_dict

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=80),
    )
    raw_matches = flann.knnMatch(des1, des2, k=2)
    
    # Relaxed ratio test to capture more candidates (MAGSAC will filter outliers)
    good =[]
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if debug:
        debug_dict['matches_before_ransac'] = _draw_sift_matches(eq_orig, eq_gen, kp1, kp2, good, None)

    if len(good) < 10:
        if debug:
            debug_dict['match_viz'] = _draw_sift_matches(eq_orig, eq_gen, kp1, kp2, good, None)
        return gen_float, None, 0, fallback_valid, debug_dict

    pts_orig = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_gen  = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    ransac_method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
    H_mat, inlier_mask = cv2.findHomography(pts_gen, pts_orig, ransac_method, 4.0)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if debug:
        debug_dict['match_viz'] = _draw_sift_matches(eq_orig, eq_gen, kp1, kp2, good, inlier_mask)
        debug_dict['n_matches'] = len(good)
        debug_dict['n_inliers'] = n_inliers

    if H_mat is None:
        return gen_float, None, 0, fallback_valid, debug_dict

    det = H_mat[0, 0] * H_mat[1, 1] - H_mat[0, 1] * H_mat[1, 0]
    if not (0.2 < det < 5.0):
        if debug:
            debug_dict['failure_reason'] = f"Bad determinant: {det:.2f} (needs 0.2-5.0)"
        return gen_float, None, 0, fallback_valid, debug_dict

    aligned = cv2.warpPerspective(
        gen_float, H_mat, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    valid_mask = cv2.warpPerspective(
        np.ones((H, W), dtype=np.float32), H_mat, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if debug:
        debug_dict['homography']  = H_mat
        debug_dict['determinant'] = det

    return aligned, H_mat, n_inliers, valid_mask, debug_dict


# ---------------------------------------------------------------------------
# Optical flow & Morphology helpers
# ---------------------------------------------------------------------------

def _dis_flow(gray_a: np.ndarray, gray_b: np.ndarray, preset: int) -> np.ndarray:
    """Computes DIS flow and median blurs the vector field to prevent extreme tearing."""
    flow = cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)
    flow[..., 0] = cv2.medianBlur(flow[..., 0], 5)
    flow[..., 1] = cv2.medianBlur(flow[..., 1], 5)
    return flow


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)


def _fwd_bwd_error(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    return np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)


def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    return (_fwd_bwd_error(flow_fwd, flow_bwd) > threshold).astype(np.float32)


def _grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    if grow_px == 0:
        return mask
    radius = abs(grow_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.MORPH_DILATE if grow_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), op, k).astype(np.float32)


def _open_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Remove speckles via opening-by-reconstruction.

    Standard opening (erode then dilate) kills isolated noise but also rounds off
    sharp points and thin protrusions on real regions.  Opening by reconstruction
    fixes this: after erosion the seed is geodesically dilated — it can only grow
    back within the bounds of the original mask.  Any region that survived erosion
    is fully restored to its original shape; any region that was completely erased
    (i.e. genuine speckle) cannot recover and stays gone.
    """
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))
    marker = cv2.erode(mask.astype(np.uint8), k).astype(np.float32)  # seed: speckles are gone
    orig   = mask.astype(np.float32)                                   # ceiling: can't grow beyond original
    k3     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))     # unit dilation step
    while True:
        expanded = np.minimum(cv2.dilate(marker, k3), orig)  # dilate then clip to original
        if np.array_equal(expanded, marker):                  # converged — no further change
            break
        marker = expanded
    return marker


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    h, w = binary.shape
    n, labeled = cv2.connectedComponents(binary, connectivity=8)
    result = binary.copy()
    for island_id in range(1, n):
        island = (labeled == island_id).astype(np.uint8)
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:h+1, 1:w+1] = island
        inv = 1 - padded
        cv2.floodFill(inv, None, (0, 0), 0)
        interior = inv[1:h+1, 1:w+1]
        result = np.clip(result + interior, 0, 1).astype(np.uint8)
    return result.astype(np.float32)


def _bleed_mask(mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Extrapolates the mask into invalid border regions (where the generated image shrunk).
    This prevents the original image from showing through as a thin frame around edits.
    """
    invalid = (valid_mask < 0.5).astype(np.uint8)
    if invalid.max() == 0:
        return mask
        
    dist = cv2.distanceTransform(invalid, cv2.DIST_L2, 3)
    max_depth = int(np.ceil(np.max(dist)))
    
    if max_depth == 0:
        return mask
        
    bled_mask = mask.copy()
    remaining = max_depth
    # Iterative dilation guarantees we reach the edge without massive slow kernels
    while remaining > 0:
        step = min(remaining, 60)
        k_size = step * 2 + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        bled_mask = cv2.dilate(bled_mask, k)
        remaining -= step
        
    # Apply the bled values ONLY in the void regions
    return np.where(valid_mask > 0.5, mask, bled_mask).astype(np.float32)


# ---------------------------------------------------------------------------
# Auto-tuning
# ---------------------------------------------------------------------------

def _auto_threshold_mad(diff_map: np.ndarray, valid_mask: np.ndarray = None, 
                        k: float = 6.0, min_t: float = 3.0, max_t: float = 60.0) -> float:
    """Computes a highly robust threshold based on Median Absolute Deviation (MAD)."""
    if valid_mask is not None and valid_mask.sum() > 100:
        sample = diff_map[valid_mask > 0.5]
    else:
        sample = diff_map.flatten()
        
    if len(sample) > 50000:
        sample = np.random.choice(sample, 50000, replace=False)
        
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    mad = max(mad, 0.5)  # Enforce minimum noise floor
    
    threshold = med + k * mad
    return float(np.clip(threshold, min_t, max_t))


def _local_threshold_map(diff_map: np.ndarray, valid_mask: np.ndarray,
                         diag: float, k: float = 6.0,
                         min_t: float = 3.0, max_t: float = 60.0) -> np.ndarray:
    """
    Computes a per-pixel adaptive threshold map using local MAD statistics.

    On smooth surfaces gradients are near zero so the global MAD threshold (driven
    by high-activity regions elsewhere) is too coarse to detect subtle changes.
    A large local window captures the neighbourhood statistics instead, producing a
    lower threshold in flat/smooth regions while leaving textured regions unaffected.

    Window size is set to ~12.5% of the image diagonal — large enough to gather
    robust statistics but small enough to stay local.
    """
    # Window radius ~6% of diagonal, must be odd
    r = max(15, int(round(diag * 0.06)))
    if r % 2 == 0:
        r += 1
    ksize = (r, r)

    d = diff_map.astype(np.float32)

    # Local mean via box filter
    local_mean = cv2.blur(d, ksize)
    # Local MAD approximated as: blur(|d - local_mean|)
    local_mad  = cv2.blur(np.abs(d - local_mean), ksize)
    local_mad  = np.maximum(local_mad, 0.5)  # same noise floor as global version

    local_thresh = local_mean + k * local_mad
    local_thresh = np.clip(local_thresh, min_t, max_t).astype(np.float32)

    # Only apply inside valid region; outside just use max_t (won't be composited anyway)
    if valid_mask is not None:
        local_thresh = np.where(valid_mask > 0.5, local_thresh, max_t)

    return local_thresh


# ---------------------------------------------------------------------------
# Core Composite Logic
# ---------------------------------------------------------------------------

FLOW_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast":      cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium":    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}


def _composite(original_np: np.ndarray,
               generated_np: np.ndarray,
               delta_e_threshold: float,
               flow_preset: int,
               occlusion_threshold: float,
               grow_px: int,
               close_radius: int,
               feather_px: float,
               color_match_blend: float,
               noise_removal_px: int = 0,
               max_islands: int = 0,
               fill_holes: bool = False,
               use_occlusion: bool = False,
               fill_borders: bool = True,
               custom_mask: np.ndarray = None,
               custom_mask_mode: str = "replace",
               poisson_blend_edges: bool = False,
               debug: bool = False) -> tuple:

    H, W = original_np.shape[:2]
    diag = _diag(H, W)
    debug_images = {} if debug else None

    orig_u8   = (np.clip(original_np,  0, 1) * 255).astype(np.uint8)
    gen_u8    = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)
    gray_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
    gray_gen  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

    blur_kernel = _blur_kernel_for_diag(diag)
    sk = max(_blur_kernel_for_diag(diag)[0], 5)
    if sk % 2 == 0: sk += 1

    auto_report = {}

    # ------------------------------------------------------------------
    # Custom mask override (replace mode): bypass internal detection entirely
    # ------------------------------------------------------------------
    if custom_mask is not None and custom_mask_mode == "replace":
        if debug:
            debug_images['custom_mask_override'] = np.stack([custom_mask] * 3, axis=-1)
            debug_images['mask_overlay'] = original_np.copy()
            debug_images['mask_overlay'][custom_mask > 0.5] = (
                debug_images['mask_overlay'][custom_mask > 0.5] * 0.5 + np.array([0, 0.5, 0])
            )

        gen_pre, _, inliers, valid, debug_sift = _sift_prealign(
            gray_orig, gray_gen, generated_np, orig_mask=None, debug=debug
        )
        if debug:
            debug_images['pass1_sift_matches'] = debug_sift.get('match_viz', np.zeros((H, W, 3)))
            debug_images['pass1_validity_mask'] = valid.copy()

        sharp_mask = np.clip(custom_mask, 0.0, 1.0) * valid
        
        if fill_borders:
            sharp_mask = _bleed_mask(sharp_mask, valid)

        if feather_px > 0:
            inv_mask = (sharp_mask < 0.5).astype(np.uint8)
            if inv_mask.min() == 0:
                dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
                fade_dist = feather_px * 2.5
                t = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
                composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
            else:
                composite_mask = sharp_mask
        else:
            composite_mask = sharp_mask

        if not fill_borders:
            composite_mask *= valid
        
        # --- COLOR MATCH & POISSON BLEND ---
        gen_pre, color_matched = _apply_color_match(
            original_np, gen_pre, composite_mask, valid, color_match_blend
        )
        if color_matched: auto_report['color_match_applied'] = True

        if poisson_blend_edges:
            result = _seamless_blend(original_np, gen_pre, composite_mask)
        else:
            m3     = composite_mask[..., np.newaxis]
            result = np.clip(original_np * (1.0 - m3) + gen_pre * m3, 0, 1)
        # -----------------------------------

        flow_fwd_final = _dis_flow(
            gray_orig,
            cv2.cvtColor((np.clip(gen_pre, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
            flow_preset,
        )
        if use_occlusion:
            flow_bwd_final = _dis_flow(
                cv2.cvtColor((np.clip(gen_pre, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                gray_orig, flow_preset
            )
            occ_thresh = occlusion_threshold if occlusion_threshold >= 0 else _auto_threshold_mad(_fwd_bwd_error(flow_fwd_final, flow_bwd_final), valid, k=5.0, min_t=1.0, max_t=30.0)
            if occlusion_threshold < 0: auto_report['auto_occlusion'] = occ_thresh
        else:
            flow_bwd_final = None
            occ_thresh     = 0

        flow_mag  = np.sqrt((flow_fwd_final**2).sum(axis=2))
        stats = {
            "changed_pct":    100 * float((sharp_mask > 0.5).sum()) / (H * W),
            "occluded_px":    int((_fwd_bwd_error(flow_fwd_final, flow_bwd_final) > occ_thresh).sum()) if use_occlusion else 0,
            "flow_mean_px":   float(flow_mag.mean()),
            "flow_p99_px":    float(np.percentile(flow_mag, 99)),
            "median_de":      0.0,
            "resolution":     f"{W}x{H}",
            "diagonal_px":    round(diag),
            "pass1_inliers":  inliers,
            "pass2_used":     False,
            "custom_mask":    True,
            "poisson_used":   poisson_blend_edges,
        }
        stats.update(auto_report)

        if debug:
            debug_images['final_flow']      = _flow_to_color(flow_fwd_final)
            debug_images['final_alignment'] = _create_side_by_side(
                original_np, gen_pre, ("Original", "Aligned (custom mask)")
            )
            debug_images['composite_breakdown'] = np.hstack([
                original_np, gen_pre, result, np.stack([composite_mask] * 3, axis=-1),
            ])

        return result, composite_mask, stats, debug_images

    # ------------------------------------------------------------------
    # PASS 1: Blind Alignment → Coarse Mask
    # ------------------------------------------------------------------
    gen_pre_1, H_sift1, inliers_1, valid_1, debug_sift1 = _sift_prealign(
        gray_orig, gray_gen, generated_np, orig_mask=None, debug=debug
    )

    if debug:
        debug_images['pass1_sift_matches'] = debug_sift1.get('match_viz', np.zeros((H, W, 3)))
        debug_images['pass1_validity_mask'] = valid_1.copy()
        valid_overlay = original_np.copy()
        valid_overlay[valid_1 < 0.5] = [1, 0, 0]
        debug_images['pass1_validity_overlay'] = valid_overlay

    gray_gen_pre_1 = cv2.cvtColor((np.clip(gen_pre_1, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    flow_fwd_1 = _dis_flow(gray_orig, gray_gen_pre_1, flow_preset)
    flow_bwd_1 = _dis_flow(gray_gen_pre_1, gray_orig, flow_preset) if use_occlusion else None

    warped_gen_1 = _warp(gen_pre_1, flow_fwd_1)

    diff_1_direct = _compute_diff_map(original_np, gen_pre_1, blur_kernel)
    diff_1_warped = _compute_diff_map(original_np, warped_gen_1, blur_kernel)

    de_thresh = delta_e_threshold if delta_e_threshold >= 0 else _auto_threshold_mad(diff_1_direct, valid_1)

    # Blend: direct diff signals strongly inside the edit, warped diff is cleaner in the background
    blend_w_1 = np.clip(diff_1_direct / (de_thresh + 1e-6), 0.0, 1.0)
    delta_e_1_raw = blend_w_1 * diff_1_direct + (1.0 - blend_w_1) * diff_1_warped
    delta_e_1 = cv2.GaussianBlur(delta_e_1_raw, (sk, sk), 0)

    # Local adaptive threshold — more sensitive in smooth/flat regions
    local_thresh_1 = _local_threshold_map(delta_e_1, valid_1, diag)
    thresh_map_1   = np.minimum(de_thresh, local_thresh_1)
    coarse_mask = (delta_e_1 > thresh_map_1).astype(np.float32)

    if use_occlusion:
        occ_thresh = occlusion_threshold if occlusion_threshold >= 0 else _auto_threshold_mad(_fwd_bwd_error(flow_fwd_1, flow_bwd_1), valid_1, k=5.0, min_t=1.0, max_t=30.0)
        coarse_mask = np.maximum(coarse_mask, _occlusion_mask(flow_fwd_1, flow_bwd_1, occ_thresh))
        
    coarse_mask *= valid_1

    if debug:
        de_normalized = np.clip(delta_e_1 / (de_thresh * 1.5 + 1e-6), 0, 1)
        debug_images['pass1_delta_e']     = _apply_heatmap(original_np, de_normalized)
        debug_images['pass1_coarse_mask'] = coarse_mask.copy()
        debug_images['pass1_flow']        = _flow_to_color(flow_fwd_1)
        debug_images['pass1_alignment']   = _create_side_by_side(original_np, warped_gen_1, ("Original", "Pass1 Warped"))

    # ------------------------------------------------------------------
    # PASS 2: Masked SIFT Alignment (Background Only)
    # ------------------------------------------------------------------
    bg_mask_u8   = (coarse_mask < 0.1).astype(np.uint8) * 255
    safe_k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg_mask_safe = cv2.erode(bg_mask_u8, safe_k)

    pass2_used = False
    if (bg_mask_safe > 0).sum() > (H * W * 0.05):
        final_aligned_gen, H_sift2, inliers_2, valid_2, debug_sift2 = _sift_prealign(
            gray_orig, gray_gen, generated_np, orig_mask=bg_mask_safe, debug=debug
        )
        if H_sift2 is not None:
            pass2_used = True
            auto_report["pass2_inliers"] = inliers_2
            if debug:
                debug_images['pass2_sift_matches']  = debug_sift2.get('match_viz', np.zeros((H, W, 3)))
                debug_images['pass2_validity_mask'] = valid_2.copy()
        else:
            final_aligned_gen, valid_2 = gen_pre_1, valid_1
            auto_report["pass2_inliers"] = 0
    else:
        final_aligned_gen, valid_2 = gen_pre_1, valid_1
        auto_report["pass2_inliers"] = 0
        if debug:
            debug_images['pass2_skip_reason'] = f"Background too small ({(bg_mask_safe > 0).sum()} px)"

    # ------------------------------------------------------------------
    # PASS 3: Final Precision Mask
    # ------------------------------------------------------------------
    gray_gen_pre_2 = cv2.cvtColor((np.clip(final_aligned_gen, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    flow_fwd_2 = _dis_flow(gray_orig, gray_gen_pre_2, flow_preset)
    flow_bwd_2 = _dis_flow(gray_gen_pre_2, gray_orig, flow_preset) if use_occlusion else None

    warped_gen_final = _warp(final_aligned_gen, flow_fwd_2)

    diff_2_direct = _compute_diff_map(original_np, final_aligned_gen, blur_kernel)
    diff_2_warped = _compute_diff_map(original_np, warped_gen_final, blur_kernel)

    blend_w_2 = np.clip(diff_2_direct / (de_thresh + 1e-6), 0.0, 1.0)
    delta_e_2_raw = blend_w_2 * diff_2_direct + (1.0 - blend_w_2) * diff_2_warped
    delta_e_2 = cv2.GaussianBlur(delta_e_2_raw, (sk, sk), 0)

    # Local adaptive threshold — more sensitive in smooth/flat regions
    local_thresh_2 = _local_threshold_map(delta_e_2, valid_2, diag)
    thresh_map_2   = np.minimum(de_thresh, local_thresh_2)
    sharp_mask = (delta_e_2 > thresh_map_2).astype(np.float32)
    
    if use_occlusion:
        sharp_mask = np.maximum(sharp_mask, _occlusion_mask(flow_fwd_2, flow_bwd_2, occ_thresh))
        
    sharp_mask *= valid_2

    if delta_e_threshold < 0: auto_report["auto_delta_e"] = de_thresh
    if use_occlusion and occlusion_threshold < 0: auto_report["auto_occlusion"] = occ_thresh

    # Morphological clean up
    if noise_removal_px > 0:
        sharp_mask = _open_mask(sharp_mask, noise_removal_px)
        sharp_mask *= valid_2
    if grow_px != 0:
        sharp_mask = _grow_mask(sharp_mask, grow_px)
        sharp_mask *= valid_2
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        sharp_mask = cv2.morphologyEx(sharp_mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)
    if max_islands > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats((sharp_mask > 0.5).astype(np.uint8), connectivity=8)
        if n - 1 > max_islands:
            areas =[(stats_cc[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
            areas.sort(reverse=True)
            keep = {i for _, i in areas[:max_islands]}
            sharp_mask[np.isin(labeled, list(keep), invert=True) & (labeled > 0)] = 0

    if fill_holes:
        sharp_mask = _fill_holes(sharp_mask)
        sharp_mask *= valid_2

    # ------------------------------------------------------------------
    # Custom mask combination (add / subtract modes)
    # ------------------------------------------------------------------
    if custom_mask is not None and custom_mask_mode in ("add", "subtract"):
        cm = np.clip(custom_mask, 0.0, 1.0)
        if custom_mask_mode == "add":
            sharp_mask = np.clip(sharp_mask + cm, 0.0, 1.0)
        else:  # subtract
            sharp_mask = np.clip(sharp_mask - cm, 0.0, 1.0)
        sharp_mask *= valid_2
        if debug:
            debug_images['custom_mask_override'] = np.stack([cm] * 3, axis=-1)
            auto_report['custom_mask_mode'] = custom_mask_mode

    # Extrapolate mask into invalid regions (shrinkage voids)
    if fill_borders:
        sharp_mask = _bleed_mask(sharp_mask, valid_2)

    # High quality edge-aware mask feathering
    if feather_px > 0:
        inv_mask = (sharp_mask < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:
            dist      = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            fade_dist = feather_px * 2.5
            t         = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        else:
            composite_mask = sharp_mask
    else:
        composite_mask = sharp_mask

    if not fill_borders:
        composite_mask *= valid_2

    # --- COLOR MATCH & POISSON BLEND ---
    final_aligned_gen, color_matched = _apply_color_match(
        original_np, final_aligned_gen, composite_mask, valid_2, color_match_blend
    )
    if color_matched: auto_report['color_match_applied'] = True

    if poisson_blend_edges:
        result = _seamless_blend(original_np, final_aligned_gen, composite_mask)
    else:
        m3     = composite_mask[..., np.newaxis]
        result = np.clip(original_np * (1.0 - m3) + final_aligned_gen * m3, 0, 1)
    # -----------------------------------

    # Reporting Stats
    flow_mag  = np.sqrt((flow_fwd_2**2).sum(axis=2))
    n_changed = int((sharp_mask > 0.5).sum())
    stats = {
        "changed_pct":    100 * n_changed / (H * W),
        "occluded_px":    int((_fwd_bwd_error(flow_fwd_2, flow_bwd_2) > occ_thresh).sum()) if use_occlusion else 0,
        "flow_mean_px":   float(flow_mag.mean()),
        "flow_p99_px":    float(np.percentile(flow_mag, 99)),
        "median_de":      float(np.median(delta_e_2)),
        "resolution":     f"{W}x{H}",
        "diagonal_px":    round(diag),
        "pass1_inliers":  inliers_1,
        "pass2_used":     pass2_used,
        "poisson_used":   poisson_blend_edges,
    }
    stats.update(auto_report)

    if debug:
        debug_images['final_flow']           = _flow_to_color(flow_fwd_2)
        debug_images['final_flow_magnitude'] = flow_mag / (flow_mag.max() + 1e-6)

        de_norm = np.clip(delta_e_2 / (de_thresh * 1.5 + 1e-6), 0, 1)
        debug_images['final_delta_e'] = _apply_heatmap(original_np, de_norm)

        debug_images['final_sharp_mask']     = sharp_mask.copy()
        debug_images['final_composite_mask'] = composite_mask.copy()

        mask_overlay = original_np.copy()
        mask_overlay[sharp_mask > 0.5] = (
            mask_overlay[sharp_mask > 0.5] * 0.5 + np.array([0, 0.5, 0])
        )
        debug_images['mask_overlay'] = mask_overlay

        debug_images['final_alignment'] = _create_side_by_side(
            original_np, warped_gen_final, ("Original", "Final Warped")
        )
        debug_images['composite_breakdown'] = np.hstack([
            original_np, final_aligned_gen, result, np.stack([composite_mask] * 3, axis=-1),
        ])

    return result, composite_mask, stats, debug_images


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class KleinEditComposite:
    """
    Composites a Klein edit onto the original image with full debug visualization.
    Uses robust MAGSAC SIFT alignment, Gradient+LAB structure difference, and 
    optional Poisson Blending for seamless lighting transfers.
    """

    CATEGORY = "image/Klein"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "original_image":  ("IMAGE",),
                # --- Detection ---
                "delta_e_threshold": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 100.0, "step": 1.0,
                    "tooltip": (
                        "Controls how different a pixel must be (in perceptual color distance) "
                        "to be considered 'changed' and included in the composite mask. "
                        "Lower values = more sensitive, picking up subtle edits but risking false "
                        "positives in unchanged areas. Higher values = less sensitive, only catching "
                        "obvious changes but potentially missing fine details. "
                        "Set to -1 to let the node auto-tune this threshold using Median Absolute "
                        "Deviation (MAD) — recommended for most cases."
                    ),
                }),
                "flow_quality": (["medium", "fast", "ultrafast"], {
                    "default": "medium",
                    "tooltip": (
                        "Sets the precision of the optical flow calculation used to detect "
                        "motion and structural changes between the original and generated images. "
                        "'medium' gives the best accuracy and is recommended for most edits. "
                        "'fast' reduces computation time at a slight accuracy cost — useful for "
                        "quick previews or large images. "
                        "'ultrafast' is the lowest quality but quickest — suitable for rough drafts "
                        "or iterating on settings before a final render."
                    ),
                }),
                "use_occlusion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When enabled, adds occlusion detection to the change mask — areas where "
                        "one image occludes content visible in the other (e.g. an object moves and "
                        "reveals background behind it). "
                        "Turn ON when the edit involves camera movement or objects shifting position, "
                        "as it ensures those newly revealed regions are included in the mask. "
                        "Leave OFF for color, texture, or style edits — occlusion detection can "
                        "bleed the mask into unchanged areas when there is no real motion."
                    ),
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 50.0, "step": 0.5,
                    "tooltip": (
                        "Only active when 'use_occlusion' is enabled. Controls how large the "
                        "forward-backward optical flow error must be before a pixel is classified "
                        "as occluded. Lower values flag more pixels as occluded (wider mask, "
                        "more bleed risk). Higher values only flag clearly inconsistent flow "
                        "(tighter mask, may miss subtle occlusions). "
                        "Set to -1 to auto-tune using Median Absolute Deviation — recommended."
                    ),
                }),
                # --- Mask cleanup ---
                "noise_removal_pct": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": (
                        "Removes speckle noise from the raw change mask using morphological opening "
                        "(erode then dilate). Value is the radius as a percentage of the image diagonal, "
                        "so it scales automatically with resolution. "
                        "Increase this when the mask has many tiny isolated dots or grains in areas "
                        "that should be clean background — higher values eliminate larger speckles. "
                        "Keep at 0 (disabled) if the edit has fine detail at the edges, as too high "
                        "a value will erode legitimate thin features like hair or text."
                    ),
                }),
                "close_radius_pct": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "Fills small gaps and connects nearby mask regions using morphological closing "
                        "(dilate then erode). Value is the radius as a percentage of the image diagonal. "
                        "Increase this to bridge gaps between nearby changed regions so they merge into "
                        "one solid area — useful when the mask has broken edges or thin holes cutting "
                        "through the middle of an edited object. "
                        "Lower values preserve tight separation between distinct changed regions. "
                        "Too high a value can merge separate objects that should remain independent."
                    ),
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Flood-fills enclosed interior holes inside the mask. "
                        "Turn ON when the detected mask has gaps or voids inside an edited region — "
                        "for example, a changed object whose interior was not fully detected. This "
                        "ensures the entire interior is composited rather than leaving patches of the "
                        "original showing through. "
                        "Leave OFF if you intentionally need transparency or cut-outs inside the mask, "
                        "or if interior regions are genuinely unchanged background."
                    ),
                }),
                "fill_borders": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When alignment warps or shifts the generated image, it can leave void border "
                        "regions where no generated pixel exists. "
                        "Turn ON (default) to extrapolate the mask into those borders and fill them "
                        "with mirrored pixels from the generated image, hiding the void seamlessly. "
                        "Turn OFF to instead fall back to the original image in those border areas — "
                        "useful if the mirrored fill looks wrong or if the warp is large enough that "
                        "mirroring produces visible artifacts."
                    ),
                }),
                "max_islands": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "Limits the mask to only the N largest connected regions, discarding all smaller "
                        "isolated patches. Set to 1 to keep only the single largest changed area. "
                        "Increase to allow a small number of distinct regions (e.g. 2 for a foreground "
                        "object and a shadow). "
                        "Set to 0 (default) to keep all regions regardless of size — best when the edit "
                        "spans many small areas like texture or pattern changes. "
                        "Use in combination with noise_removal to first eliminate tiny speckles before "
                        "this filter acts on the remaining islands."
                    ),
                }),
                "grow_mask_pct": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
                    "tooltip": (
                        "Expands or contracts the final mask boundary as a percentage of the image diagonal. "
                        "Positive values grow the mask outward — use this to ensure the composite fully "
                        "covers the edited region when detection falls slightly short of the true edges. "
                        "Negative values shrink the mask inward — useful for tightening a mask that bleeds "
                        "into unchanged background. "
                        "0 leaves the mask exactly as detected. Feathering is applied after this step, "
                        "so growth and feather work together to produce a clean edge."
                    ),
                }),
                "feather_pct": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.25,
                    "tooltip": (
                        "Controls the width of the soft blend transition at mask edges, as a percentage "
                        "of the image diagonal. Applies a smooth distance-based falloff from the mask "
                        "boundary, producing a clean gradual transition between the composited and "
                        "original regions. "
                        "Higher values create a wider, softer blend — good for smooth subjects like skin "
                        "or fabric where a hard edge would look artificial. "
                        "Lower values produce a tighter, harder edge — useful for geometric subjects or "
                        "when you need precise control and the edges are already well-detected. "
                        "0 disables feathering entirely."
                    ),
                }),
                "color_match_blend": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Corrects lighting and color tone differences between the original and generated "
                        "images before compositing, using Reinhard color transfer on the background region. "
                        "1.0 (default) fully matches the generated image's colors to the original — "
                        "recommended when the AI model has shifted the overall brightness, white balance, "
                        "or saturation compared to the source. "
                        "Lower values partially blend the correction, preserving some of the generated "
                        "image's own color grading. "
                        "Set to 0.0 to disable color matching entirely — use this if the generated image's "
                        "color shift is intentional (e.g. a day-to-night conversion)."
                    ),
                }),
                "poisson_blend_edges": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": (
                        "Uses Poisson Blending (cv2.seamlessClone) to mathematically eliminate lighting "
                        "and color seams introduced by the AI model. If disabled, uses alpha blending. "
                    )
                }),
            },
            "optional": {
                "custom_mask": ("MASK", {
                    "tooltip": (
                        "An optional mask to modify or replace the auto-detected composite boundary. "
                        "Behaviour is controlled by the custom_mask_mode setting below. "
                        "Feathering, grow, and fill_borders always apply after the mask is combined. "
                        "Leave disconnected to use the node's built-in detection pipeline."
                    ),
                }),
                "custom_mask_mode": (["replace", "add", "subtract"], {
                    "default": "replace",
                    "tooltip": (
                        "Controls how the custom_mask is combined with the auto-detected mask.\n"
                        "replace: skips internal detection entirely and uses only the custom mask "
                        "(original behaviour — fastest, use when auto-detection is unreliable).\n"
                        "add: runs the full detection pipeline, then unions the custom mask on top — "
                        "useful for forcing extra regions into the composite that detection missed.\n"
                        "subtract: runs the full detection pipeline, then removes the custom mask area "
                        "from the result — useful for protecting specific regions (e.g. background "
                        "objects that are incorrectly flagged as changed) from being composited."
                    ),
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Outputs a debug gallery image showing internal intermediate states: SIFT feature "
                        "matches, alignment results, difference maps, flow visualization, and the detected "
                        "change mask overlay. "
                        "Turn ON to diagnose alignment failures, unexpected mask shapes, or blending "
                        "artifacts — the gallery makes it easy to see exactly where the pipeline is "
                        "succeeding or going wrong. "
                        "Leave OFF during normal use to save memory and processing time."
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES  = ("composited_image", "change_mask", "report", "debug_gallery")
    FUNCTION      = "run"
    OUTPUT_NODE   = True

    def run(self, generated_image, original_image,
            delta_e_threshold=-1.0, grow_mask_pct=0.0, feather_pct=2.0, color_match_blend=1.0,
            flow_quality="medium", occlusion_threshold=-1.0,
            close_radius_pct=0.5, noise_removal_pct=0.0, max_islands=0,
            fill_holes=False, fill_borders=True, use_occlusion=False, enable_debug=False,
            custom_mask=None, custom_mask_mode="replace", poisson_blend_edges=False):

        orig_np = original_image[0].cpu().float().numpy()
        gen_np  = generated_image[0].cpu().float().numpy()

        if orig_np.shape != gen_np.shape:
            H, W    = gen_np.shape[:2]
            pil     = Image.fromarray((orig_np * 255).astype(np.uint8))
            orig_np = np.array(pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        H, W       = gen_np.shape[:2]
        diag       = _diag(H, W)

        grow_px          = round(grow_mask_pct * diag / 100.0)
        feather_px       = abs(feather_pct) * diag / 100.0
        close_px         = _pct_to_px(close_radius_pct, diag)
        noise_removal_px = _pct_to_px(noise_removal_pct, diag)

        custom_mask_np = None
        if custom_mask is not None:
            custom_mask_np = custom_mask[0].cpu().float().numpy()
            if custom_mask_np.shape != (H, W):
                pil_mask       = Image.fromarray((custom_mask_np * 255).astype(np.uint8))
                custom_mask_np = np.array(pil_mask.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        result, change_mask, stats, debug_images = _composite(
            orig_np, gen_np,
            delta_e_threshold   = delta_e_threshold,
            flow_preset         = FLOW_PRESETS[flow_quality],
            occlusion_threshold = occlusion_threshold,
            grow_px             = grow_px,
            close_radius        = close_px,
            noise_removal_px    = noise_removal_px,
            max_islands         = max_islands,
            fill_holes          = fill_holes,
            use_occlusion       = use_occlusion,
            fill_borders        = fill_borders,
            feather_px          = feather_px,
            color_match_blend   = color_match_blend,
            custom_mask         = custom_mask_np,
            custom_mask_mode    = custom_mask_mode,
            poisson_blend_edges = poisson_blend_edges,
            debug               = enable_debug,
        )

        report_lines =[
            "=== Klein Edit Composite ===",
            f"Resolution:       {stats['resolution']}  (diag {stats['diagonal_px']}px)",
            f"Poisson Blending: {'ENABLED (Illumination fixed)' if stats.get('poisson_used') else 'Disabled (Classic Alpha Blend)'}",
            "",
        ]

        if stats.get("custom_mask"):
            report_lines.append("Mask source:      CUSTOM replace (internal detection bypassed)")
        elif stats.get("custom_mask_mode"):
            mode = stats["custom_mask_mode"]
            report_lines.append(f"Mask source:      AUTO + custom {mode.upper()}")
        elif stats.get("pass2_inliers", 0) > 0:
            report_lines.append("Alignment:        Two-Pass Success! Background rigidly locked.")
            report_lines.append(f"Pass 1 Inliers:   {stats['pass1_inliers']} (Blind SIFT+MAGSAC)")
            report_lines.append(f"Pass 2 Inliers:   {stats['pass2_inliers']} (Masked background only)")
        else:
            report_lines.append(
                f"Alignment:        {'Pass 2 executed' if stats['pass2_used'] else 'Pass 2 skipped'} "
                f"(fallback to Pass 1)"
            )

        if enable_debug and debug_images:
            report_lines += ["", "=== Debug Info ==="]
            if 'pass1_sift_matches' in debug_images:
                h, w = debug_images['pass1_sift_matches'].shape[:2]
                report_lines.append(f"SIFT Match Viz:   {w}x{h} (Green=Inliers, Red=Outliers)")
            if 'pass1_validity_mask' in debug_images:
                valid_pct = 100 * debug_images['pass1_validity_mask'].mean()
                report_lines.append(f"Validity Coverage: {valid_pct:.1f}% (black areas = void after warp)")
            if 'final_flow_magnitude' in debug_images:
                report_lines.append("Flow visualization available in debug gallery")
            if 'custom_mask_override' in debug_images:
                report_lines.append("Custom mask override active — debug shows alignment only")

        report_lines.append("")

        if "auto_delta_e" in stats:
            report_lines.append(f"Diff Threshold:   AUTO (MAD) → {stats['auto_delta_e']:.1f}")
        else:
            report_lines.append(f"Diff Threshold:   {delta_e_threshold:.1f}")

        if "auto_occlusion" in stats:
            report_lines.append(f"Occlusion Thresh: AUTO (MAD) → {stats['auto_occlusion']:.1f}")
        else:
            report_lines.append(f"Occlusion Thresh: {occlusion_threshold:.1f}")

        report_lines +=[
            f"Grow mask:        {grow_mask_pct:+.1f}% → {grow_px:+d}px",
            f"Noise removal:    {noise_removal_pct:.2f}% → {noise_removal_px}px (opening radius)",
            f"Max islands:      {max_islands if max_islands > 0 else 'disabled'}",
            f"Fill holes:       {'yes' if fill_holes else 'no'}",
            f"Fill borders:     {'yes (mirrored pixels)' if fill_borders else 'no (show original)'}",
            f"Occlusion:        {'enabled' if use_occlusion else 'disabled'}",
            f"Feather:          {feather_pct:.1f}% → {feather_px:.0f}px",
            f"Color Match:      {color_match_blend * 100:.0f}% {'(Applied)' if stats.get('color_match_applied') else '(Skipped)'}",
            f"Flow quality:     {flow_quality}",
            "",
            f"Changed region:   {stats['changed_pct']:.1f}% of image",
            f"Flow mean shift:  {stats['flow_mean_px']:.2f}px  (Residual after Homography)",
            f"Median Diff:      {stats['median_de']:.2f}",
        ]

        debug_gallery = None
        if enable_debug:
            if not debug_images:
                warning = np.zeros((512, 512, 3), dtype=np.uint8)
                cv2.putText(warning, "DEBUG ENABLED BUT NO DATA",  (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,   (255, 255, 255), 2)
                cv2.putText(warning, "Check that enable_debug=True", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                debug_gallery = torch.from_numpy(warning.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                order =[
                    ('custom_mask_override',  'Custom Mask Override'),
                    ('pass1_sift_matches',     'SIFT Matches (Pass 1)'),
                    ('pass1_alignment',        'Pass 1 Alignment'),
                    ('pass1_delta_e',          'Diff Map (Pass 1)'),
                    ('pass1_validity_overlay', 'Validity Mask (Red=Void)'),
                    ('final_alignment',        'Final Alignment'),
                    ('final_delta_e',          'Diff Map (Final)'),
                    ('final_flow',             'Optical Flow'),
                    ('mask_overlay',           'Detected Changes'),
                    ('composite_breakdown',    'Original | Aligned | Result | Mask'),
                ]

                viz_images =[]
                for key, _label in order:
                    if key in debug_images and debug_images[key] is not None:
                        img = debug_images[key]
                        if len(img.shape) == 2:
                            img = np.stack([img] * 3, axis=-1)
                        if img.dtype != np.uint8:
                            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
                        viz_images.append(img)

                if not viz_images:
                    debug_gallery = torch.zeros((1, 512, 512, 3))
                else:
                    target_h = 400
                    rows =[]

                    for i in range(0, len(viz_images), 3):
                        row_imgs = viz_images[i:i+3]
                        resized  =[]
                        for img in row_imgs:
                            if img.shape[0] != target_h:
                                scale = target_h / img.shape[0]
                                img   = cv2.resize(img, (int(img.shape[1] * scale), target_h))
                            resized.append(img)

                        while len(resized) < 3:
                            h, w = resized[0].shape[:2] if resized else (target_h, target_h)
                            resized.append(np.full((h, w, 3), 64, dtype=np.uint8))

                        rows.append(np.hstack(resized))

                    if len(rows) > 1:
                        max_width = max(row.shape[1] for row in rows)
                        rows = [
                            np.hstack([row, np.full((target_h, max_width - row.shape[1], 3), 64, dtype=np.uint8)])
                            if row.shape[1] < max_width else row
                            for row in rows
                        ]

                    gallery       = np.vstack(rows)
                    debug_gallery = torch.from_numpy(gallery.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            debug_gallery = torch.zeros((1, 64, 64, 3))

        return (
            torch.from_numpy(result).unsqueeze(0),
            torch.from_numpy(change_mask).unsqueeze(0),
            "\n".join(report_lines),
            debug_gallery,
        )


NODE_CLASS_MAPPINGS      = {"KleinEditComposite": KleinEditComposite}
NODE_DISPLAY_NAME_MAPPINGS = {"KleinEditComposite": "Klein Edit Composite"}
