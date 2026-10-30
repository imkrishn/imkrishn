"""
Prepare a portrait photo for clean ASCII conversion:
  1. remove the background (rembg) so the subject is isolated
  2. boost LOCAL contrast (CLAHE) so a flatly-lit face gains highlights and
     shadows -- this is what turns a dark blob into a recognizable face
  3. composite the subject onto pure white so the background reads as blank
     (white -> spaces in the ascii ramp)

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.
Run once whenever the source photo changes; the ascii SVG itself is static.

    python scripts/prep_photo.py <input.jpg> [output.png]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
try:
    from rembg import remove
except Exception:          # rembg missing / model unavailable -> OpenCV GrabCut fallback
    remove = None


def grabcut_remove(img):
    """Fallback subject cut-out using OpenCV GrabCut (no model download needed)."""
    rgb_in = np.array(img.convert("RGB"))
    h, w = rgb_in.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    # assume subject roughly centred; borders are definitely background
    x0, y0, x1, y1 = int(w * .12), int(h * .06), int(w * .90), int(h * .98)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    mask[int(h * .25):int(h * .75), int(w * .30):int(w * .70)] = cv2.GC_FGD
    mask[:int(h * .03), :] = cv2.GC_BGD
    mask[:, :int(w * .03)] = cv2.GC_BGD
    mask[:, int(w * .97):] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    a = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # constrain to a head + neck silhouette (relative coords) so stray background
    # patches / shirt are dropped, then fade the neck out at the bottom
    poly = np.array([(.10,.05),(.90,.05),(.89,.37),(.80,.42),(.735,.46),(.725,.52),(.70,.62),
                     (.65,.72),(.61,.78),(.60,.92),(.28,.92),(.28,.66),(.27,.60),
                     (.20,.58),(.19,.40),(.12,.34)], np.float32) * [w, h]
    pm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(pm, [poly.astype(np.int32)], 255)
    a = cv2.bitwise_and(a, pm)
    ys = np.linspace(0, 1, h)[:, None]
    fade = np.clip((0.92 - ys) / 0.14, 0, 1)
    a = (a * fade).astype(np.uint8)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(a))
    return out

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

# 1. cut out the subject
src_img = Image.open(INP).convert("RGBA")
try:
    cut = remove(src_img) if remove else grabcut_remove(src_img)
except Exception as e:      # e.g. rembg model download blocked
    print("rembg failed (%s) -> using GrabCut fallback" % e)
    cut = grabcut_remove(src_img)
rgb = np.array(cut.convert("RGB"))
alpha = np.array(cut.split()[-1])                 # 0 = background

# 2. local-contrast the luminance (CLAHE)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# a touch of global lift so the face sits in the sparse end of the ramp
gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=18)

# 3. paste onto white using the alpha mask (feathered a hair to avoid a halo)
mask = (alpha.astype(np.float32) / 255.0)
mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
out = gray.astype(np.float32) * mask + 255.0 * (1.0 - mask)
out = np.clip(out, 0, 255).astype(np.uint8)

# crop to the subject and pad to the ascii grid's aspect ratio (800x795 cells)
# so the portrait is not stretched horizontally
ys_, xs_ = np.where(out < 245)
y0, y1, x0, x1 = ys_.min(), ys_.max(), xs_.min(), xs_.max()
m = int(0.04 * max(y1 - y0, x1 - x0))
out = out[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
h_, w_ = out.shape
target = 800 / 795
if w_ / h_ < target:
    pad = int(h_ * target - w_)
    out = np.pad(out, ((0, 0), (pad // 2, pad - pad // 2)), constant_values=255)
else:
    pad = int(w_ / target - h_)
    out = np.pad(out, ((pad // 2, pad - pad // 2), (0, 0)), constant_values=255)
Image.fromarray(out, mode="L").save(OUT)
print("wrote", OUT, out.shape)
