"""Quick diagnostic: export the silhouette mask to verify it's correct."""
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
sys.path.insert(0, '.')

# Reuse silhouette function from HALLEY_generator
W, H = 850, 1100

def build_silhouette(W, H):
    canvas = Image.new('L', (W, H), 255)
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    cx = W // 2

    # Hair: crown mass + left flow (longer) + right flow
    d.ellipse([cx-168, 26, cx+148, 272], fill=0)
    d.polygon([(cx-152,90),(cx-218,212),(cx-238,370),(cx-208,510),
               (cx-168,590),(cx-118,572),(cx-78,432),(cx-108,272),
               (cx-118,172),(cx-138,112)], fill=0)
    d.polygon([(cx+122,90),(cx+182,192),(cx+192,342),(cx+162,442),
               (cx+112,462),(cx+82,372),(cx+92,232),(cx+102,152)], fill=0)

    # Head / face oval
    d.ellipse([cx-128, 96, cx+128, 362], fill=0)

    # Neck
    d.rectangle([cx-44, 336, cx+44, 432], fill=0)

    # Shoulders + upper torso
    d.polygon([(cx-218,412),(cx+218,412),
               (cx+198,552),(cx+148,642),
               (cx+128,792),(cx+118,1005),
               (cx-118,1005),(cx-128,792),
               (cx-148,642),(cx-198,552)], fill=0)

    # Arm suggestions
    d.ellipse([cx-268,462,cx-198,688], fill=0)
    d.ellipse([cx+198,462,cx+268,668], fill=0)

    arr = np.array(canvas).astype(float)
    figure = (arr < 128).astype(float)
    return gaussian_filter(figure, sigma=10)

print("Building silhouette...")
sil = build_silhouette(W, H)
print(f"Silhouette shape: {sil.shape}")
print(f"Silhouette min/max: {sil.min():.3f} / {sil.max():.3f}")
print(f"Non-zero pixels: {np.count_nonzero(sil > 0)}")

# Export as visualization (brighten for visibility)
sil_255 = (sil * 255).astype(np.uint8)
sil_img = Image.fromarray(sil_255, mode='L')
sil_img.save('HALLEY_silhouette_diagnostic.png')
print("Saved: HALLEY_silhouette_diagnostic.png")

# Check a few y-values to see if bounds are being detected
cx = W // 2
for y_test in [50, 200, 400, 600, 900]:
    row = sil[y_test, :]
    inside = np.where(row > 0.05)[0]
    if inside.size > 0:
        x_min, x_max = int(inside[0]), int(inside[-1])
        print(f"  y={y_test}: bounds ({x_min}, {x_max}), width={x_max-x_min}")
    else:
        print(f"  y={y_test}: no bounds found")
