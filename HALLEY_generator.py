"""
HALLEY — Typographic Portrait Generator
Phase 2 · AI Persona Spec Portfolio

Generates a text-density portrait of HALLEY: a human-form figure constructed
entirely from semantically chosen English-language text. No illustration.
No skin tone. The text IS the form.

Usage:
    pip install Pillow numpy scipy
    python HALLEY_generator.py

Output: HALLEY_portrait.png (850x1100px, 300dpi)

Parameters to experiment with:
    - BACKGROUND: change the paper color
    - INK depth values: adjust the green palette
    - STREAMS: modify the text corpus for each zone
    - font size per zone (fs): larger = more readable words, less dense
    - line_step (ls): smaller = denser, more shadow depth
    - sigma in build_silhouette: larger = softer figure edges
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import re
import math

# ============================================================
# CONFIGURATION — adjust these to iterate
# ============================================================
W, H = 850, 1100          # canvas: 8.5x11 portrait ratio
BACKGROUND = (252, 252, 248)   # warm near-white paper
SEED = 42                       # change for different random arrangement

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# TEXT CORPUS — HALLEY's five semantic zones
# Zone 1: Outline/Contour     — King's formal public register
# Zone 2: Chest interior      — theological foundation
# Zone 3: Neck/Jawline        — intellectual lineage (names)
# Zone 4: Face                — most charged vocabulary
# Zone 5: Name integration    — structural (see name section below)
# ============================================================

STREAMS = {
    # Zone 4 — Face: highest density, most specific language
    'face': (
        "witnessolidarityreconciliationaccompanimentbearingwitness"
        "whatisrequiredwhoseburdenthelong arcsatyagrahaahimsasoul-force"
        "truth-forcenoncooperationfirmnessintruthagapejustice-seekinglove"
        "seeingknowingbeholdingperceivingjusticetruthbelovedcommunity"
        "witnesssolidarityreconcileaccompanybeartheburdenthelongarc"
    ) * 30,  # doubled for more density

    # Zone 3 — Neck/Jawline: intellectual lineage as column supporting head
    'neck': (
        "KingGandhiHowardThurmanEllaBakerFannieLouHamerJamesBaldwin"
        "WendellBerryMarilynneRobinsonDorothyDayJohnLewisVanderbilt"
        "TheSocialGospelProgressiveMethodistDivinityNonviolentPastoral"
    ) * 16,  # doubled

    # Zone 2 — Chest interior: theological bedrock, structural mass
    'chest': (
        "belovedcommunitydojusticlovemercywalkhumblyperfectlovecastethoutfear"
        "whatdoestheLordrequireAmazingGracehowsweetthesoundagape"
        "thykingdomcomeIwasaStrangerandyouwelcomedme1Corinthians13"
        "faithhopelovegraceupongracesteadfastloveneverceases"
        "bodyofChristpeacethatpassesunderstandingmercyandtruthhavemet"
        "WesleyanProgressiveMethodistIncarnationnReconcileCovenantSpirit"
    ) * 20,  # doubled

    # Hair/crown: intellectual inheritance, lighter register
    'hair': (
        "HowardThurmanMarilynneRobinsonWendellBerryJamesBaldwinDorothyDay"
        "JohnLewisEllabakerFannieLouHamerGandhiKingTheSocialGospelVanderbilt"
        "DivinitybelovedcommunityagapesolidarityprogressiveMethodistDivinity"
    ) * 24,  # doubled

    # General/lower torso
    'general': (
        "nonviolentpastoralwitnesscommunityprogressiveMethodistdivinitylovejustice"
        "mercygracepeacesolidarityaccompanimentbelovedtruthfreedomhope"
        "reconcilediscipleshipincarnationkingdomspiritagapeahimsasatyagraha"
        "Birmingham1963sermonprayercovenant"
    ) * 16,  # doubled

    # Zone 1 — Outline: King's formal register, sparse boundary language
    'outline': (
        "injusticeanywheretheatreatmenttojusticeeverywherethearcofsemoraluniversebendstojustice"
        "creativeextremistsLetterFromBirminghamJail1963humanprogressneverrollswheelsofinevitability"
        "onemustusetime creativelyonehasonlyalegalbutamoralresponsibility"
        "justicetolong delayedisjusticedeniedfreedomisnevervoluntarilygivenbytheoppressor"
    ) * 10,  # doubled
}

# Tokenize streams into readable word tokens when possible.
def tokens_from_stream(s):
    if isinstance(s, list):
        return s
    # If the author already used spaces, preserve words
    if re.search(r"\s+", s):
        return [w for w in re.findall(r"\S+", s)]
    # Try to split CamelCase and numbers
    toks = re.findall(r"[A-Z][a-z]+|[a-z]+|\d+", s)
    # If splitting yields too few tokens (likely long concatenation), chunk it
    if len(toks) < 8:
        chunk = 6
        toks = [s[i:i+chunk] for i in range(0, len(s), chunk)]
    return toks

STREAM_TOKENS = {k: tokens_from_stream(v) for k, v in STREAMS.items()}

# ============================================================
# SILHOUETTE — geometric figure, abstracted from pose reference
# Upright, composed, open-palmed. HALLEY's quality of stillness.
# Modify coordinates here to adjust the figure's proportions.
# ============================================================

def build_silhouette(W, H):
    """
    Builds the figure mask using geometric primitives.
    Returns a float array 0.0-1.0 where 1.0 = deepest figure interior.
    The gaussian blur (sigma) controls edge softness.
    """
    canvas = Image.new('L', (W, H), 255)
    d = ImageDraw.Draw(canvas)
    cx = W // 2   # horizontal center

    # Hair: crown mass + left flow (longer) + right flow
    d.ellipse([cx-168, 26, cx+148, 272], fill=0)
    d.polygon([(cx-152,90),(cx-218,212),(cx-238,370),(cx-208,510),
               (cx-168,590),(cx-118,572),(cx-78,432),(cx-108,272),
               (cx-118,172),(cx-138,112)], fill=0)
    d.polygon([(cx+122,90),(cx+182,192),(cx+192,342),(cx+162,442),
               (cx+112,462),(cx+82,372),(cx+92,232),(cx+102,152)], fill=0)

    # Head / face oval
    d.ellipse([cx-128, 96, cx+128, 362], fill=0)

    # Neck — anatomical site of voice (Zone 3 lives here)
    d.rectangle([cx-44, 336, cx+44, 432], fill=0)

    # Shoulders + upper torso: open, composed, slight forward lean
    d.polygon([(cx-218,412),(cx+218,412),
               (cx+198,552),(cx+148,642),
               (cx+128,792),(cx+118,1005),
               (cx-118,1005),(cx-128,792),
               (cx-148,642),(cx-198,552)], fill=0)

    # Arm suggestions (open-palmed, per spec)
    d.ellipse([cx-268,462,cx-198,688], fill=0)
    d.ellipse([cx+198,462,cx+268,668], fill=0)

    arr = np.array(canvas).astype(float)
    figure = (arr < 128).astype(float)
    # sigma=10: moderate edge softness. Increase for more feathered edges.
    return gaussian_filter(figure, sigma=10)


# ============================================================
# FONT LOADER
# ============================================================

def get_font(size):
    """Load system font. Falls back gracefully."""
    for path in [
        # Windows common fonts
        'C:\\Windows\\Fonts\\arial.ttf',
        'C:\\Windows\\Fonts\\seguiemj.ttf',
        'C:\\Windows\\Fonts\\calibri.ttf',
        # Linux fallbacks
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()


# ============================================================
# MAIN GENERATION LOOP
# ============================================================

print("Building silhouette...")
sil = build_silhouette(W, H)

# Larger font sizes for legibility and typographic density control
fonts = {s: get_font(s) for s in [8, 10, 12, 14, 16, 20, 28, 36]}
sidx = {k: 0 for k in STREAMS}

# Use RGBA for proper alpha blending of rotated text
output = Image.new('RGBA', (W, H), (*BACKGROUND, 255))
draw = ImageDraw.Draw(output)
placed = 0
cx = W // 2

print("Generating typographic portrait...")
y = 10

# Helper: draw a word with rotation and jitter using an RGBA buffer
# Note: output is the main PIL Image, so we need to reference it globally
def draw_word_rot(img_output, pos, word, font, fill, angle=0, jitter=(0,0)):
    if not word:
        return 0
    try:
        bb = font.getbbox(word)
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
    except Exception:
        w = int(font.size * len(word) * 0.6)
        h = int(font.size * 1.2)

    buf_w = max(32, w + 8)
    buf_h = max(32, h + 8)
    im = Image.new('RGBA', (buf_w, buf_h), (0,0,0,0))
    d = ImageDraw.Draw(im)
    d.text((4, 2), word, font=font, fill=fill)
    if angle != 0:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=1)
    ox = int(pos[0] + jitter[0])
    oy = int(pos[1] + jitter[1])
    # Clamp to image bounds
    ox = max(0, min(ox, img_output.width - 1))
    oy = max(0, min(oy, img_output.height - 1))
    # paste RGBA buffer onto output using alpha composite
    img_output.paste(im, (ox, oy), im)
    return im.size[0]


# Build a list of available fonts (various sizes) for more wild variation
font_sizes = [10, 12, 14, 16, 18, 22, 28, 34]
font_pool = {s: get_font(s) for s in font_sizes}

def line_bounds_at_y(sil_array, y):
    # returns (x_left, x_right) interior bounds for a given integer y
    if y < 0 or y >= sil_array.shape[0]:
        return None
    row = sil_array[y, :]
    inside = np.where(row > 0.05)[0]
    if inside.size == 0:
        return None
    return int(inside[0]), int(inside[-1])


# Contour-tracing: find the outline edge at each y
def get_left_edge(sil_array, y):
    """Return the x-coordinate of the left silhouette edge at row y."""
    bounds = line_bounds_at_y(sil_array, y)
    if bounds:
        return bounds[0]
    return None

def get_right_edge(sil_array, y):
    """Return the x-coordinate of the right silhouette edge at row y."""
    bounds = line_bounds_at_y(sil_array, y)
    if bounds:
        return bounds[1]
    return None

def draw_text_along_path(img_output, path_points, text_stream, font, color, vertical_offset=0):
    """Draw text tokens along a path (list of (x, y) points)."""
    if not path_points or not text_stream:
        return
    si = 0
    for i, (x, y) in enumerate(path_points):
        if si >= len(text_stream):
            break
        token = text_stream[si % len(text_stream)]
        si += 1
        # slight angle based on local path direction
        if i < len(path_points) - 1:
            dx = path_points[i+1][0] - x
            dy = path_points[i+1][1] - y
            angle = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0
        else:
            angle = 0
        y_adj = int(y + vertical_offset)
        try:
            draw_word_rot(img_output, (int(x), y_adj), token, font, color, angle=angle, jitter=(0, 0))
        except Exception:
            pass
    return si


# Line-driven renderer: for each scanline, place tokens along the interior
while y < H - 6:
    ny = y / H
    # zone mapping with slightly different bands for facial shaping
    if ny < 0.20:
        zone = 'hair'; base_ls = 8   # tighter spacing
    elif ny < 0.42:
        zone = 'face'; base_ls = 8
    elif ny < 0.50:
        zone = 'neck'; base_ls = 8
    elif ny < 0.68:
        zone = 'chest'; base_ls = 10
    else:
        zone = 'general'; base_ls = 10

    bounds = line_bounds_at_y(sil, int(y))
    if not bounds:
        y += base_ls
        continue
    x_left, x_right = bounds
    line_width = x_right - x_left
    if line_width < 20:
        y += base_ls
        continue

    # per-line style randomness
    angle_jitter_base = random.uniform(-12, 12)
    density = min(1.0, max(0.1, (line_width / W)))

    stream = STREAM_TOKENS[zone]
    si = sidx[zone]
    x = x_left

    # face area: increase facial-feature placement frequency
    is_face_area = (zone == 'face')

    while x < x_right - 4:
        token = stream[si % len(stream)]
        si += 1

        # choose font size influenced by silhouette depth under center of token
        sx = min(W-1, max(0, int((x + min(x+20, x_right))/2)))
        sy = min(H-1, int(y))
        sv = float(sil[sy, sx])

        # size: deeper interior -> bigger + bolder
        if sv > 0.85:
            fsize = random.choice([22, 28, 34])
        elif sv > 0.6:
            fsize = random.choice([16, 18, 22])
        elif sv > 0.3:
            fsize = random.choice([12, 14, 16])
        else:
            fsize = random.choice([10, 12, 14])

        # face gets slightly smaller, denser words
        if is_face_area:
            fsize = max(10, int(fsize * 0.85))

        font = font_pool.get(fsize, get_font(fsize))

        # rotation more extreme for sketch-like feel
        angle = angle_jitter_base + random.uniform(-25, 25)
        if is_face_area:
            angle = angle_jitter_base + random.uniform(-8, 8)

        # color based on silhouette depth, but allow some warm highlights
        if sv > 0.75:
            color = (8,  28,  8, 255)
        elif sv > 0.45:
            color = (30, 70, 30, 255)
        else:
            # lighter strokes with occasional warm tint
            if random.random() < 0.08:
                color = (80, 40, 36, 255)
            else:
                color = (100, 140, 100, 255)

        # small jitter to avoid rigid grid
        jitter = (random.uniform(-2, 2), random.uniform(-1, 1))

        # draw token using rotated buffer and get width used
        try:
            used_w = draw_word_rot(output, (x, y), token, font, color, angle=angle, jitter=jitter)
        except Exception as e:
            # fallback: draw plain text
            try:
                draw.text((x, y), token, fill=color, font=font)
                used_w = font.getbbox(token)[2]
            except:
                used_w = int(fsize * 0.6)

        placed += 1
        x += max(used_w * (0.9 + random.uniform(-0.15, 0.25)), 3)

    sidx[zone] = si
    # vary line step so density changes vertically
    y += int(base_ls * (0.9 + random.uniform(-0.3, 0.6)))

print(f"Placed {placed} characters")

# ============================================================
# CONTOUR-FOLLOWING TEXT RENDERING
# Text undulates through hair, circles irises, outlines features
# ============================================================

contour_font_size = 12
contour_font = font_pool.get(contour_font_size, get_font(contour_font_size))
contour_color = (18, 45, 18, 255)

try:
    # LEFT HAIR ARC — trace left edge from top, curve around head
    left_hair_path = []
    for y_scan in range(20, 280, 8):
        x_left = get_left_edge(sil, y_scan)
        if x_left:
            left_hair_path.append((x_left - 5, y_scan))
    if left_hair_path:
        hair_tokens = STREAM_TOKENS['hair'][:50]
        draw_text_along_path(output, left_hair_path, hair_tokens, contour_font, contour_color, vertical_offset=-8)

    # RIGHT HAIR ARC
    right_hair_path = []
    for y_scan in range(20, 280, 8):
        x_right = get_right_edge(sil, y_scan)
        if x_right:
            right_hair_path.append((x_right + 5, y_scan))
    if right_hair_path:
        hair_tokens = STREAM_TOKENS['hair'][50:100]
        draw_text_along_path(output, right_hair_path, hair_tokens, contour_font, contour_color, vertical_offset=-8)

    # LEFT SHOULDER/ARM CONTOUR — trace left edge from shoulder down
    left_shoulder_path = []
    for y_scan in range(380, 800, 10):
        x_left = get_left_edge(sil, y_scan)
        if x_left:
            left_shoulder_path.append((x_left - 3, y_scan))
    if left_shoulder_path:
        shoulder_tokens = STREAM_TOKENS['chest'][:60]
        draw_text_along_path(output, left_shoulder_path, shoulder_tokens, contour_font, contour_color, vertical_offset=-5)

    # RIGHT SHOULDER/ARM CONTOUR
    right_shoulder_path = []
    for y_scan in range(380, 800, 10):
        x_right = get_right_edge(sil, y_scan)
        if x_right:
            right_shoulder_path.append((x_right + 3, y_scan))
    if right_shoulder_path:
        shoulder_tokens = STREAM_TOKENS['chest'][60:120]
        draw_text_along_path(output, right_shoulder_path, shoulder_tokens, contour_font, contour_color, vertical_offset=-5)

    # NOSE CONTOUR — vertical centerline from forehead to upper lip
    nose_centerline = [(cx, y) for y in range(130, 340, 6)]
    if nose_centerline:
        nose_tokens = STREAM_TOKENS['neck'][:30]
        draw_text_along_path(output, nose_centerline, nose_tokens, contour_font, (15, 40, 15, 255), vertical_offset=0)

except Exception as e:
    print(f"Warning: contour rendering failed: {e}")

# ============================================================
# DETAILED FEATURE CONTOURS — eyes, nostrils, ears, hands
# ============================================================

try:
    # Helper to draw circular text around a center point
    def draw_circular_text(img_output, cx_circle, cy_circle, radius, text_tokens, font, color, start_angle=0):
        """Draw text around a circle."""
        if not text_tokens:
            return
        num_chars = len(text_tokens)
        angle_step = 360.0 / num_chars if num_chars > 0 else 0
        for i, token in enumerate(text_tokens):
            angle = start_angle + (i * angle_step)
            rad = math.radians(angle)
            x = cx_circle + radius * math.cos(rad)
            y = cy_circle + radius * math.sin(rad)
            draw_word_rot(img_output, (int(x), int(y)), token, font, color, angle=angle, jitter=(0, 0))

    iris_font = font_pool.get(10, get_font(10))
    iris_color = (10, 30, 10, 255)

    # LEFT IRIS — circular text around left eye center
    eye_left_center_x = cx - 100
    eye_left_center_y = int(H * 0.18)
    iris_tokens = STREAM_TOKENS['face'][:20]
    draw_circular_text(output, eye_left_center_x, eye_left_center_y, 20, iris_tokens, iris_font, iris_color)

    # RIGHT IRIS — circular text around right eye center
    eye_right_center_x = cx + 100
    eye_right_center_y = int(H * 0.18)
    iris_tokens_r = STREAM_TOKENS['face'][20:40]
    draw_circular_text(output, eye_right_center_x, eye_right_center_y, 20, iris_tokens_r, iris_font, iris_color)

    # NOSTRIL ARCS — small curves at base of nose
    nostril_font = font_pool.get(9, get_font(9))
    nostril_color = (12, 35, 12, 255)

    # Left nostril
    left_nostril_arc = []
    for angle in range(-60, 60, 10):
        rad = math.radians(angle)
        x = (cx - 15) + 8 * math.cos(rad)
        y = int(H * 0.33) + 5 * math.sin(rad)
        left_nostril_arc.append((x, y))
    nostril_tokens_l = STREAM_TOKENS['neck'][:8]
    if left_nostril_arc:
        draw_text_along_path(output, left_nostril_arc, nostril_tokens_l, nostril_font, nostril_color, vertical_offset=0)

    # Right nostril
    right_nostril_arc = []
    for angle in range(-60, 60, 10):
        rad = math.radians(angle)
        x = (cx + 15) + 8 * math.cos(rad)
        y = int(H * 0.33) + 5 * math.sin(rad)
        right_nostril_arc.append((x, y))
    nostril_tokens_r = STREAM_TOKENS['neck'][8:16]
    if right_nostril_arc:
        draw_text_along_path(output, right_nostril_arc, nostril_tokens_r, nostril_font, nostril_color, vertical_offset=0)

    # MOUTH CREASE — sinuous curve along mouth line
    mouth_font = font_pool.get(11, get_font(11))
    mouth_color = (18, 45, 18, 255)
    mouth_crease = []
    for x_scan in range(cx - 60, cx + 61, 6):
        # sinuous curve: y dips slightly in middle
        normalized_x = (x_scan - (cx - 60)) / 120.0
        dip = 8 * math.sin(math.pi * normalized_x)
        y = int(H * 0.37) + int(dip)
        mouth_crease.append((x_scan, y))
    mouth_tokens = STREAM_TOKENS['face'][40:70]
    if mouth_crease:
        draw_text_along_path(output, mouth_crease, mouth_tokens, mouth_font, mouth_color, vertical_offset=0)

    # LEFT EAR CONTOUR — arc on left side of head
    left_ear = []
    for y_scan in range(120, 280, 8):
        x_ear = int(cx - 190 + 20 * math.sin((y_scan - 120) / 160.0 * math.pi))
        left_ear.append((x_ear, y_scan))
    ear_tokens = STREAM_TOKENS['hair'][:30]
    if left_ear:
        draw_text_along_path(output, left_ear, ear_tokens, contour_font, (16, 42, 16, 255), vertical_offset=-4)

    # RIGHT EAR CONTOUR
    right_ear = []
    for y_scan in range(120, 280, 8):
        x_ear = int(cx + 190 - 20 * math.sin((y_scan - 120) / 160.0 * math.pi))
        right_ear.append((x_ear, y_scan))
    ear_tokens_r = STREAM_TOKENS['hair'][30:60]
    if right_ear:
        draw_text_along_path(output, right_ear, ear_tokens_r, contour_font, (16, 42, 16, 255), vertical_offset=-4)

    # LEFT HAND/FINGERS — trace along interior left edge in lower body area
    left_hand_path = []
    for y_scan in range(700, 1000, 8):
        x_hand = get_left_edge(sil, y_scan)
        if x_hand:
            left_hand_path.append((x_hand + 3, y_scan))
    hand_tokens = STREAM_TOKENS['general'][:50]
    if left_hand_path:
        draw_text_along_path(output, left_hand_path, hand_tokens, contour_font, (14, 38, 14, 255), vertical_offset=-3)

    # RIGHT HAND/FINGERS
    right_hand_path = []
    for y_scan in range(700, 1000, 8):
        x_hand = get_right_edge(sil, y_scan)
        if x_hand:
            right_hand_path.append((x_hand - 3, y_scan))
    hand_tokens_r = STREAM_TOKENS['general'][50:100]
    if right_hand_path:
        draw_text_along_path(output, right_hand_path, hand_tokens_r, contour_font, (14, 38, 14, 255), vertical_offset=-3)

except Exception as e:
    print(f"Warning: detailed feature contours failed: {e}")

print(f"Placed {placed} characters")

# ============================================================
# FACIAL FEATURES — semantic placement of expressive vocabulary
# Eyes, nose, mouth with high-meaning text for visual + semantic impact
# ============================================================

eye_left_x = int(cx - 100)
eye_left_y = int(H * 0.18)
eye_right_x = int(cx + 100)
eye_right_y = int(H * 0.18)
nose_x = cx
nose_y = int(H * 0.30)
mouth_x = cx
mouth_y = int(H * 0.37)

eye_vocabulary = ['seeing', 'knowing', 'beholding', 'perceiving', 'witnessing', 'aware']
nose_vocabulary = ['bearing', 'witness', 'solid', 'voice', 'truth', 'presence']
mouth_vocabulary = ['speaks', 'truth', 'love', 'justice', 'peace', 'reconciles', 'I am']

try:
    # LEFT EYE — small, dense
    eye_font_sz = 14
    eye_font = font_pool.get(eye_font_sz, get_font(eye_font_sz))
    for i in range(3):
        word = random.choice(eye_vocabulary)
        angle = random.uniform(-8, 8)
        draw_word_rot(output, (eye_left_x - 20 + i*15, eye_left_y - 5), word, eye_font, (12, 35, 12, 255), angle=angle, jitter=(0, 0))

    # RIGHT EYE
    for i in range(3):
        word = random.choice(eye_vocabulary)
        angle = random.uniform(-8, 8)
        draw_word_rot(output, (eye_right_x - 20 + i*15, eye_right_y - 5), word, eye_font, (12, 35, 12, 255), angle=angle, jitter=(0, 0))

    # NOSE — vertical emphasis
    nose_font_sz = 16
    nose_font = font_pool.get(nose_font_sz, get_font(nose_font_sz))
    for i in range(2):
        word = random.choice(nose_vocabulary)
        angle = random.uniform(-5, 5) if i == 0 else random.uniform(175, 185)
        draw_word_rot(output, (nose_x - 30, nose_y + i*20), word, nose_font, (15, 40, 15, 255), angle=angle, jitter=(0, 0))

    # MOUTH — widest, open variation
    mouth_font_sz = 18
    mouth_font = font_pool.get(mouth_font_sz, get_font(mouth_font_sz))
    for i in range(3):
        word = random.choice(mouth_vocabulary)
        angle = random.uniform(-12, 12)
        draw_word_rot(output, (mouth_x - 50 + i*40, mouth_y), word, mouth_font, (20, 50, 20, 255), angle=angle, jitter=(0, 0))

except Exception as e:
    print(f"Warning: facial feature rendering failed: {e}")

# ============================================================
# ZONE 5 — NAME INTEGRATION
# HALLEY as vertical spine: load-bearing architecture, not label.
# Spec: runs vertically from base of neck downward along centerline.
# ============================================================
name_font = get_font(48)
ly = 490   # start at neck base
for letter in "HALLEY":
    try:
        bb = name_font.getbbox(letter)
        lw, lh = bb[2]-bb[0], bb[3]-bb[1]
    except:
        lw, lh = 34, 56
    draw.text((cx - lw//2, ly), letter, fill=(20, 50, 20), font=name_font)
    ly += lh + 4

# ============================================================
# SAVE
# ============================================================
# Convert RGBA back to RGB for PNG output
output_rgb = output.convert('RGB')
output_rgb.save('HALLEY_portrait.png', dpi=(300, 300))
print(f"Saved: HALLEY_portrait.png ({W}x{H}px, 300dpi)")
print("Open in Illustrator or any image viewer to examine.")
