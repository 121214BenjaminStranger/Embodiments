"""
MILO — Typographic Portrait Generator
Phase 2 · AI Persona Spec Portfolio
Ben Pranger · AI Persona Design

Generates a text-density portrait of MILO: a human-form figure constructed
entirely from semantically chosen English-language text. No illustration.
No skin tone. The text IS the form.

MILO's formal argument: forward motion. The figure is caught mid-lean,
pressing toward the viewer. Density is directional — heaviest at the leading
edge (face, right shoulder), lightest at the trailing edge (back of head,
left shoulder). The text carries the momentum.

Palette: press red (vs. HALLEY's forest green, PAINE's constitutional blue)
Name: MILO runs horizontally on the jawline — the anatomical site of
speaking out, of holding the jaw set, of not backing down.

Usage:
    pip install Pillow numpy scipy
    python MILO_generator.py

Output: MILO_embodiment_v1.png (850x1100px, 300dpi)

Parameters to experiment with:
    - BACKGROUND: change the paper color
    - INK depth values: adjust the red palette
    - STREAMS: modify the text corpus for each zone
    - LEAN_OFFSET: controls the lateral shift of the figure (pixels)
    - font size per zone: larger = more readable words, less dense
    - line_step: smaller = denser, more shadow depth
    - sigma in build_silhouette: larger = softer figure edges
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import random
import re
import math

# ============================================================
# CONFIGURATION — adjust these to iterate
# ============================================================
W, H = 850, 1100          # canvas: 8.5x11 portrait ratio
BACKGROUND = (252, 251, 249)   # warm near-white paper, slightly warmer than HALLEY
SEED = 17                       # change for different random arrangement
LEAN_OFFSET = 28               # pixels: how far the head is shifted right of center (forward lean)

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# TEXT CORPUS — MILO's five semantic zones
#
# Zone 1: Outline / Contour (LEADING EDGE)
#    — AP Style doctrine, Tribune masthead language, objectivity-framework.
#      This is the framework Milo left. It defines the boundary.
#      Trailing edge: sparser, dissolving, shorter phrases.
#
# Zone 2: Chest & Torso Interior
#    — Active practice: podcast episode register, field notes, Breaking Blocks vocabulary.
#      The accumulated weight of the work.
#
# Zone 3: Jawline (Name Integration & Voice Site)
#    — Intellectual lineage: Coates, Didion, Maron, Baldwin, Wells.
#      MILO runs horizontally here. The hinge between framework and voice.
#
# Zone 4: Face (Eyes, Brow, Mouth)
#    — Eyes/brow: diagnostic vocabulary — false balance, erasure, manufactured neutrality.
#      Mouth: testimony and first-person authority — I was there, I am telling you.
#      Highest density, forward pressure.
#
# Zone 5: Hair & Crown
#    — Lightest register. Titles of Coates essays, Didion chapter headings,
#      Maron fragments. Float. The intellectual inheritance in the wake of the lean.
# ============================================================

STREAMS = {

    # Zone 4 — Face: highest density, most charged language
    # Eyes/brow: diagnostic vocabulary
    # Mouth: testimony, first-person authority
    'face': (
        "falsebalancewhosobjectivityerasuresuppressionmanufacturedneutrality"
        "whoisprotectedbythisframingbothsidesofwhatexactlythestorythestorywonttell"
        "Iwasthere Iamtellingyou thisiswhatIsaw ontherecord Iputmynameonit witness"
        "falsebalancewhoseobjectivitywhobenefitsfromthisframingreportbeforearguing"
        "thestorybelowthestorywhosesilenceisthisprotectingIwasintheroomIsaw it"
        "notbothsidesthisoneside andawallwitnessIamtellingyouIputmynameonit"
    ) * 28,

    # Zone 3 — Jawline: intellectual lineage as hinge between framework and voice
    'jawline': (
        "CoatesDidionMaronBaldwinIdaBWellsHunterSThompsonRebeccaSolnit"
        "BreakingBlocksMinneapolisTribuneformerMizzoujournalismGeorgeFloyd"
        "ReneeGoodAlexPrettiMinneapolisUpperMidwestactivistjournalismpositionality"
        "transparencyoverfalseojectivitydeclaredlenstheexaminedposition"
    ) * 16,

    # Zone 2 — Chest & Torso: podcast register, field notes, Breaking Blocks
    'chest': (
        "breakingblocksepisode47ontherecordIwasintheroom Minneapolis2020"
        "whodoesthisstoryprotectthesourceaskednottobenamed andIsaidfine tellmeanyway"
        "thisisnotbothsidesthisoneside andawallMinneapolis2026"
        "notepadnotebook fieldnotesonthegroundsourcesaysbackgroundonly"
        "whosestoryisthiswhosesilencewhobenefitswhopaysthecoststhereal question"
        "activistjournalismpositionaldeclaredlensnotobjectivitytransparency"
        "KernerCommissionreportAPstylebookevolutionnewsroomdemographics1960s"
        "storiesnotcoveredsourcesdeemedanecdotalobjectivityasapower arrangement"
    ) * 20,

    # Zone 5 — Hair & Crown: lightest, most open — trailing release
    # Titles and fragments from Coates, Didion, Maron
    'hair': (
        "TheCaseforReparations SlouchingTowardsBethlehem WeTellOurselvesStories"
        "BetweentheWorldandMe WTFwithMarcMaron TheYearofMagicalThinking"
        "NotesofaNativeSon TheWhiteAlbum TheColossusofNewYork"
        "BetweentheWorldandMeTaNehhisiCoatesJoanDidionMarcMaronMarcMaron"
        "SlouchingBethlehemCentercannothold MagicalThinkinggrief certainty"
    ) * 20,

    # General / lower body — continuation of the work
    'general': (
        "activistjournalismdeclaredpositionality transparencynotobjectivity"
        "amplifyingunderrepresentedvoicescorrectionnotpropaganda"
        "Minneapolisuppermidwestracegenderincomeinequality climatecultural"
        "breakingblocksindependentpodcast challengeconventional wisdoms"
        "therearestillunreportedstoriesinthisroomthemicisalwayson"
        "Ibringmybodyandbiographytothisstory itisntdecorationitistheargument"
    ) * 16,

    # Zone 1 — Outline (leading edge): AP Style doctrine, objectivity framework
    # The framework MILO left — defines the boundary, the skin of the portrait
    'outline': (
        "attributionrequiredbothsidesoftheissue thepaperof record"
        "objectivityasprofessionalstandard unnamedsourcesfamiliar withthematter"
        "nocommentwasreturned EditorsnotethisstoryhasbeenupdatedAPstyle"
        "balancedcoverage representativevoices themasthead correction"
        "editorsnoteversion2asinpreviouslyreported objectivityprotocol"
    ) * 12,
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
# SILHOUETTE — MILO's forward-leaning figure
#
# The pose is a forward lean caught at its most committed moment.
# Head slightly ahead of shoulders (shifted right/forward by LEAN_OFFSET).
# Leading (right) shoulder dropped and pressing forward.
# Trailing (left) shoulder pulled back.
# One hand, if visible, in gesture — open, mid-emphasis.
#
# Asymmetry is intentional: this figure is leaving its left edge behind.
# ============================================================

def build_silhouette(W, H):
    """
    Builds MILO's asymmetric, forward-leaning figure mask.
    Returns a float array 0.0-1.0 where 1.0 = deepest figure interior.
    The gaussian blur (sigma) controls edge softness.
    
    Key asymmetry: right shoulder (viewer's left) is the LEADING edge —
    dropped, compressed, pressing forward. Left shoulder trails.
    In the density rendering, the leading edge gets denser text.
    """
    canvas = Image.new('L', (W, H), 255)
    d = ImageDraw.Draw(canvas)
    
    # Head center: shifted right of canvas center to simulate lean
    hcx = W // 2 + LEAN_OFFSET   # head center x
    cx = W // 2                    # canvas center x (torso anchors near center)

    # Hair: crown + flow. The lean means hair trails slightly left at crown.
    # Leading edge (right side of hair) is denser in rendering.
    d.ellipse([hcx-152, 22, hcx+138, 262], fill=0)
    # Left hair flow — trailing, lighter in rendering
    d.polygon([(hcx-140, 82), (hcx-200, 198), (hcx-215, 345),
               (hcx-185, 480), (hcx-148, 560),
               (hcx-108, 545), (hcx-72, 415),
               (hcx-102, 255), (hcx-112, 162), (hcx-128, 108)], fill=0)
    # Right hair flow — leading, compressed
    d.polygon([(hcx+110, 82), (hcx+162, 178), (hcx+170, 322),
               (hcx+148, 425), (hcx+105, 448),
               (hcx+78, 362), (hcx+86, 225), (hcx+96, 145)], fill=0)

    # Head / face oval — shifted right (forward lean)
    d.ellipse([hcx-118, 88, hcx+118, 348], fill=0)

    # Neck — anatomical site of voice, MILO's jawline name zone
    # Slight rightward tilt in neck suggests the forward lean
    d.polygon([(hcx-38, 320), (hcx+42, 320),
               (hcx+52, 415), (hcx-28, 420)], fill=0)

    # Shoulders: asymmetric for the lean
    # RIGHT shoulder (leading): dropped, lower y, compressed, pressing forward
    # LEFT shoulder (trailing): higher, pulled back
    #
    # The torso anchors near canvas center (cx) while head is at hcx (shifted right)
    # This creates the diagonal body line of the forward lean.
    d.polygon([
        (cx - 228, 408),          # left shoulder edge (trailing, high)
        (cx + 240, 430),          # right shoulder edge (leading, slightly lower)
        (cx + 225, 568),
        (cx + 172, 658),
        (cx + 148, 808),
        (cx + 132, 1005),
        (cx - 122, 1005),
        (cx - 138, 808),
        (cx - 158, 658),
        (cx - 208, 568),
    ], fill=0)

    # Left arm (trailing arm, pulled back) — less defined, sparser in rendering
    d.ellipse([cx - 278, 448, cx - 208, 662], fill=0)

    # Right arm (leading arm, pressed forward) — hand in gesture
    # Slightly larger and lower to suggest the forward reach
    d.ellipse([cx + 210, 432, cx + 285, 648], fill=0)

    # Right hand — visible at lower frame, open gesture, pointing
    # A rounded polygon suggesting an open palm / gesture
    d.ellipse([cx + 218, 635, cx + 275, 720], fill=0)

    arr = np.array(canvas).astype(float)
    figure = (arr < 128).astype(float)
    # sigma=10: moderate edge softness.
    return gaussian_filter(figure, sigma=10)


# ============================================================
# DIRECTIONAL DENSITY MAP
# MILO's shadow/density follows the forward lean, not a vertical gradient.
# Leading edge (right side, face) = denser text, darker ink.
# Trailing edge (left side, back of head) = sparser text, lighter ink.
# ============================================================

def leading_edge_weight(x, y, W, H, lean_offset=LEAN_OFFSET):
    """
    Returns a float 0.0–1.0 representing how 'leading' a point is.
    Points on the right side (leading edge) return values closer to 1.0.
    Points on the left side (trailing edge) return values closer to 0.0.
    The gradient also favors the upper face area.
    """
    # Horizontal component: right = leading (1.0), left = trailing (0.0)
    x_weight = x / W

    # Vertical component: face area (upper) = slightly more leading pressure
    face_band = 1.0 - abs((y / H) - 0.22) / 0.35
    face_weight = max(0.0, min(1.0, face_band)) * 0.25

    return min(1.0, x_weight * 0.75 + face_weight)


# ============================================================
# FONT LOADER
# ============================================================

def get_font(size):
    """Load system font. Falls back gracefully."""
    for path in [
        # Windows
        'C:\\Windows\\Fonts\\arial.ttf',
        'C:\\Windows\\Fonts\\calibri.ttf',
        'C:\\Windows\\Fonts\\seguiemj.ttf',
        # Linux
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()


def next_output_filename(prefix="MILO_portrait", ext="png"):
    files = [f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.' + ext)]
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)\.{re.escape(ext)}$')
    nums = [int(m.group(1)) for f in files if (m := pattern.match(f))]
    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}_{next_num:02d}.{ext}"


# ============================================================
# PRESS RED PALETTE
# MILO's color signature. Forest green belongs to HALLEY.
# These are depth-mapped: darker (more saturated) for interior mass,
# lighter/cooler red for outline and trailing edges.
#
# ink_deep:    The interior mass at maximum density
# ink_mid:     Mid-tone — most of the face and chest
# ink_light:   Trailing edge, hair crown, sparse outline
# ink_accent:  Occasional warm dark accent (replaces HALLEY's warm tint)
# ============================================================
INK = {
    'deep':    (90,  10,  10, 255),   # near-black red, deepest interior
    'rich':    (135, 18,  18, 255),   # rich press red
    'mid':     (168, 40,  38, 255),   # mid red — working register
    'warm':    (192, 72,  60, 255),   # warmer, slightly lighter
    'light':   (210, 118, 105, 255),  # trailing edge, lighter strokes
    'pale':    (228, 158, 148, 255),  # near-outline, palest strokes
    'outline': (140, 24,  20, 255),   # contour / boundary layer
}


def ink_for_depth(sv, leading_w, is_face=False, is_trailing=False):
    """
    Select ink color based on silhouette depth (sv) and leading-edge weight.
    sv: 0.0 (edge) to 1.0 (deep interior)
    leading_w: 0.0 (trailing left) to 1.0 (leading right)
    """
    if is_trailing:
        # Trailing edge: always lighter, more open
        if sv > 0.6:
            return INK['warm']
        elif sv > 0.3:
            return INK['light']
        else:
            return INK['pale']

    # Interior / leading edge
    adjusted = sv * (0.7 + 0.3 * leading_w)
    if adjusted > 0.80:
        return INK['deep']
    elif adjusted > 0.60:
        return INK['rich']
    elif adjusted > 0.40:
        return INK['mid']
    elif adjusted > 0.20:
        return INK['warm']
    else:
        return INK['light']


# ============================================================
# MAIN GENERATION LOOP
# ============================================================

print("Building MILO silhouette...")
sil = build_silhouette(W, H)

font_sizes = [10, 12, 14, 16, 18, 22, 28, 34]
font_pool = {s: get_font(s) for s in font_sizes}
sidx = {k: 0 for k in STREAMS}

output = Image.new('RGBA', (W, H), (*BACKGROUND, 255))
draw = ImageDraw.Draw(output)
placed = 0
cx = W // 2
hcx = cx + LEAN_OFFSET   # head center x for facial features

print("Generating typographic portrait...")


# ============================================================
# UTILITY FUNCTIONS — shared with HALLEY, adapted for MILO
# ============================================================

def draw_word_rot(img_output, pos, word, font, fill, angle=0, jitter=(0, 0)):
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
    im = Image.new('RGBA', (buf_w, buf_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((4, 2), word, font=font, fill=fill)
    if angle != 0:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=1)
    ox = int(pos[0] + jitter[0])
    oy = int(pos[1] + jitter[1])
    ox = max(0, min(ox, img_output.width - 1))
    oy = max(0, min(oy, img_output.height - 1))
    img_output.paste(im, (ox, oy), im)
    return im.size[0]


def line_bounds_at_y(sil_array, y):
    if y < 0 or y >= sil_array.shape[0]:
        return None
    row = sil_array[y, :]
    inside = np.where(row > 0.05)[0]
    if inside.size == 0:
        return None
    return int(inside[0]), int(inside[-1])


def get_left_edge(sil_array, y):
    bounds = line_bounds_at_y(sil_array, y)
    if bounds:
        return bounds[0]
    return None


def get_right_edge(sil_array, y):
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
        if i < len(path_points) - 1:
            dx = path_points[i + 1][0] - x
            dy = path_points[i + 1][1] - y
            angle = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0
        else:
            angle = 0
        y_adj = int(y + vertical_offset)
        try:
            draw_word_rot(img_output, (int(x), y_adj), token, font, color, angle=angle, jitter=(0, 0))
        except Exception:
            pass
    return si


def generate_wave_path(x_left, x_right, y, steps=20, amplitude=5, wavelength=1.5, phase=0.0):
    """Generate a gentle wavy path across an interior scanline."""
    if x_right <= x_left:
        return []
    points = []
    width = x_right - x_left
    for i in range(steps + 1):
        t = i / max(1, steps)
        x = x_left + t * width
        offset = amplitude * math.sin(2 * math.pi * (t * wavelength + phase))
        points.append((x, int(y + offset)))
    return points


def draw_text_flow(img_output, path_points, text_stream, zone, color, min_size=10, max_size=24, jitter_scale=1.0, rotation_range=18):
    """Draw tokens along a curving path with varied size, rotation and jitter.
    For MILO, rotation_range is tighter on the face (urgency, directionality)
    and looser on the trailing body."""
    if not path_points or not text_stream:
        return 0
    si = 0
    for i, (x, y) in enumerate(path_points):
        if si >= len(text_stream):
            break
        token = text_stream[si % len(text_stream)]
        si += 1

        x_clamped = min(W - 1, max(0, int(x)))
        y_clamped = min(H - 1, max(0, int(y)))
        sv = float(sil[y_clamped, x_clamped])
        lw = leading_edge_weight(x_clamped, y_clamped, W, H)

        if sv > 0.80:
            fsize = random.choice([18, 20, 24])
        elif sv > 0.55:
            fsize = random.choice([14, 16, 18])
        elif sv > 0.30:
            fsize = random.choice([12, 14, 16])
        else:
            fsize = random.choice([10, 12, 14])

        # Face zone: tighter, denser, smaller for maximum compression
        if zone == 'face':
            fsize = max(10, int(fsize * 0.82))
        # Hair / trailing: slightly more open
        if zone == 'hair':
            fsize = max(10, int(fsize * 1.05))

        fsize = max(min_size, min(fsize, max_size))
        font = font_pool.get(fsize, get_font(fsize))

        if i < len(path_points) - 1:
            dx = path_points[i + 1][0] - x
            dy = path_points[i + 1][1] - y
            angle = math.degrees(math.atan2(dy, dx))
        else:
            angle = 0
        if rotation_range:
            angle += random.uniform(-rotation_range, rotation_range)

        jitter = (random.uniform(-jitter_scale, jitter_scale),
                  random.uniform(-jitter_scale * 0.8, jitter_scale * 0.8))

        draw_word_rot(img_output, (x, y), token, font, color, angle=angle, jitter=jitter)
    return si


def draw_spiral_text(img_output, center_x, center_y, start_radius, turns, tokens, color, font_size, start_angle=0, clockwise=True):
    """Draw a spiral band of text around a point."""
    radius = start_radius
    angle = start_angle
    dr = start_radius / max(12, len(tokens))
    dtheta = (360.0 * turns) / max(1, len(tokens))
    for token in tokens:
        rad = math.radians(angle)
        x = center_x + radius * math.cos(rad)
        y = center_y + radius * math.sin(rad)
        font = font_pool.get(font_size, get_font(font_size))
        draw_word_rot(img_output, (x, y), token, font, color,
                      angle=angle + (90 if clockwise else -90), jitter=(0, 0))
        radius += dr
        angle += dtheta if clockwise else -dtheta


def draw_eye_cluster(img_output, cx_eye, cy_eye, radius, tokens, font_size, color):
    """Circular eye cluster — MILO's eyes hold the diagnostic vocabulary."""
    font = font_pool.get(font_size, get_font(font_size))
    for layer in range(3):
        ring_radius = radius - layer * 6
        step = 360 // (8 + layer * 2)
        for angle in range(0, 360, step):
            rad = math.radians(angle + random.uniform(-8, 8))
            x = cx_eye + ring_radius * math.cos(rad)
            y = cy_eye + ring_radius * math.sin(rad)
            token = random.choice(tokens)
            draw_word_rot(img_output, (x, y), token, font, color,
                          angle=angle + 100 + random.uniform(-20, 20),
                          jitter=(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)))


def draw_nose_phrase(img_output, cx_center, y_top, y_bottom, phrase_tokens, font_size, color):
    """Vertical nose phrase — MILO's nose carries the witness register."""
    path = []
    steps = 14
    for i in range(steps + 1):
        t = i / steps
        # Slight right lean in nose ridge to echo the forward lean
        x = cx_center + 6 * math.sin(t * math.pi * 1.5) + t * 4
        y = y_top + t * (y_bottom - y_top)
        path.append((x, y))
    draw_text_along_path(img_output, path, phrase_tokens, get_font(font_size), color, vertical_offset=0)


def draw_face_sketch_marks(img_output, cx_center, eye_y, nose_top, nose_bottom):
    """MILO's face sketching — tighter, more angular than HALLEY's."""
    sketch_tokens = STREAM_TOKENS['face'][:16]
    sketch_font = get_font(10)
    sketch_color = INK['deep']

    # Eyelid arcs — tighter than HALLEY's, more pressed forward
    left_eyelid = [(cx_center - 95 + i * 7, eye_y - 14 + 4 * math.sin(i * 0.55))
                   for i in range(10)]
    right_eyelid = [(cx_center + 78 + i * 7, eye_y - 14 + 4 * math.sin(i * 0.55))
                    for i in range(10)]
    draw_text_along_path(img_output, left_eyelid, sketch_tokens, sketch_font, sketch_color)
    draw_text_along_path(img_output, right_eyelid, sketch_tokens, sketch_font, sketch_color)

    # Nose ridge — slight diagonal for the lean
    nose_ridge = []
    for i in range(10):
        t = i / 9.0
        x = cx_center + 4 + 6 * math.sin(t * math.pi * 1.2)
        y = nose_top + t * (nose_bottom - nose_top)
        nose_ridge.append((x, y))
    draw_text_along_path(img_output, nose_ridge, sketch_tokens, sketch_font, sketch_color)

    # Face side marks — angular, suggesting the jaw set
    left_face = [(cx_center - 108 + 3 * math.sin(i * 0.9), eye_y + 18 + i * 9)
                 for i in range(6)]
    right_face = [(cx_center + 108 + 5 * math.sin(i * 0.9), eye_y + 14 + i * 9)
                  for i in range(6)]
    draw_text_along_path(img_output, left_face, sketch_tokens, sketch_font, sketch_color)
    draw_text_along_path(img_output, right_face, sketch_tokens, sketch_font, sketch_color)


def draw_gesture_hand(img_output, cx_center, y_start, tokens, color):
    """
    Draw the leading hand in gesture — open, pointing, mid-emphasis.
    This is unique to MILO's pose (cf. PAINE's raised hand).
    The hand traces a forward-pressing arc from the wrist downward.
    """
    hand_font = font_pool.get(11, get_font(11))
    # Wrist arc
    for angle in range(-40, 50, 8):
        rad = math.radians(angle)
        x = cx_center + 248 + 18 * math.cos(rad)
        y = y_start + 30 + 20 * math.sin(rad)
        token = random.choice(tokens[:12])
        draw_word_rot(img_output, (x, y), token, hand_font, color,
                      angle=angle + random.uniform(-10, 10), jitter=(1, 1))

    # Finger traces — 3 fingers in gesture
    finger_offsets = [(-8, 0), (0, -6), (8, -10)]
    for fo_x, fo_y in finger_offsets:
        finger_path = []
        for i in range(8):
            x = cx_center + 240 + fo_x + i * 5 * math.cos(math.radians(-15 + fo_y))
            y = y_start + 20 + fo_y + i * 6 * math.sin(math.radians(-15 + fo_y))
            finger_path.append((x, y))
        draw_text_along_path(img_output, finger_path, tokens[:16], hand_font, color)


# ============================================================
# LINE-DRIVEN RENDERER
# For each scanline, place tokens along the interior.
# MILO's key addition: density is modulated by leading_edge_weight,
# not just by silhouette depth — right side gets more/darker text.
# ============================================================

y = 10
while y < H - 6:
    ny = y / H
    # Zone mapping — note 'jawline' is a narrow band around 0.42–0.50
    if ny < 0.20:
        zone = 'hair'; base_ls = 9
    elif ny < 0.42:
        zone = 'face'; base_ls = 8
    elif ny < 0.50:
        zone = 'jawline'; base_ls = 8
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

    angle_jitter_base = random.uniform(-10, 10)
    density = min(1.0, max(0.1, (line_width / W)))

    stream = STREAM_TOKENS[zone]
    si = sidx[zone]
    x = x_left

    is_face_area = (zone == 'face')

    if line_width > 40 and random.random() < 0.38:
        phase = random.random() * 2 * math.pi
        path = generate_wave_path(
            x_left + 4, x_right - 4, y,
            steps=max(12, int(line_width / 18)),
            amplitude=3 + 4 * random.random(),
            wavelength=1.4 + random.random() * 0.6,
            phase=phase
        )
        # Leading edge gets deep ink, trailing gets lighter
        lw_mid = leading_edge_weight((x_left + x_right) / 2, y, W, H)
        if lw_mid > 0.6:
            path_color = INK['deep']
        elif lw_mid > 0.4:
            path_color = INK['rich']
        else:
            path_color = INK['warm']
        used = draw_text_flow(output, path, stream[si:si + len(path) * 2],
                              zone, path_color, min_size=10, max_size=22, jitter_scale=2.0)
        si += used
        placed += used
    else:
        while x < x_right - 4:
            token = stream[si % len(stream)]
            si += 1

            sx = min(W - 1, max(0, int((x + min(x + 20, x_right)) / 2)))
            sy = min(H - 1, int(y))
            sv = float(sil[sy, sx])
            lw = leading_edge_weight(sx, sy, W, H)
            is_trailing = (lw < 0.35)

            # Size: deeper interior or more leading -> bigger
            adjusted_sv = sv * (0.65 + 0.35 * lw)
            if adjusted_sv > 0.85:
                fsize = random.choice([22, 28, 34])
            elif adjusted_sv > 0.60:
                fsize = random.choice([16, 18, 22])
            elif adjusted_sv > 0.35:
                fsize = random.choice([12, 14, 16])
            else:
                fsize = random.choice([10, 12, 14])

            # Face: smaller, denser — maximum forward pressure
            if is_face_area:
                fsize = max(10, int(fsize * 0.82))

            font = font_pool.get(fsize, get_font(fsize))

            # Rotation: face area tighter (urgency), body more varied
            if is_face_area:
                angle = angle_jitter_base + random.uniform(-7, 7)
            elif is_trailing:
                angle = angle_jitter_base + random.uniform(-18, 18)
            else:
                angle = angle_jitter_base + random.uniform(-22, 22)

            # Ink color via palette + leading edge weight
            color = ink_for_depth(sv, lw, is_face=is_face_area, is_trailing=is_trailing)

            jitter = (random.uniform(-2, 2), random.uniform(-1, 1))

            try:
                used_w = draw_word_rot(output, (x, y), token, font, color,
                                       angle=angle, jitter=jitter)
            except Exception:
                try:
                    draw.text((x, y), token, fill=color, font=font)
                    used_w = font.getbbox(token)[2]
                except:
                    used_w = int(fsize * 0.6)

            placed += 1
            x += max(used_w * (0.9 + random.uniform(-0.15, 0.25)), 3)

    sidx[zone] = si
    y += int(base_ls * (0.9 + random.uniform(-0.3, 0.6)))

print(f"Placed {placed} characters (main scan)")


# ============================================================
# CONTOUR-FOLLOWING TEXT RENDERING
# Text undulates through hair, outlines features, traces the body.
# MILO's leading contour: denser ink, AP Style doctrine.
# Trailing contour: lighter, dissolving.
# ============================================================

contour_font_size = 12
contour_font = font_pool.get(contour_font_size, get_font(contour_font_size))

try:
    # LEFT HAIR ARC — trailing edge, lighter, dissolving
    left_hair_path = []
    for y_scan in range(20, 270, 9):
        x_left = get_left_edge(sil, y_scan)
        if x_left:
            left_hair_path.append((x_left - 6, y_scan))
    if left_hair_path:
        draw_text_along_path(output, left_hair_path, STREAM_TOKENS['hair'][:50],
                             contour_font, INK['light'], vertical_offset=-8)

    # RIGHT HAIR ARC — leading edge, richer ink, AP Style outline language
    right_hair_path = []
    for y_scan in range(20, 270, 9):
        x_right = get_right_edge(sil, y_scan)
        if x_right:
            right_hair_path.append((x_right + 5, y_scan))
    if right_hair_path:
        draw_text_along_path(output, right_hair_path, STREAM_TOKENS['outline'][:50],
                             contour_font, INK['outline'], vertical_offset=-8)

    # LEFT SHOULDER/ARM CONTOUR — trailing, lighter
    left_shoulder_path = []
    for y_scan in range(380, 800, 10):
        x_left = get_left_edge(sil, y_scan)
        if x_left:
            left_shoulder_path.append((x_left - 3, y_scan))
    if left_shoulder_path:
        draw_text_along_path(output, left_shoulder_path, STREAM_TOKENS['outline'][:60],
                             contour_font, INK['light'], vertical_offset=-5)

    # RIGHT SHOULDER/ARM CONTOUR — leading, denser, AP Style doctrine
    right_shoulder_path = []
    for y_scan in range(380, 800, 10):
        x_right = get_right_edge(sil, y_scan)
        if x_right:
            right_shoulder_path.append((x_right + 3, y_scan))
    if right_shoulder_path:
        draw_text_along_path(output, right_shoulder_path, STREAM_TOKENS['outline'][60:120],
                             contour_font, INK['outline'], vertical_offset=-5)

    # NOSE CENTERLINE — slight right lean per the pose
    nose_centerline = [(hcx + int(i * 0.3), y) for i, y in
                       enumerate(range(128, 332, 6))]
    if nose_centerline:
        draw_text_along_path(output, nose_centerline, STREAM_TOKENS['jawline'][:30],
                             contour_font, INK['rich'], vertical_offset=0)

except Exception as e:
    print(f"Warning: contour rendering failed: {e}")


# ============================================================
# FACIAL FEATURE RENDERING
# ============================================================

# Facial anchor points — all adjusted for the forward lean (hcx)
eye_left_x  = int(hcx - 92)
eye_left_y  = int(H * 0.185)
eye_right_x = int(hcx + 92)
eye_right_y = int(H * 0.180)   # right eye (leading) slightly lower in the lean
nose_x = hcx
nose_y = int(H * 0.295)
mouth_x = hcx
mouth_y = int(H * 0.365)

# Eye vocabulary: MILO's diagnostic register
eye_vocabulary = ['falsebalance', 'erasure', 'suppression', 'whoseobjectivity',
                  'framing', 'manufactured', 'neutrality', 'witness']
# Mouth vocabulary: testimony, first-person authority
mouth_vocabulary = ['Iwasthere', 'Iamtellingyou', 'ontherecord', 'Iputmynameonit',
                    'witness', 'thisiswhatIsaw', 'Iwasinthere']

try:
    # LEFT EYE — diagnostic vocabulary, slightly smaller (trailing)
    draw_eye_cluster(output, eye_left_x, eye_left_y, 24,
                     eye_vocabulary, 10, INK['rich'])

    # RIGHT EYE — same vocabulary, slightly larger/denser (leading)
    draw_eye_cluster(output, eye_right_x, eye_right_y, 27,
                     eye_vocabulary, 11, INK['deep'])

    # Minimal sketch marks around eyes and face
    draw_face_sketch_marks(output, hcx, int(H * 0.182),
                           int(H * 0.225), int(H * 0.335))

    # NOSE — vertical emphasis, slight rightward lean
    nose_phrase_tokens = ['witness', 'Isaw', 'ontherecord', 'Iwasintheroom', 'testimony']
    draw_nose_phrase(output, hcx, int(H * 0.225), int(H * 0.335),
                     nose_phrase_tokens, 15, INK['rich'])

    # MOUTH — widest, testimony register, open in mid-sentence
    mouth_font_sz = 17
    mouth_font = font_pool.get(mouth_font_sz, get_font(mouth_font_sz))
    for i in range(4):
        word = random.choice(mouth_vocabulary)
        angle = random.uniform(-10, 10)
        draw_word_rot(output, (mouth_x - 55 + i * 35, mouth_y),
                      word, mouth_font, INK['deep'], angle=angle, jitter=(0, 0))

    # Mouth crease — sinuous curve, testimony text
    mouth_crease = []
    for x_scan in range(hcx - 58, hcx + 62, 5):
        normalized_x = (x_scan - (hcx - 58)) / 120.0
        dip = 7 * math.sin(math.pi * normalized_x)
        y = int(H * 0.370) + int(dip)
        mouth_crease.append((x_scan, y))
    mouth_tokens = STREAM_TOKENS['face'][40:70]
    if mouth_crease:
        draw_text_along_path(output, mouth_crease, mouth_tokens,
                             font_pool.get(10, get_font(10)), INK['deep'], vertical_offset=0)

except Exception as e:
    print(f"Warning: facial feature rendering failed: {e}")


# ============================================================
# DETAILED FEATURE CONTOURS — eyes, nostrils, ears
# ============================================================

try:
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
            draw_word_rot(img_output, (int(x), int(y)), token, font, color,
                          angle=angle, jitter=(0, 0))

    iris_font = font_pool.get(10, get_font(10))

    # LEFT IRIS — trailing, slightly lighter
    draw_circular_text(output, eye_left_x, eye_left_y, 18,
                       STREAM_TOKENS['face'][:20], iris_font, INK['mid'])

    # RIGHT IRIS — leading, denser
    draw_circular_text(output, eye_right_x, eye_right_y, 20,
                       STREAM_TOKENS['face'][20:40], iris_font, INK['deep'])

    # NOSTRIL ARCS
    nostril_font = font_pool.get(9, get_font(9))
    left_nostril_arc = []
    for angle in range(-55, 55, 10):
        rad = math.radians(angle)
        x = (hcx - 14) + 8 * math.cos(rad)
        y = int(H * 0.328) + 5 * math.sin(rad)
        left_nostril_arc.append((x, y))
    if left_nostril_arc:
        draw_text_along_path(output, left_nostril_arc, STREAM_TOKENS['jawline'][:8],
                             nostril_font, INK['mid'], vertical_offset=0)

    right_nostril_arc = []
    for angle in range(-55, 55, 10):
        rad = math.radians(angle)
        x = (hcx + 14) + 8 * math.cos(rad)
        y = int(H * 0.328) + 5 * math.sin(rad)
        right_nostril_arc.append((x, y))
    if right_nostril_arc:
        draw_text_along_path(output, right_nostril_arc, STREAM_TOKENS['jawline'][8:16],
                             nostril_font, INK['deep'], vertical_offset=0)

    # LEFT EAR CONTOUR — trailing, lighter
    left_ear = []
    for y_scan in range(115, 272, 8):
        x_ear = int(hcx - 180 + 18 * math.sin((y_scan - 115) / 157.0 * math.pi))
        left_ear.append((x_ear, y_scan))
    if left_ear:
        draw_text_along_path(output, left_ear, STREAM_TOKENS['hair'][:30],
                             contour_font, INK['light'], vertical_offset=-4)

    # RIGHT EAR CONTOUR — leading, richer
    right_ear = []
    for y_scan in range(112, 268, 8):
        x_ear = int(hcx + 178 - 18 * math.sin((y_scan - 112) / 156.0 * math.pi))
        right_ear.append((x_ear, y_scan))
    if right_ear:
        draw_text_along_path(output, right_ear, STREAM_TOKENS['outline'][:30],
                             contour_font, INK['outline'], vertical_offset=-4)

    # LEADING HAND — gesture (right side, visible at lower frame)
    right_hand_path = []
    for y_scan in range(640, 940, 8):
        x_hand = get_right_edge(sil, y_scan)
        if x_hand:
            right_hand_path.append((x_hand - 3, y_scan))
    if right_hand_path:
        draw_text_along_path(output, right_hand_path, STREAM_TOKENS['face'][:60],
                             contour_font, INK['deep'], vertical_offset=-3)

    # TRAILING HAND — left side, sparse, dissolving
    left_hand_path = []
    for y_scan in range(660, 960, 9):
        x_hand = get_left_edge(sil, y_scan)
        if x_hand:
            left_hand_path.append((x_hand + 3, y_scan))
    if left_hand_path:
        draw_text_along_path(output, left_hand_path, STREAM_TOKENS['general'][:50],
                             contour_font, INK['light'], vertical_offset=-3)

except Exception as e:
    print(f"Warning: detailed feature contours failed: {e}")


# ============================================================
# ZONE 3 — NAME INTEGRATION: MILO on the JAWLINE
#
# Spec: "MILO is structurally integrated along the jawline,
# the anatomical site of speaking out, of holding the jaw set,
# of not backing down. The name runs horizontally."
#
# Placement: a single horizontal line across the lower face/jaw,
# large enough to read at a glance but weighted to feel architectural.
# Surrounding the name, in smaller text, the intellectual lineage.
# ============================================================

jaw_y = int(H * 0.395)        # just below the mouth, upper jaw

# Surrounding lineage text — smaller, flanking the name
lineage_font = get_font(11)
lineage_tokens = STREAM_TOKENS['jawline'][:40]
lineage_x = hcx - 200
for token in lineage_tokens[:20]:
    draw_word_rot(output, (lineage_x, jaw_y - 4), token,
                 lineage_font, INK['mid'], angle=random.uniform(-4, 4), jitter=(2, 1))
    lineage_x += len(token) * 6 + 4
    if lineage_x > hcx + 190:
        break

lineage_x2 = hcx - 180
for token in lineage_tokens[20:]:
    draw_word_rot(output, (lineage_x2, jaw_y + 14), token,
                 lineage_font, INK['mid'], angle=random.uniform(-4, 4), jitter=(2, 1))
    lineage_x2 += len(token) * 6 + 4
    if lineage_x2 > hcx + 175:
        break

# The name MILO — architectural, not decorative
name_font = get_font(68)
try:
    bb = name_font.getbbox("MILO")
    name_w = bb[2] - bb[0]
except:
    name_w = 68 * 4
name_x = hcx - name_w // 2

# Render MILO horizontally across the jawline
draw.text((name_x, jaw_y - 2), "MILO", fill=(120, 14, 12), font=name_font)


# ============================================================
# SAVE
# ============================================================

output_rgb = output.convert('RGB')
output_name = "MILO_embodiment_v1.png"
output_rgb.save(output_name, dpi=(300, 300))
print(f"Saved: {output_name} ({W}x{H}px, 300dpi)")
print(f"Total characters placed: {placed}")
print("Open in Illustrator or any image viewer to examine.")
print()
print("MILO iteration notes:")
print("  - LEAN_OFFSET controls how far the head is shifted right (currently:", LEAN_OFFSET, "px)")
print("  - leading_edge_weight() governs the directional density gradient")
print("  - INK dict controls the full press-red palette: deep/rich/mid/warm/light/pale/outline")
print("  - Zone 'jawline' (ny 0.42–0.50) places MILO lineage names + horizontal name bar")
print("  - To increase forward-lean drama: increase LEAN_OFFSET and tighten rotation_range in face zone")
