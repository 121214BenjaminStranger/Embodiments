"""Test font loading and rendering."""
from PIL import Image, ImageDraw, ImageFont

W, H = 200, 100
img = Image.new('RGB', (W, H), (252, 252, 248))
draw = ImageDraw.Draw(img)

print("Testing font loading...")
for path in [
    'C:\\Windows\\Fonts\\arial.ttf',
    'C:\\Windows\\Fonts\\calibri.ttf',
    'C:\\Windows\\Fonts\\seguiemj.ttf',
]:
    try:
        f = ImageFont.truetype(path, 20)
        print(f"✓ Loaded: {path}")
        draw.text((10, 10), f"Font OK: {path.split(chr(92))[-1]}", fill=(0, 0, 0), font=f)
    except Exception as e:
        print(f"✗ Failed {path}: {e}")

try:
    f_default = ImageFont.load_default()
    print("✓ Loaded default font")
    draw.text((10, 50), "Default font test", fill=(0, 0, 0), font=f_default)
except Exception as e:
    print(f"✗ Default font failed: {e}")

img.save('HALLEY_fonttest.png')
print("Saved: HALLEY_fonttest.png")
