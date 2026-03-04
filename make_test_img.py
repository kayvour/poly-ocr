from PIL import Image, ImageDraw, ImageFont
import os

img = Image.new('RGB', (200, 100), color=(255, 255, 255))
d = ImageDraw.Draw(img)

# Try to use a default font, otherwise just draw text
try:
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font = ImageFont.load_default()

d.text((10, 30), "Hello World", fill=(0, 0, 0), font=font)
img.save(r"c:\Users\Arnav\Coding\poly-ocr\tmp_dataset\images\test_img.png")
