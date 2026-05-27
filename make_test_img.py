"""
Utility to generate a test image for benchmarking.
Usage: python make_test_img.py --text "Hello World" --out tmp_dataset/images/test_img.png
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont


def make_test_image(text: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    d.text((10, 30), text, fill=(0, 0, 0), font=font)
    img.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Hello World")
    parser.add_argument("--out", type=str, default="tmp_dataset/images/test_img.png")
    args = parser.parse_args()
    make_test_image(args.text, args.out)
    