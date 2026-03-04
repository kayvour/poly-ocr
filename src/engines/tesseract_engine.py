import pytesseract
from PIL import Image
import time
from .base_engine import BaseOCREngine


class TesseractEngine(BaseOCREngine):
    def __init__(self, tesseract_path=None):
        super().__init__("tesseract")

        # 🔥 FORCE Windows path
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        # If installed elsewhere, change path above

    def predict(self, image_path: str, lang: str = "eng"):
        start = time.time()

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)

        end = time.time()

        return text.strip(), None, end - start
        