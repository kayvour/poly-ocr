import pytesseract
from PIL import Image
import time
import shutil
from .base_engine import BaseOCREngine


class TesseractEngine(BaseOCREngine):
    def __init__(self, tesseract_path=None):
        super().__init__("tesseract")

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            found = shutil.which("tesseract")
            if found:
                pytesseract.pytesseract.tesseract_cmd = found
            # if not found, pytesseract will use its own default and
            # raise a clear error at prediction time

    def predict(self, image_path: str, lang: str = "eng"):
        start = time.time()
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        end = time.time()
        return text.strip(), None, end - start
