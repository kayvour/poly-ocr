import easyocr
import time
from .base_engine import BaseOCREngine


class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages=None):
        super().__init__("easyocr")

        if languages is None:
            languages = ['en']
            
        self.languages = languages
        self.reader = None

    def predict(self, image_path: str, lang: str = "eng"):
        if self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=False)

        start = time.time()

        results = self.reader.readtext(image_path)
        text = " ".join([res[1] for res in results])

        end = time.time()

        return text.strip(), None, end - start
        