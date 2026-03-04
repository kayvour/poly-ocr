import time
from .base_engine import BaseOCREngine

class PaddleOCREngine(BaseOCREngine):
    def __init__(self, lang="en"):
        super().__init__("paddleocr")
        self.lang = lang
        self.ocr = None

    def predict(self, image_path: str, lang: str = "eng"):
        if self.ocr is None:
            # Lazy load
            from paddleocr import PaddleOCR
            
            # Map standard 'eng' to paddleocr's 'en'
            paddle_lang = "en" if lang == "eng" else lang
            # Disable unneeded logging
            self.ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
            
        start = time.time()
        result = self.ocr.ocr(image_path, cls=True)
        end = time.time()
        
        text = ""
        if result and result[0]:
            # result[0] is a list of lines, each line is [box, (text, confidence)]
            text = " ".join([line[1][0] for line in result[0]])
            
        return text.strip(), None, end - start
