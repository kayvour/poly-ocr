import time
from .base_engine import BaseOCREngine

class DocTREngine(BaseOCREngine):
    def __init__(self):
        super().__init__("doctr")
        self.model = None

    def predict(self, image_path: str, lang: str = "eng"):
        if self.model is None:
            # Lazy load
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            
            self.model = ocr_predictor(pretrained=True)
            self.DocumentFile = DocumentFile
            
        start = time.time()
        
        doc = self.DocumentFile.from_images(image_path)
        result = self.model(doc)
        
        # Extract text blocks
        pages = result.pages
        text = ""
        for page in pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text += word.value + " "
                    text += "\n"
        
        end = time.time()
        
        return text.strip(), None, end - start
