import time
from PIL import Image
from .base_engine import BaseOCREngine

class TrOCREngine(BaseOCREngine):
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        super().__init__("trocr")
        self.model_name = model_name
        self.processor = None
        self.model = None

    def predict(self, image_path: str, lang: str = "eng"):
        if self.processor is None or self.model is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            
        start = time.time()
        
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        end = time.time()
        
        return text.strip(), None, end - start
