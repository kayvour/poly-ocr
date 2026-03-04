import time
from PIL import Image
from .base_engine import BaseOCREngine

class NougatEngine(BaseOCREngine):
    def __init__(self, model_name="facebook/nougat-base"):
        super().__init__("nougat")
        self.model_name = model_name
        self.processor = None
        self.model = None

    def predict(self, image_path: str, lang: str = "eng"):
        if self.processor is None or self.model is None:
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.processor = NougatProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            
        start = time.time()
        
        image = Image.open(image_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            min_length=1,
            max_new_tokens=30,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        )
        
        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        sequence = self.processor.post_process_generation(sequence, fix_markdown=False)

        end = time.time()
        
        return sequence.strip(), None, end - start
