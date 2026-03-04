import time
from PIL import Image
from .base_engine import BaseOCREngine
import re

class DonutEngine(BaseOCREngine):
    def __init__(self, model_name="naver-clova-ix/donut-base"):
        super().__init__("donut")
        self.model_name = model_name
        self.processor = None
        self.model = None

    def predict(self, image_path: str, lang: str = "eng"):
        if self.processor is None or self.model is None:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            
        start = time.time()
        
        image = Image.open(image_path).convert("RGB")
        
        task_prompt = "<s_dataset-receipt>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip() # Remove first task start token
        
        text = self.processor.token2json(sequence)
        
        # Donut returns a dictionary for structured tasks, fall back to string extraction if needed
        if isinstance(text, dict):
             # Try a generic extraction of values
             extracted_text = " ".join([str(v) for v in text.values()])
        else:
             extracted_text = str(text)

        end = time.time()
        
        return extracted_text.strip(), None, end - start
