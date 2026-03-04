import os
import json
import pandas as pd
import psutil
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine
from src.engines.paddleocr_engine import PaddleOCREngine
from src.engines.trocr_engine import TrOCREngine
from src.engines.donut_engine import DonutEngine
from src.engines.doctr_engine import DocTREngine
from src.engines.nougat_engine import NougatEngine
from src.dataset.corruptions import apply_corruptions
from src.benchmark.evaluator import evaluate
from src.utils.plotter import generate_plots


class BenchmarkRunner:
    def __init__(self, dataset_path, lang="eng", engine_names=None, enable_corruptions=False):
        self.dataset_path = dataset_path
        self.lang = lang
        self.enable_corruptions = enable_corruptions

        if engine_names is None:
            engine_names = ["tesseract", "easyocr"]

        self.engines = self._init_engines(engine_names)

    def _init_engines(self, names):
        easyocr_lang = "en" if self.lang == "eng" else self.lang
        engine_map = {
            "tesseract": lambda: TesseractEngine(),
            "easyocr": lambda: EasyOCREngine([easyocr_lang]),
            "trocr": lambda: TrOCREngine(),
            "donut": lambda: DonutEngine(),
            "doctr": lambda: DocTREngine(),
            "nougat": lambda: NougatEngine(),
        }
        return [engine_map[name.strip()]() for name in names if name.strip() in engine_map]

    def run(self):
        images_path = os.path.join(self.dataset_path, "images")
        gt_path = os.path.join(self.dataset_path, "ground_truth")

        if not os.path.exists(images_path):
            print("Images folder not found.")
            return

        results = []

        process = psutil.Process(os.getpid())
        
        for file in os.listdir(images_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_file = os.path.join(images_path, file)
            gt_file = os.path.join(
                gt_path,
                os.path.splitext(file)[0] + ".txt"
            )

            if not os.path.exists(gt_file):
                continue

            with open(gt_file, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()

            corrupted_images = {"clean": image_file}
            if self.enable_corruptions:
                corrupted_dir = os.path.join(self.dataset_path, "corrupted_images")
                corrupted_images = apply_corruptions(image_file, corrupted_dir)

            for engine in self.engines:
                for c_name, c_path in corrupted_images.items():
                
                    mem_before = process.memory_info().rss / 1024 / 1024
                    
                    predicted, _, inference_time = engine.predict(
                        c_path,
                        self.lang
                    )
                    
                    mem_after = process.memory_info().rss / 1024 / 1024
                    memory_usage = max(0, mem_after - mem_before)

                    metrics = evaluate(
                        predicted,
                        ground_truth,
                        inference_time,
                        memory_usage
                    )

                    result = {
                        "file": file,
                        "corruption": c_name,
                        "engine": engine.name,
                        "predicted_text": predicted,
                        "ground_truth": ground_truth,
                        **metrics
                    }

                    print(f"[{engine.name} | {c_name}] {file} - CER: {metrics['cer']:.4f}")
                    results.append(result)

        self._save_results(results)
        return results

    def _save_results(self, results):
        if not results:
            print("No results to save.")
            return

        os.makedirs("results/reports", exist_ok=True)

        df = pd.DataFrame(results)

        csv_path = "results/reports/benchmark.csv"
        json_path = "results/reports/benchmark.json"

        df.to_csv(csv_path, index=False)

        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        print("Results saved to results/reports/")

        # ---- Generate plots (NEW) ----
        generate_plots(csv_path)
        