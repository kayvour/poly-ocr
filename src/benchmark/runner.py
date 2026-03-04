import os
import json
import pandas as pd
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine
from src.benchmark.evaluator import evaluate
from src.utils.plotter import generate_plots


class BenchmarkRunner:
    def __init__(self, dataset_path, lang="eng"):
        self.dataset_path = dataset_path
        self.lang = lang

        self.engines = [
            TesseractEngine(),
            EasyOCREngine(['en'])
        ]

    def run(self):
        images_path = os.path.join(self.dataset_path, "images")
        gt_path = os.path.join(self.dataset_path, "ground_truth")

        if not os.path.exists(images_path):
            print("Images folder not found.")
            return

        results = []

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

            for engine in self.engines:
                predicted, _, inference_time = engine.recognize(
                    image_file,
                    self.lang
                )

                metrics = evaluate(
                    predicted,
                    ground_truth,
                    inference_time
                )

                result = {
                    "file": file,
                    "engine": engine.name,
                    "predicted_text": predicted,
                    "ground_truth": ground_truth,
                    **metrics
                }

                print(result)
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
        