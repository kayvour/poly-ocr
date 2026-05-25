import argparse
from src.benchmark.runner import BenchmarkRunner
from src.config.settings import DEFAULT_ENGINES


def main():
    parser = argparse.ArgumentParser(
        description="PolyOCR-Bench: Multilingual OCR Benchmarking Tool"
    )
    parser.add_argument("--lang", type=str, default="eng",
                        help="Language code (eng, hin, chi_sim, ara)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset folder")
    parser.add_argument("--engines", type=str, default=",".join(DEFAULT_ENGINES),
                        help="Comma-separated engines: tesseract, easyocr, trocr, donut, doctr, nougat, paddleocr")
    parser.add_argument("--corrupt", action="store_true",
                        help="Enable robustness evaluation on corrupted images")
    parser.add_argument("--tesseract-path", type=str, default=None,
                        help="Path to tesseract executable (Windows users)")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        dataset_path=args.dataset,
        lang=args.lang,
        engine_names=args.engines.split(","),
        enable_corruptions=args.corrupt,
        tesseract_path=args.tesseract_path,
    )
    runner.run()


if __name__ == "__main__":
    main()
    