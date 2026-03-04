import argparse
from src.benchmark.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="PolyOCR-Bench: Multilingual OCR Benchmarking Tool"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Language code (eng, hin, tam, etc.)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset folder"
    )

    parser.add_argument(
        "--engines",
        type=str,
        default="tesseract,easyocr",
        help="Comma-separated list of OCR engines to benchmark"
    )

    parser.add_argument(
        "--corrupt",
        action="store_true",
        help="Enable robustness evaluation on corrupted images"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(
        dataset_path=args.dataset,
        lang=args.lang,
        engine_names=args.engines.split(","),
        enable_corruptions=args.corrupt
    )

    runner.run()


if __name__ == "__main__":
    main()
    