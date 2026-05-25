# Model identifiers
TROCR_MODEL = "microsoft/trocr-base-printed"
DONUT_MODEL = "naver-clova-ix/donut-base"
NOUGAT_MODEL = "facebook/nougat-base"

# Default engines to run if none specified
DEFAULT_ENGINES = ["tesseract", "easyocr", "trocr", "donut", "doctr", "nougat"]

# Output paths
REPORTS_DIR = "results/reports"
PLOTS_DIR = "results/plots"
REPORT_CSV = "results/reports/benchmark.csv"
REPORT_JSON = "results/reports/benchmark.json"
