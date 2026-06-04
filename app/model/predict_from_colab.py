import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model.train_model import ensure_model_artifacts
from app.utils.helper import predict_gdp, resolve_prediction_target, try_load_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a GDP prediction using the Colab notebook workflow."
    )
    parser.add_argument("--country", dest="country_name")
    parser.add_argument("--country-label", dest="country_label", type=int)
    parser.add_argument("--year", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.country_name is None and args.country_label is None:
        raise SystemExit("Provide either --country or --country-label.")

    ensure_model_artifacts()
    model, preprocess = try_load_artifacts()
    if model is None or not isinstance(preprocess, dict):
        raise SystemExit("Model artifacts could not be loaded.")

    country_name, country_label = resolve_prediction_target(
        preprocess=preprocess,
        country_name=args.country_name,
        country_label=args.country_label,
    )
    predicted_gdp = predict_gdp(
        year=args.year,
        model=model,
        preprocess=preprocess,
        country_name=country_name,
    )

    print(f"Dataset: {preprocess['dataset_name']}")
    print(f"Country: {country_name}")
    print(f"Country_Label: {country_label}")
    print(f"Year: {args.year}")
    print(f"Predicted GDP: {predicted_gdp}")


if __name__ == "__main__":
    main()
