from pathlib import Path

import joblib
import pandas as pd

from app.utils.data_formatter import load_training_dataset, resolve_dataset_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "app" / "model" / "model.pkl"
PREPROCESS_PATH = PROJECT_ROOT / "app" / "model" / "preprocess.pkl"
NOTEBOOK_PATH = PROJECT_ROOT / "2_GDP_Decison_Tree.ipynb"


def try_load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        preprocess = joblib.load(PREPROCESS_PATH)
        return model, preprocess
    except Exception:
        return None, None


def _read_dataset_details():
    try:
        dataset_path = resolve_dataset_path()
    except FileNotFoundError as exc:
        return {
            "dataset_name": None,
            "dataset_rows": 0,
            "countries": [],
            "year_min": None,
            "year_max": None,
            "dataset_error": str(exc),
        }

    try:
        dataset = load_training_dataset(dataset_path)
    except Exception as exc:
        return {
            "dataset_name": dataset_path.name,
            "dataset_rows": 0,
            "countries": [],
            "year_min": None,
            "year_max": None,
            "dataset_error": str(exc),
        }

    countries = sorted(dataset["Country Name"].dropna().unique().tolist())
    return {
        "dataset_name": dataset_path.name,
        "dataset_rows": int(len(dataset)),
        "countries": countries,
        "year_min": int(dataset["Years"].min()) if not dataset.empty else None,
        "year_max": int(dataset["Years"].max()) if not dataset.empty else None,
        "dataset_error": None,
    }


def resolve_prediction_target(
    preprocess: dict,
    country_name: str | None = None,
    country_label: int | None = None,
) -> tuple[str, int]:
    country_to_label = preprocess.get("country_to_label", {})
    label_to_country = preprocess.get("label_to_country", {})

    if country_name is None and country_label is None:
        raise ValueError("Provide either country_name or country_label.")

    if country_name is not None:
        if country_name not in country_to_label:
            available = ", ".join(preprocess.get("countries", [])[:5])
            raise ValueError(
                f"Unknown country '{country_name}'. Try one of: {available}"
            )
        return country_name, int(country_to_label[country_name])

    if country_label not in label_to_country:
        sample_labels = ", ".join(str(label) for label in sorted(label_to_country)[:5])
        raise ValueError(
            f"Unknown country_label '{country_label}'. Try one of: {sample_labels}"
        )

    return str(label_to_country[country_label]), int(country_label)


def build_prediction_frame(
    year: int,
    preprocess: dict,
    country_name: str | None = None,
    country_label: int | None = None,
) -> pd.DataFrame:
    resolved_country_name, resolved_country_label = resolve_prediction_target(
        preprocess=preprocess,
        country_name=country_name,
        country_label=country_label,
    )

    if resolved_country_name not in preprocess.get("country_to_label", {}):
        available = ", ".join(preprocess.get("countries", [])[:5])
        raise ValueError(
            f"Unknown country '{resolved_country_name}'. Try one of: {available}"
        )

    year_min = preprocess.get("year_min")
    if year_min is not None and year < year_min:
        raise ValueError(f"Year must be at least {year_min}.")

    return pd.DataFrame(
        [
            {
                "Country_Label": resolved_country_label,
                "Years": int(year),
            }
        ]
    )


def predict_gdp(
    year: int,
    model,
    preprocess: dict,
    country_name: str | None = None,
    country_label: int | None = None,
) -> float:
    features = build_prediction_frame(
        year=year,
        preprocess=preprocess,
        country_name=country_name,
        country_label=country_label,
    )
    prediction = float(model.predict(features)[0])
    return round(prediction, 2)


def get_project_status(startup_error: str | None = None):
    model, preprocess = try_load_artifacts()
    trained = model is not None and isinstance(preprocess, dict)
    dataset_details = _read_dataset_details()

    if startup_error:
        message = startup_error
    elif dataset_details["dataset_error"]:
        message = f"Dataset issue: {dataset_details['dataset_error']}"
    elif trained:
        message = "Artifacts loaded successfully. You can test predictions below."
    elif MODEL_PATH.exists() and PREPROCESS_PATH.exists():
        message = (
            "Model files exist but need to be regenerated. Run the training script "
            "to rebuild them from the formatted dataset."
        )
    else:
        message = "No model artifacts found yet. Run the training script first."

    return {
        "dataset_exists": dataset_details["dataset_name"] is not None,
        "dataset_name": preprocess.get("dataset_name", dataset_details["dataset_name"])
        if isinstance(preprocess, dict)
        else dataset_details["dataset_name"],
        "model_exists": MODEL_PATH.exists(),
        "preprocess_exists": PREPROCESS_PATH.exists(),
        "trained": trained,
        "notebook_exists": NOTEBOOK_PATH.exists(),
        "message": message,
        "dataset_rows": dataset_details["dataset_rows"],
        "countries": preprocess.get("countries", dataset_details["countries"])
        if isinstance(preprocess, dict)
        else dataset_details["countries"],
        "country_to_label": preprocess.get("country_to_label", {})
        if isinstance(preprocess, dict)
        else {},
        "year_min": preprocess.get("year_min", dataset_details["year_min"])
        if isinstance(preprocess, dict)
        else dataset_details["year_min"],
        "year_max": preprocess.get("year_max", dataset_details["year_max"])
        if isinstance(preprocess, dict)
        else dataset_details["year_max"],
    }
