import sys
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_formatter import (
    build_dataset_signature,
    load_training_dataset,
    resolve_dataset_path,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
PREPROCESS_PATH = Path(__file__).resolve().parent / "preprocess.pkl"


def train_and_persist_model():
    dataset_path = resolve_dataset_path()
    formatted = load_training_dataset(dataset_path)

    encoder = LabelEncoder()
    formatted["Country_Label"] = encoder.fit_transform(formatted["Country Name"])

    inputs = formatted[["Country_Label", "Years"]]
    target = formatted["GDP"]

    model = DecisionTreeRegressor(random_state=42)
    model.fit(inputs, target)

    country_to_label = {
        country_name: int(label)
        for label, country_name in enumerate(encoder.classes_)
    }
    country_code_lookup = (
        formatted[["Country Name", "Country Code"]]
        .drop_duplicates()
        .sort_values("Country Name")
        .set_index("Country Name")["Country Code"]
        .to_dict()
    )

    preprocess = {
        "feature_names": ["Country_Label", "Years"],
        "countries": encoder.classes_.tolist(),
        "country_to_label": country_to_label,
        "label_to_country": {
            int(label): country_name
            for country_name, label in country_to_label.items()
        },
        "country_code_lookup": country_code_lookup,
        "year_min": int(formatted["Years"].min()),
        "year_max": int(formatted["Years"].max()),
        "row_count": int(len(formatted)),
        **build_dataset_signature(dataset_path),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocess, PREPROCESS_PATH)

    return preprocess


def artifacts_are_valid() -> bool:
    if not MODEL_PATH.exists() or not PREPROCESS_PATH.exists():
        return False

    try:
        dataset_path = resolve_dataset_path()
        current_signature = build_dataset_signature(dataset_path)
        preprocess = joblib.load(PREPROCESS_PATH)
        joblib.load(MODEL_PATH)
    except Exception:
        return False

    required_keys = {
        "country_to_label",
        "label_to_country",
        "dataset_name",
        "dataset_size",
        "dataset_mtime_ns",
    }
    if not isinstance(preprocess, dict) or not required_keys.issubset(preprocess):
        return False

    return all(preprocess[key] == value for key, value in current_signature.items())


def ensure_model_artifacts(force: bool = False):
    if force or not artifacts_are_valid():
        return train_and_persist_model()

    return joblib.load(PREPROCESS_PATH)


def main():
    preprocess = train_and_persist_model()
    print("Model training completed successfully.")
    print(f"Dataset used: {preprocess['dataset_name']}")
    print(f"Countries available: {len(preprocess['countries'])}")
    print(f"Year range: {preprocess['year_min']} to {preprocess['year_max']}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Preprocess data saved to: {PREPROCESS_PATH}")


if __name__ == "__main__":
    main()
