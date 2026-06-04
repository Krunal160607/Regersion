from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
LONG_FORMAT_COLUMNS = ["Country Name", "Country Code", "Years", "GDP"]
WORLD_BANK_ID_COLUMNS = [
    "Country Name",
    "Country Code",
    "Indicator Name",
    "Indicator Code",
]
PREFERRED_DATASET_FILENAMES = ("FINAL_GDP.csv", "gdp.csv")


def resolve_dataset_path(dataset_dir: Path | None = None) -> Path:
    search_dir = dataset_dir or DATASET_DIR
    csv_files = list(search_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset was found in {search_dir}.")

    filename_map = {path.name.lower(): path for path in csv_files}
    for filename in PREFERRED_DATASET_FILENAMES:
        preferred = filename_map.get(filename.lower())
        if preferred is not None:
            return preferred

    return sorted(
        csv_files,
        key=lambda path: (-path.stat().st_mtime_ns, path.name.lower()),
    )[0]


def build_dataset_signature(dataset_path: Path) -> dict:
    stat = dataset_path.stat()
    return {
        "dataset_name": dataset_path.name,
        "dataset_size": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
    }


def _is_long_format(dataframe: pd.DataFrame) -> bool:
    return set(LONG_FORMAT_COLUMNS).issubset(dataframe.columns)


def _detect_year_columns(columns) -> list[str]:
    return [column for column in columns if str(column).isdigit() and len(str(column)) == 4]


def _is_world_bank_wide_format(dataframe: pd.DataFrame) -> bool:
    return {"Country Name", "Country Code"}.issubset(dataframe.columns) and bool(
        _detect_year_columns(dataframe.columns)
    )


def _read_candidates(dataset_path: Path) -> list[pd.DataFrame]:
    candidates = []
    for skiprows in (0, 4):
        try:
            candidates.append(pd.read_csv(dataset_path, skiprows=skiprows))
        except Exception:
            continue
    return candidates


def normalize_long_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    formatted = dataframe[LONG_FORMAT_COLUMNS].copy()
    formatted = formatted.dropna(subset=LONG_FORMAT_COLUMNS)
    formatted["Country Name"] = formatted["Country Name"].astype(str).str.strip()
    formatted["Country Code"] = formatted["Country Code"].astype(str).str.strip().str.upper()
    formatted["Years"] = pd.to_numeric(formatted["Years"], errors="coerce")
    formatted["GDP"] = pd.to_numeric(formatted["GDP"], errors="coerce")
    formatted = formatted.dropna(subset=["Years", "GDP"])
    formatted = formatted[
        (formatted["Country Name"] != "") & (formatted["Country Code"] != "")
    ].copy()
    formatted["Years"] = formatted["Years"].astype(int)
    formatted["GDP"] = formatted["GDP"].astype(float)
    formatted = formatted.sort_values(["Country Name", "Years"]).reset_index(drop=True)
    return formatted


def world_bank_to_long(dataframe: pd.DataFrame) -> pd.DataFrame:
    year_columns = _detect_year_columns(dataframe.columns)
    if not year_columns:
        raise ValueError("No year columns were found in the raw dataset.")

    formatted = dataframe.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_columns,
        var_name="Years",
        value_name="GDP",
    )
    return normalize_long_dataset(formatted)


def load_training_dataset(dataset_path: Path) -> pd.DataFrame:
    for dataframe in _read_candidates(dataset_path):
        if _is_long_format(dataframe):
            return normalize_long_dataset(dataframe)

    for dataframe in _read_candidates(dataset_path):
        if _is_world_bank_wide_format(dataframe):
            return world_bank_to_long(dataframe)

    raise ValueError(
        "Dataset must either already be formatted with "
        "'Country Name, Country Code, Years, GDP' columns or be a raw World Bank GDP CSV."
    )
