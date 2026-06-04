# GDP Decision Tree Project

This project turns your GDP Colab workflow into a FastAPI application.

The notebook logic has been moved into the training pipeline:

- raw World Bank GDP data can be converted into long format
- the formatted dataset uses `Country Name`, `Country Code`, `Years`, and `GDP`
- `Country Name` is label-encoded into `Country_Label`
- a `DecisionTreeRegressor` is trained with `Country_Label` and `Years`
- predictions are served through FastAPI and shown on a simple web page

## Project Structure

```text
gdp_decision_tree_project/
|-- app/
|   |-- main.py
|   |-- model/
|   |   |-- train_model.py
|   |   |-- model.pkl
|   |   `-- preprocess.pkl
|   |-- templates/
|   |   `-- index.html
|   |-- static/
|   |   |-- style.css
|   |   `-- script.js
|   `-- utils/
|       |-- data_formatter.py
|       `-- helper.py
|-- dataset/
|   `-- FINAL_GDP.csv
|-- 2_GDP_Decison_Tree.ipynb
|-- requirements.txt
|-- README.md
`-- run.py
```

## Dataset Format

The app expects either:

1. A formatted dataset with these columns:

   ```text
   Country Name, Country Code, Years, GDP
   ```

2. Or a raw World Bank GDP per capita CSV like the one used in your Colab notebook.

The app auto-detects the CSV inside `dataset/` and currently prefers `FINAL_GDP.csv` when it exists.
If you replace the dataset file later, the training step will rebuild the model from the new CSV.

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:

   ```bash
   python app/model/train_model.py
   ```

3. Run the app:

   ```bash
   python run.py
   ```

4. Open:

   ```text
   http://127.0.0.1:8000
   ```

## API Endpoints

- `GET /` renders the web UI
- `GET /api/status` returns dataset and model status
- `GET /api/countries` returns available countries and year range
- `POST /api/predict` predicts GDP for a selected country name or a notebook-style country label

Example prediction body:

```json
{
  "country_name": "India",
  "year": 2023
}
```

Colab-style example:

```json
{
  "country_label": 109,
  "year": 2023
}
```

You can also generate a direct notebook-style prediction from the terminal:

```bash
python app/model/predict_from_colab.py --country-label 109 --year 2023
```
