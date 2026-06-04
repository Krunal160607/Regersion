from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.model.train_model import ensure_model_artifacts
from app.utils.helper import get_project_status, predict_gdp, try_load_artifacts


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class PredictionRequest(BaseModel):
    country_name: str | None = Field(default=None, min_length=2)
    country_label: int | None = Field(default=None, ge=0)
    year: int = Field(..., ge=1900, le=2100)


def load_app_state(app: FastAPI) -> None:
    model, preprocess = try_load_artifacts()
    app.state.model = model
    app.state.preprocess = preprocess


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_error = None
    try:
        ensure_model_artifacts()
    except Exception as exc:
        app.state.startup_error = str(exc)
    load_app_state(app)
    yield


app = FastAPI(
    title="GDP Decision Tree API",
    description="Notebook-aligned GDP prediction service built with FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    status = get_project_status(startup_error=request.app.state.startup_error)
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "status": status,
            "countries": status["countries"],
            
            "country_to_label": status["country_to_label"],
            "default_year": status["year_max"] if status["year_max"] is not None else "",
        },
    )


@app.get("/api/status")
async def api_status(request: Request):
    return get_project_status(startup_error=request.app.state.startup_error)


@app.get("/api/countries")
async def api_countries(request: Request):
    status = get_project_status(startup_error=request.app.state.startup_error)
    return {
        "countries": status["countries"],
        "country_to_label": status["country_to_label"],
        "year_min": status["year_min"],
        "year_max": status["year_max"],
    }


@app.post("/api/predict")
async def api_predict(payload: PredictionRequest, request: Request):
    model = getattr(request.app.state, "model", None)
    preprocess = getattr(request.app.state, "preprocess", None)

    if model is None or not isinstance(preprocess, dict):
        try:
            ensure_model_artifacts(force=True)
            request.app.state.startup_error = None
        except Exception as exc:
            request.app.state.startup_error = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        load_app_state(request.app)
        model = request.app.state.model
        preprocess = request.app.state.preprocess

    if payload.country_name is None and payload.country_label is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either country_name or country_label.",
        )

    try:
        predicted_gdp = predict_gdp(
            year=payload.year,
            model=model,
            preprocess=preprocess,
            country_name=payload.country_name,
            country_label=payload.country_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.country_name is not None:
        resolved_country_name = payload.country_name
        resolved_country_label = preprocess["country_to_label"][payload.country_name]
    else:
        resolved_country_label = int(payload.country_label)
        resolved_country_name = preprocess["label_to_country"][resolved_country_label]

    return {
        "country_name": resolved_country_name,
        "country_label": resolved_country_label,
        "country_code": preprocess["country_code_lookup"].get(resolved_country_name),
        "year": payload.year,
        "predicted_gdp": predicted_gdp,
    }
