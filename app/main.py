from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import zipfile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.ml.model_service import SepsisModelService
from app.ml.gemini_service import GeminiClinicalAssistant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


app = FastAPI(
    title="DPCT Sepsis Detection — Clinical Inference API",
    description=(
        "Serves patient-level sepsis risk predictions using the Dual-Path Clinical Transformer (DPCT). "
        "Training requires a GPU — run sepsis_transformer_paper.ipynb on Kaggle, then copy the "
        "saved dpct_model_weights.pth into the artifacts/ directory."
    ),
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
model_service = SepsisModelService(artifact_dir=ARTIFACT_DIR)
gemini_assistant = GeminiClinicalAssistant()

SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".psv"}
SUPPORTED_BUNDLE_EXTENSIONS = {".zip"}
MAX_MULTIPART_FILES = 50000
MAX_MULTIPART_FIELDS = 50000


class Measurement(BaseModel):
    hr: float = Field(..., ge=20, le=260)
    o2sat: float = Field(..., ge=40, le=100)
    temp: float = Field(..., ge=30, le=43)
    sbp: float = Field(..., ge=40, le=260)
    map: float = Field(..., ge=20, le=200)
    resp: float = Field(..., ge=4, le=80)
    age: float = Field(..., ge=0, le=120)
    gender: Union[str, int, float] = Field(default=0)
    dbp: Optional[float] = Field(default=None, ge=20, le=180)
    wbc: Optional[float] = Field(default=None, ge=0.1, le=200)
    iculos: float = Field(default=6, ge=0, le=1000)


class ManualPredictionRequest(BaseModel):
    measurements: List[Measurement]


class ExplainRequest(BaseModel):
    query: str
    patient_context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None


def _has_patient_identifier(columns: List[str]) -> bool:
    return any("patient" in str(column).lower() and "id" in str(column).lower() for column in columns)


def _has_time_identifier(columns: List[str]) -> bool:
    markers = ("hour", "iculos", "icu", "time")
    return any(any(marker in str(column).lower() for marker in markers) for column in columns)


def _read_table_from_bytes(file_bytes: bytes, suffix: str, source_tag: str) -> pd.DataFrame:
    if suffix == ".csv":
        dataframe = pd.read_csv(BytesIO(file_bytes))
    elif suffix == ".psv":
        dataframe = pd.read_csv(BytesIO(file_bytes), sep="|")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")

    if dataframe.empty:
        raise HTTPException(status_code=400, detail="Uploaded data file is empty.")

    columns = [str(column) for column in dataframe.columns]

    if not _has_patient_identifier(columns):
        dataframe["Patient_ID"] = source_tag

    if not _has_time_identifier(columns):
        dataframe["Hour"] = pd.RangeIndex(start=0, stop=len(dataframe), step=1)

    return dataframe


def _read_zip_bundle(file_bytes: bytes) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            for index, info in enumerate(archive.infolist()):
                if info.is_dir():
                    continue

                inner_name = str(info.filename).replace("\\", "/")
                suffix = Path(inner_name).suffix.lower()
                if suffix not in SUPPORTED_TABULAR_EXTENSIONS:
                    continue

                source_tag = (Path(inner_name).stem or f"patient_{index}").replace(" ", "_")
                inner_bytes = archive.read(info)
                frames.append(_read_table_from_bytes(inner_bytes, suffix=suffix, source_tag=source_tag))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {exc}") from exc

    if not frames:
        raise HTTPException(status_code=400, detail="ZIP does not contain any supported .csv or .psv files.")

    return pd.concat(frames, ignore_index=True, sort=False)


def _read_uploaded_dataset(upload: UploadFile) -> pd.DataFrame:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Please upload a data file.")

    suffix = Path(upload.filename).suffix.lower()

    if suffix not in SUPPORTED_TABULAR_EXTENSIONS.union(SUPPORTED_BUNDLE_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Supported file types: .csv, .psv, .zip")

    file_bytes = upload.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        if suffix in SUPPORTED_TABULAR_EXTENSIONS:
            source_tag = (Path(upload.filename).stem or "uploaded_patient").replace(" ", "_")
            return _read_table_from_bytes(file_bytes, suffix=suffix, source_tag=source_tag)
        return _read_zip_bundle(file_bytes)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive parsing path
        raise HTTPException(status_code=400, detail=f"Unable to parse upload: {exc}") from exc


def _read_uploaded_files(datasets: List[UploadFile]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for dataset in datasets:
        if dataset is None:
            continue
        if not hasattr(dataset, "filename") or not hasattr(dataset, "file"):
            continue
        if not dataset.filename:
            continue
        frames.append(_read_uploaded_dataset(dataset))

    if not frames:
        raise HTTPException(status_code=400, detail="No usable files were uploaded.")

    return pd.concat(frames, ignore_index=True, sort=False)


async def _extract_uploads_from_form(request: Request, field_name: str = "datasets") -> List[UploadFile]:
    try:
        form = await request.form(max_files=MAX_MULTIPART_FILES, max_fields=MAX_MULTIPART_FIELDS)
    except TypeError:
        form = await request.form()

    def _is_upload_like(item: Any) -> bool:
        return hasattr(item, "filename") and (hasattr(item, "file") or hasattr(item, "read"))

    uploads = [item for item in form.getlist(field_name) if _is_upload_like(item)]

    if not uploads:
        uploads = [item for _, item in form.multi_items() if _is_upload_like(item)]

    if not uploads:
        raise HTTPException(status_code=400, detail=f"No usable files were uploaded in field '{field_name}'.")

    return uploads


def _predict_and_respond(dataframe: pd.DataFrame) -> Dict[str, Any]:
    try:
        predictions = model_service.predict_from_dataframe(dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive service path
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "count": len(predictions),
        "threshold": model_service.threshold,
        "predictions": predictions,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": model_service.status(),
        },
    )


@app.get("/api/status")
async def status() -> Dict[str, Any]:
    return model_service.status()


@app.post("/api/train")
async def train_model_not_supported() -> Dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail=(
            "In-browser training is not supported for the DPCT model — it requires a GPU and ~30 min. "
            "Please train using sepsis_transformer_paper.ipynb on Kaggle, then save and download "
            "dpct_model_weights.pth and place it in the artifacts/ directory."
        ),
    )


@app.post("/api/predict/csv")
async def predict_from_csv(dataset: UploadFile = File(...)) -> Dict[str, Any]:
    dataframe = _read_uploaded_dataset(dataset)
    return _predict_and_respond(dataframe)


@app.post("/api/predict/files")
async def predict_from_files(request: Request) -> Dict[str, Any]:
    datasets = await _extract_uploads_from_form(request, field_name="datasets")
    dataframe = _read_uploaded_files(datasets)
    return _predict_and_respond(dataframe)

@app.post("/api/predict/manual")
async def predict_from_manual(payload: ManualPredictionRequest) -> Dict[str, Any]:
    body = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    try:
        prediction = model_service.predict_manual(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive service path
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "prediction": prediction,
    }


@app.post("/api/explain")
async def explain_prediction(payload: ExplainRequest) -> Dict[str, Any]:
    response_text = gemini_assistant.explain_prediction(
        query=payload.query,
        patient_context=payload.patient_context,
        history=payload.history
    )
    return {"explanation": response_text}
