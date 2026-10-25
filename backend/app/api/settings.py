from fastapi import APIRouter, HTTPException

from app.core.config import DATA_DIR, settings_store
from app.core.database import database
from app.prediction.train_model import load_trained_model, train_and_evaluate
from app.schemas import SettingsUpdate, validate_settings_update

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_settings():
    return settings_store.get().model_dump()


@router.put("")
def update_settings(request: SettingsUpdate):
    try:
        candidate = validate_settings_update(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if candidate.prediction_model == "logistic_regression":
        model = load_trained_model()
        if model is None:
            raise HTTPException(
                status_code=409,
                detail="Train a logistic-regression model before selecting it.",
            )
        if int(model["horizon_s"]) != candidate.prediction_horizon:
            raise HTTPException(
                status_code=409,
                detail="The selected model was trained for a different prediction horizon.",
            )
    values = candidate.model_dump()
    for key, value in values.items():
        database.execute(
            "INSERT INTO app_settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
    return settings_store.update(request.model_dump(exclude_none=True)).model_dump()


@router.get("/model")
def get_prediction_model():
    model_path = DATA_DIR / "risk_model.joblib"
    model = load_trained_model()
    return {
        "active_model": settings_store.get().prediction_model,
        "trained_model_available": model is not None,
        "trained_horizon_s": model["horizon_s"] if model is not None else None,
        "metrics": model.get("evaluation") if model is not None else None,
        "training_report": model.get("training_report") if model is not None else None,
        "model_path": str(model_path) if model_path.exists() else None,
    }


@router.post("/model/train")
def train_prediction_model(
    horizon_s: int = 30,
    activate: bool = False,
    classification_threshold: float | None = None,
):
    current_settings = settings_store.get()
    if (
        current_settings.prediction_model == "logistic_regression"
        and current_settings.prediction_horizon != horizon_s
        and not activate
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Training at a different horizon would invalidate the active model. "
                "Select the rule-based estimator first or activate the new model."
            ),
        )
    try:
        result = train_and_evaluate(
            horizon_s=horizon_s,
            classification_threshold=(
                classification_threshold
                if classification_threshold is not None
                else current_settings.high_risk_threshold
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if activate:
        updated = settings_store.update({
            "prediction_model": "logistic_regression",
            "prediction_horizon": horizon_s,
        })
        database.execute(
            """INSERT INTO app_settings(key,value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            ("prediction_model", updated.prediction_model),
        )
        database.execute(
            """INSERT INTO app_settings(key,value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            ("prediction_horizon", str(updated.prediction_horizon)),
        )
    return result
