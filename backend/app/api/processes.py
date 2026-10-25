from fastapi import APIRouter, HTTPException, status

from app.core.database import database
from app.schemas import WorkloadRequest
from app.workloads.manager import process_manager

router = APIRouter(prefix="/api/processes", tags=["processes"])


@router.get("")
def list_processes():
    return process_manager.list_processes()


@router.post("", status_code=status.HTTP_201_CREATED)
def start_process(request: WorkloadRequest):
    return process_manager.start(request)


@router.get("/{process_id}")
def get_process(process_id: str):
    try:
        return process_manager.get_process(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Managed process not found.") from exc


@router.post("/{process_id}/stop")
def stop_process(process_id: str):
    try:
        return process_manager.stop(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Managed process is not active.") from exc


@router.post("/{process_id}/restart")
async def restart_process(process_id: str):
    try:
        return await process_manager.restart(process_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Managed process not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{process_id}/history")
def process_history(process_id: str, limit: int = 300):
    if not 1 <= limit <= 2000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 2000.")
    if not database.query("SELECT id FROM managed_processes WHERE id=?", (process_id,)):
        raise HTTPException(status_code=404, detail="Managed process not found.")
    measurements = database.query(
        "SELECT * FROM measurements WHERE process_id=? ORDER BY timestamp DESC LIMIT ?",
        (process_id, limit),
    )
    measurements.reverse()
    predictions = database.query(
        "SELECT * FROM predictions WHERE process_id=? ORDER BY timestamp DESC LIMIT ?",
        (process_id, limit),
    )
    predictions.reverse()
    for prediction in predictions:
        prediction["features"] = __import__("json").loads(prediction.pop("features_json"))
        prediction["contributors"] = __import__("json").loads(prediction.pop("contributors_json"))
    decisions = database.query(
        "SELECT * FROM decisions WHERE process_id=? ORDER BY timestamp DESC LIMIT ?",
        (process_id, limit),
    )
    recoveries = database.query(
        "SELECT * FROM recovery_attempts WHERE process_id=? ORDER BY timestamp DESC LIMIT ?",
        (process_id, limit),
    )
    return {
        "measurements": list(reversed(measurements)),
        "predictions": list(reversed(predictions)),
        "decisions": list(reversed(decisions)),
        "recoveries": list(reversed(recoveries)),
    }
