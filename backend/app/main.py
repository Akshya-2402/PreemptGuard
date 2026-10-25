import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.experiments import router as experiments_router
from app.api.monitoring import router as monitoring_router
from app.api.processes import router as processes_router
from app.api.settings import router as settings_router
from app.core.config import settings_store
from app.core.database import database
from app.workloads.manager import process_manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize()
    database.execute(
        """UPDATE managed_processes SET status='orphaned',ended_at=?
        WHERE status IN ('running','recovering')""",
        (datetime.now(timezone.utc).isoformat(),),
    )
    stored = database.query("SELECT key,value FROM app_settings")
    if stored:
        updates = {}
        for row in stored:
            value: object = row["value"]
            if row["key"] in {
                "prediction_horizon", "max_restart_attempts", "max_operation_retries",
                "max_workload_memory_mb",
            }:
                value = int(value)
            elif row["key"] in {
                "monitoring_interval", "low_risk_threshold", "high_risk_threshold",
                "recheck_interval", "restart_cooldown", "verification_period",
                "minimum_prediction_confidence",
            }:
                value = float(value)
            updates[row["key"]] = value
        settings_store.update(updates)
    loop = asyncio.create_task(process_manager.monitor_loop())
    try:
        yield
    finally:
        loop.cancel()
        try:
            await loop
        except asyncio.CancelledError:
            pass
        process_manager.stop_all()


app = FastAPI(
    title="PreemptGuard",
    description="Local managed-process monitoring and risk-aware recovery research lab.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(processes_router)
app.include_router(monitoring_router)
app.include_router(settings_router)
app.include_router(experiments_router)
app.mount(
    "/assets",
    StaticFiles(
        directory=Path(__file__).resolve().parents[2] / "frontend" / "dist" / "assets",
        check_dir=False,
    ),
    name="dashboard-assets",
)


@app.get("/api/health")
def health():
    return {"status": "ok", "application": "PreemptGuard"}


@app.post("/api/history/reset")
def reset_history(confirm: bool = False):
    if not confirm:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Set confirm=true to delete history.")
    if process_manager.children:
        from fastapi import HTTPException

        raise HTTPException(status_code=409, detail="Stop all managed workloads before clearing history.")
    with database.connect() as connection:
        for table in (
            "measurements", "predictions", "decisions", "recovery_attempts",
        "monitor_overhead", "experiments", "managed_processes",
        ):
            connection.execute(f"DELETE FROM {table}")
    return {"status": "history cleared"}


@app.get("/")
def dashboard_page():
    index = Path(__file__).resolve().parents[2] / "frontend" / "dist" / "index.html"
    if not index.exists():
        return {
            "application": "PreemptGuard API",
            "dashboard": "Start Vite with `npm run dev` in frontend, or build it with `npm run build`.",
            "docs": "/docs",
        }
    return FileResponse(index)
