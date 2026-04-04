"""
main.py
-------
FastAPI application entry-point for AI Vehicle Health Assistant.
Run with:  uvicorn backend.api.main:app --reload
"""

import sys, os
# Ensure the project root is on sys.path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime, timezone

from backend.rag.rag_pipeline import run_rag_pipeline
from backend.db.models import init_db, get_db, VehicleProfile, QueryHistory

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Vehicle Health Assistant",
    description="RAG-powered vehicle diagnostics API",
    version="1.0.0",
)

# Allow all origins for local dev; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise SQLite tables at startup
init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    query:        str = Field(..., min_length=5,  example="My bike engine is making a knocking sound")
    vehicle_type: str = Field(..., pattern="^(bike|car|suv)$", example="bike")


class DiagnoseResponse(BaseModel):
    vehicle_type:            str
    query:                   str
    problem_summary:         str
    possible_causes:         List[str]
    severity:                str          # Low / Moderate / High
    suggested_actions:       List[str]
    estimated_cost_inr:      str
    additional_notes:        Optional[str] = ""
    retrieved_records_count: int


class VehicleProfileCreate(BaseModel):
    owner_name:    str
    vehicle_type:  str = Field(..., pattern="^(bike|car|suv)$")
    vehicle_model: Optional[str] = None
    km_driven:     Optional[int] = None


class VehicleProfileOut(VehicleProfileCreate):
    id:         int
    created_at: datetime

    class Config:
        from_attributes = True


class QueryHistoryOut(BaseModel):
    id:           int
    vehicle_type: str
    query:        str
    severity:     Optional[str]
    summary:      Optional[str]
    created_at:   datetime

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "AI Vehicle Health Assistant API is running 🚗"}


@app.post("/ask", response_model=DiagnoseResponse, tags=["diagnosis"])
def diagnose_vehicle(request: DiagnoseRequest, db: Session = Depends(get_db)):
    """
    Main RAG endpoint.
    Accepts a user query and vehicle type, returns a structured diagnosis.
    """
    try:
        result = run_rag_pipeline(
            query=request.query,
            vehicle_type=request.vehicle_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    # Persist query history
    history = QueryHistory(
        vehicle_type=request.vehicle_type,
        query=request.query,
        severity=result.get("severity"),
        summary=result.get("problem_summary"),
    )
    db.add(history)
    db.commit()

    return result


@app.post("/vehicles", response_model=VehicleProfileOut, tags=["vehicles"])
def create_vehicle_profile(profile: VehicleProfileCreate, db: Session = Depends(get_db)):
    """Store a user's vehicle profile for personalised recommendations."""
    obj = VehicleProfile(**profile.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.get("/vehicles", response_model=List[VehicleProfileOut], tags=["vehicles"])
def list_vehicle_profiles(db: Session = Depends(get_db)):
    return db.query(VehicleProfile).all()


@app.get("/history", response_model=List[QueryHistoryOut], tags=["history"])
def list_query_history(limit: int = 20, db: Session = Depends(get_db)):
    return db.query(QueryHistory).order_by(QueryHistory.created_at.desc()).limit(limit).all()


@app.delete("/history", tags=["history"])
def clear_history(db: Session = Depends(get_db)):
    db.query(QueryHistory).delete()
    db.commit()
    return {"message": "Query history cleared."}
