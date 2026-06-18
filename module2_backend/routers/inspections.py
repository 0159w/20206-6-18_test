"""FastAPI routers for inspection endpoints."""

import uuid
import math
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, Request, Query
from sqlmodel import Session

from module2_backend.core.config import UPLOAD_DIR
from module2_backend.core.database import get_session
from module2_backend.ml.detector import predict_safety
from module2_backend.schemas.inspection import (
    InspectionRecordOut,
    PaginatedResponse,
    InspectionQuery,
)
from module2_backend.services.inspection_service import InspectionService

router = APIRouter(prefix="/api/v1/inspections", tags=["inspections"])


def get_service(session: Session = Depends(get_session)) -> InspectionService:
    """Dependency: create service with DB session."""
    return InspectionService(session)


@router.post("/", response_model=InspectionRecordOut, status_code=201)
async def create_inspection(
    request: Request,
    inspection_date: str = Form(...),
    team_id: int = Form(...),
    area_id: int = Form(...),
    shift: str = Form(...),
    photo: UploadFile = File(...),
    remark: str = Form(default=""),
    service: InspectionService = Depends(get_service),
):
    """
    Submit a photo for safety check.

    - Receives the photo and metadata
    - Saves the photo to disk
    - Runs YOLOv8 model inference
    - Stores the result in the database
    """
    # 1. Validate date
    from datetime import date
    try:
        insp_date = date.fromisoformat(inspection_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # 2. Validate photo format
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if photo.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式: {photo.content_type}。允许的类型: {', '.join(allowed_types)}",
        )

    # 3. Save photo
    ext = Path(photo.filename).suffix if photo.filename else ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / filename
    content = await photo.read()
    filepath.write_bytes(content)

    # 4. Run model inference
    try:
        is_safe, confidence, inference_ms = predict_safety(str(filepath))
    except Exception as e:
        # Clean up on failure
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # 4. Store in DB via Service layer
    #    Service layer validates team_id/area_id foreign keys
    record = service.create_record(
        inspection_date=insp_date,
        team_id=team_id,
        area_id=area_id,
        shift=shift,
        photo_path=str(filepath),
        is_safe=is_safe,
        confidence=confidence,
        inference_time_ms=inference_ms,
        inspector_ip=request.client.host if request.client else None,
        remark=remark or None,
    )
    return record


@router.get("/", response_model=PaginatedResponse)
def list_inspections(
    query: InspectionQuery = Depends(),
    service: InspectionService = Depends(get_service),
):
    """Query inspection records with pagination and optional filters."""
    records, total = service.list_records(
        page=query.page,
        page_size=query.page_size,
        team_id=query.team_id,
        area_id=query.area_id,
    )
    return PaginatedResponse(
        items=[InspectionRecordOut.model_validate(r) for r in records],
        total=total,
        page=query.page,
        page_size=query.page_size,
        total_pages=math.ceil(total / query.page_size) if total > 0 else 0,
    )


@router.get("/{record_id}", response_model=InspectionRecordOut)
def get_inspection(
    record_id: int,
    service: InspectionService = Depends(get_service),
):
    """Get a single inspection record by ID."""
    record = service.get_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@router.delete("/{record_id}", status_code=204)
def delete_inspection(
    record_id: int,
    service: InspectionService = Depends(get_service),
):
    """Delete an inspection record."""
    if not service.delete_record(record_id):
        raise HTTPException(status_code=404, detail="Record not found")
