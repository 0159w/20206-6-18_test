"""Business logic layer — encapsulates all DB operations.

No direct DB Session manipulation in API routes.
"""

from datetime import date
from typing import List, Optional, Tuple

from sqlmodel import Session, select, func

from module2_backend.models.database import InspectionRecord


class InspectionService:
    """Service encapsulating inspection record CRUD operations."""

    def __init__(self, session: Session):
        self.session = session

    # ── Create ──────────────────────────────────────────────────

    def create_record(
        self,
        inspection_date: date,
        team_id: int,
        area_id: int,
        shift: str,
        photo_path: str,
        is_safe: bool,
        confidence: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        inspector_ip: Optional[str] = None,
        remark: Optional[str] = None,
    ) -> InspectionRecord:
        """Create a new inspection record."""
        record = InspectionRecord(
            inspection_date=inspection_date,
            team_id=team_id,
            area_id=area_id,
            shift=shift,
            photo_path=photo_path,
            is_safe=is_safe,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            inspector_ip=inspector_ip,
            remark=remark,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    # ── Query ────────────────────────────────────────────────────

    def get_record(self, record_id: int) -> Optional[InspectionRecord]:
        """Get a single record by ID."""
        return self.session.get(InspectionRecord, record_id)

    def list_records(
        self,
        page: int = 1,
        page_size: int = 20,
        team_id: Optional[int] = None,
        area_id: Optional[int] = None,
    ) -> Tuple[List[InspectionRecord], int]:
        """
        Paginated query with optional filters.

        Returns (records_list, total_count).
        """
        query = select(InspectionRecord)

        if team_id is not None:
            query = query.where(InspectionRecord.team_id == team_id)
        if area_id is not None:
            query = query.where(InspectionRecord.area_id == area_id)

        # Total count
        count_query = select(func.count()).select_from(query.subquery())
        total = self.session.exec(count_query).one()

        # Paginated fetch
        query = query.order_by(InspectionRecord.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        records = list(self.session.exec(query).all())

        return records, total

    # ── Delete ──────────────────────────────────────────────────

    def delete_record(self, record_id: int) -> bool:
        """Delete a record by ID. Returns True if deleted."""
        record = self.get_record(record_id)
        if record is None:
            return False
        self.session.delete(record)
        self.session.commit()
        return True
