"""Business logic layer — encapsulates all DB operations.

No direct DB Session manipulation in API routes.
"""

from datetime import date, datetime, timezone
from typing import List, Optional, Tuple

from fastapi import HTTPException
from sqlmodel import Session, select, func

from module2_backend.models.database import InspectionRecord, Team, Area


class InspectionService:
    """Service encapsulating inspection record CRUD operations."""

    def __init__(self, session: Session):
        self.session = session

    # ── Validation ───────────────────────────────────────────────

    def validate_team_exists(self, team_id: int) -> None:
        """Raise 404 if team_id does not exist."""
        team = self.session.get(Team, team_id)
        if team is None:
            raise HTTPException(status_code=404, detail=f"施工队 (team_id={team_id}) 不存在")

    def validate_area_exists(self, area_id: int) -> None:
        """Raise 404 if area_id does not exist."""
        area = self.session.get(Area, area_id)
        if area is None:
            raise HTTPException(status_code=404, detail=f"采区 (area_id={area_id}) 不存在")

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
        # Validate foreign keys
        self.validate_team_exists(team_id)
        self.validate_area_exists(area_id)

        now = datetime.now(timezone.utc)
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
            created_at=now,
            updated_at=now,
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
        base_conditions = []
        if team_id is not None:
            base_conditions.append(InspectionRecord.team_id == team_id)
        if area_id is not None:
            base_conditions.append(InspectionRecord.area_id == area_id)

        # Total count — direct count with filters, no subquery nesting
        count_query = select(func.count()).select_from(InspectionRecord)
        if base_conditions:
            count_query = count_query.where(*base_conditions)
        total = self.session.exec(count_query).one()

        # Paginated fetch
        query = select(InspectionRecord)
        if base_conditions:
            query = query.where(*base_conditions)
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
