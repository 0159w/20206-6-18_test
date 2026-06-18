"""Pydantic schemas for request/response validation."""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────


class InspectionQuery(BaseModel):
    """分页查询参数"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页条数")
    team_id: Optional[int] = Field(default=None, description="按施工队筛选")
    area_id: Optional[int] = Field(default=None, description="按采区筛选")


# ── Response Schemas ─────────────────────────────────────────────

class InspectionRecordOut(BaseModel):
    """安全检查记录响应"""
    id: int
    inspection_date: date
    team_id: int
    area_id: int
    shift: str
    photo_path: str
    is_safe: bool
    confidence: Optional[float] = None
    inference_time_ms: Optional[float] = None
    created_at: datetime
    remark: Optional[str] = None

    model_config = {"from_attributes": True}


class PaginatedResponse(BaseModel):
    """分页响应"""
    items: List[InspectionRecordOut]
    total: int
    page: int
    page_size: int
    total_pages: int
