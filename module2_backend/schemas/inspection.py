"""Pydantic schemas for request/response validation."""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────

class InspectionCreate(BaseModel):
    """创建安全检查记录的请求体（不含照片 — 照片通过 UploadFile 上传）"""
    inspection_date: date = Field(..., description="检查日期 (YYYY-MM-DD)")
    team_id: int = Field(..., description="施工队编号")
    area_id: int = Field(..., description="采区编号")
    shift: str = Field(..., max_length=50, description="班组")
    remark: Optional[str] = Field(default=None, max_length=500, description="备注")


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
