"""Database models for the mine safety inspection system."""

from datetime import date, datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class Team(SQLModel, table=True):
    """施工队"""
    __tablename__ = "teams"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, description="施工队名称")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Area(SQLModel, table=True):
    """采区"""
    __tablename__ = "areas"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, description="采区名称")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InspectionRecord(SQLModel, table=True):
    """安全检查记录"""
    __tablename__ = "inspection_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    inspection_date: date = Field(description="检查日期 (YYYY-MM-DD)")
    team_id: int = Field(foreign_key="teams.id", description="施工队编号")
    area_id: int = Field(foreign_key="areas.id", description="采区编号")
    shift: str = Field(max_length=50, description="班组")
    photo_path: str = Field(max_length=500, description="照片文件路径")
    is_safe: bool = Field(description="模型判断结果: True=安全, False=有违规")
    confidence: Optional[float] = Field(default=None, description="模型推理置信度")

    # 管理字段
    created_at: datetime = Field(default_factory=datetime.utcnow, description="记录创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="记录更新时间")
    inference_time_ms: Optional[float] = Field(default=None, description="模型推理耗时(ms)")
    inspector_ip: Optional[str] = Field(default=None, max_length=50, description="提交者IP")
    remark: Optional[str] = Field(default=None, max_length=500, description="备注")


def create_all_tables():
    """Create all tables (used at startup)."""
    from sqlmodel import SQLModel
    from module2_backend.core.database import engine
    SQLModel.metadata.create_all(engine)
