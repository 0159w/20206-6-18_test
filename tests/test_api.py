"""Tests for the mine safety inspection API."""

import os
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session

from module2_backend.app import app
from module2_backend.core.database import get_session
from module2_backend.models.database import InspectionRecord, Team, Area


# ── Test Database ────────────────────────────────────────────────

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
TEST_DB_URL = f"sqlite:///{TEST_DB.name}"

engine = create_engine(TEST_DB_URL)


def override_get_session():
    with Session(engine) as session:
        yield session


app.dependency_overrides[get_session] = override_get_session


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seed_data():
    """Insert test Team and Area records."""
    with Session(engine) as session:
        session.add(Team(id=1, name="施工一队"))
        session.add(Team(id=2, name="施工二队"))
        session.add(Area(id=1, name="采区A"))
        session.add(Area(id=2, name="采区B"))
        session.commit()


# ── Tests ────────────────────────────────────────────────────────

class TestCreateInspection:
    def test_create_success(self, client, seed_data):
        """POST /api/v1/inspections/ should create a record."""
        # Create a dummy image file
        fake_image = BytesIO(b"fake-image-data")
        fake_image.name = "test.jpg"

        response = client.post(
            "/api/v1/inspections/",
            data={
                "inspection_date": "2026-06-18",
                "team_id": 1,
                "area_id": 1,
                "shift": "白班",
                "remark": "例行检查",
            },
            files={"photo": ("test.jpg", fake_image, "image/jpeg")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["team_id"] == 1
        assert data["area_id"] == 1
        assert data["shift"] == "白班"
        assert data["is_safe"] is True  # 模拟结果
        assert data["confidence"] is not None
        assert "id" in data

    def test_create_invalid_date(self, client):
        """Invalid date should return 400."""
        response = client.post(
            "/api/v1/inspections/",
            data={
                "inspection_date": "invalid-date",
                "team_id": 1,
                "area_id": 1,
                "shift": "白班",
            },
            files={"photo": ("test.jpg", BytesIO(b"data"), "image/jpeg")},
        )
        assert response.status_code == 400


class TestListInspections:
    def test_list_empty(self, client):
        """GET /api/v1/inspections/ should return empty list."""
        response = client.get("/api/v1/inspections/")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    def test_list_with_data(self, client, seed_data):
        """Should return paginated results."""
        # Create a record first
        fake_image = BytesIO(b"fake-image-data")
        fake_image.name = "test.jpg"
        client.post(
            "/api/v1/inspections/",
            data={"inspection_date": "2026-06-18", "team_id": 1, "area_id": 1, "shift": "白班"},
            files={"photo": ("test.jpg", fake_image, "image/jpeg")},
        )

        response = client.get("/api/v1/inspections/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1

    def test_filter_by_team(self, client, seed_data):
        """Should filter by team_id."""
        fake_image = BytesIO(b"data")
        fake_image.name = "t.jpg"
        client.post(
            "/api/v1/inspections/",
            data={"inspection_date": "2026-06-18", "team_id": 1, "area_id": 1, "shift": "白班"},
            files={"photo": ("t.jpg", fake_image, "image/jpeg")},
        )
        client.post(
            "/api/v1/inspections/",
            data={"inspection_date": "2026-06-18", "team_id": 2, "area_id": 2, "shift": "夜班"},
            files={"photo": ("t.jpg", fake_image, "image/jpeg")},
        )

        response = client.get("/api/v1/inspections/?team_id=1")
        assert response.status_code == 200
        assert response.json()["total"] == 1


class TestGetInspection:
    def test_get_by_id(self, client, seed_data):
        """GET /api/v1/inspections/{id} should return the record."""
        fake_image = BytesIO(b"data")
        fake_image.name = "t.jpg"
        create_resp = client.post(
            "/api/v1/inspections/",
            data={"inspection_date": "2026-06-18", "team_id": 1, "area_id": 1, "shift": "白班"},
            files={"photo": ("t.jpg", fake_image, "image/jpeg")},
        )
        record_id = create_resp.json()["id"]

        response = client.get(f"/api/v1/inspections/{record_id}")
        assert response.status_code == 200
        assert response.json()["id"] == record_id

    def test_get_not_found(self, client):
        """Non-existent ID should return 404."""
        response = client.get("/api/v1/inspections/9999")
        assert response.status_code == 404


class TestDeleteInspection:
    def test_delete_success(self, client, seed_data):
        """DELETE /api/v1/inspections/{id} should delete the record."""
        fake_image = BytesIO(b"data")
        fake_image.name = "t.jpg"
        create_resp = client.post(
            "/api/v1/inspections/",
            data={"inspection_date": "2026-06-18", "team_id": 1, "area_id": 1, "shift": "白班"},
            files={"photo": ("t.jpg", fake_image, "image/jpeg")},
        )
        record_id = create_resp.json()["id"]

        response = client.delete(f"/api/v1/inspections/{record_id}")
        assert response.status_code == 204

        # Verify deleted
        get_resp = client.get(f"/api/v1/inspections/{record_id}")
        assert get_resp.status_code == 404

    def test_delete_not_found(self, client):
        """DELETE on non-existent ID should return 404."""
        response = client.delete("/api/v1/inspections/9999")
        assert response.status_code == 404
