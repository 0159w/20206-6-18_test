"""Tests for the mine safety inspection API."""

import os
import atexit
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session

from module2_backend.app import app
from module2_backend.core.database import get_session
from module2_backend.models.database import Team, Area


# ── Test Database ────────────────────────────────────────────────

# Use tmp_path via pytest to avoid file leaks
TEST_DB_FILE = tempfile.mktemp(suffix=".db")
TEST_DB_URL = f"sqlite:///{TEST_DB_FILE}"

engine = create_engine(TEST_DB_URL)


def override_get_session():
    with Session(engine) as session:
        yield session


app.dependency_overrides[get_session] = override_get_session

# Register cleanup: delete test DB at process exit (best-effort)
@atexit.register
def _cleanup_test_db():
    import gc
    gc.collect()  # release any remaining references
    try:
        if os.path.exists(TEST_DB_FILE):
            os.unlink(TEST_DB_FILE)
    except (PermissionError, OSError):
        pass  # best-effort cleanup


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

    def test_create_nonexistent_team(self, client, seed_data):
        """Non-existent team_id should return 404."""
        response = client.post(
            "/api/v1/inspections/",
            data={
                "inspection_date": "2026-06-18",
                "team_id": 999,  # does not exist
                "area_id": 1,
                "shift": "白班",
            },
            files={"photo": ("test.jpg", BytesIO(b"data"), "image/jpeg")},
        )
        assert response.status_code == 404

    def test_create_nonexistent_area(self, client, seed_data):
        """Non-existent area_id should return 404."""
        response = client.post(
            "/api/v1/inspections/",
            data={
                "inspection_date": "2026-06-18",
                "team_id": 1,
                "area_id": 999,  # does not exist
                "shift": "白班",
            },
            files={"photo": ("test.jpg", BytesIO(b"data"), "image/jpeg")},
        )
        assert response.status_code == 404


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

    def test_page_size_validation(self, client):
        """page_size > 100 should be rejected by schema validation."""
        response = client.get("/api/v1/inspections/?page_size=200")
        assert response.status_code == 422  # validation error


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


# ── Module-1 Import Test ─────────────────────────────────────────

class TestModule1YOLO:
    def test_module1_config_import(self):
        """module1_yolo config should be importable."""
        from module1_yolo import config
        assert config.CLASS_NAMES == ["fire"]
        assert config.MODEL_NAME == "yolov8n.pt"
        assert config.IMG_SIZE == 640
        assert config.EPOCHS == 50
        assert config.CONF_THRESHOLD == 0.5

    def test_data_converter_import(self):
        """data_converter functions should be importable."""
        from module1_yolo.data_converter import convert_coco_to_yolo, create_data_yaml
        assert callable(convert_coco_to_yolo)
        assert callable(create_data_yaml)
