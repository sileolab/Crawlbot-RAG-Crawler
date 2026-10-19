"""Integration and End-to-End API tests for Crawlbot."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.jobs.manager import JobManager
from app.jobs.models import JobInfo, JobState, JobType
from app.main import app, create_app
from app.schemas.documents import ExtractedDocument, Heading, QualityStatus


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestSystemAndGuiRoutes:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "Crawlbot"

    def test_root_gui_serves_html(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Crawlbot" in response.text
        assert 'dir="rtl"' in response.text
        # Verify static asset link paths in HTML
        assert 'href="/static/css/style.css"' in response.text
        assert 'src="/static/js/app.js"' in response.text

    def test_gui_static_assets_served(self, client: TestClient):
        # Verify CSS is served with 200 OK and text/css
        css_res = client.get("/static/css/style.css")
        assert css_res.status_code == 200
        assert "text/css" in css_res.headers.get("content-type", "")
        assert "--color-bg" in css_res.text

        # Verify JS is served with 200 OK
        js_res = client.get("/static/js/app.js")
        assert js_res.status_code == 200
        assert len(js_res.text) > 1000
        # Verify app.js correctly unpacks job_id from ApiResponse data dictionary
        assert "response.data?.job_id" in js_res.text
        assert "res.data?.job_id" in js_res.text


class TestDiscoveryWorkflow:
    def test_start_discovery_success(self, client: TestClient):
        payload = {
            "root_url": "https://example.com",
            "max_depth": 2,
            "max_pages": 50,
            "request_timeout": 15.0,
            "max_concurrency": 2,
            "delay_between_requests": 0.0,
            "use_sitemap": False,
        }
        response = client.post("/api/discovery/start", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data["data"]

        job_id = data["data"]["job_id"]

        # Check status
        status_res = client.get(f"/api/discovery/status/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["job_id"] == job_id
        assert status_data["job_type"] == "discovery"

        # Cancel
        cancel_res = client.post(f"/api/discovery/cancel/{job_id}")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["success"] is True

    def test_start_discovery_invalid_scheme(self, client: TestClient):
        payload = {
            "root_url": "ftp://example.com",
        }
        response = client.post("/api/discovery/start", json=payload)
        assert response.status_code == 422


class TestExtractionAndExportWorkflow:
    def test_full_extraction_and_export_lifecycle(self, client: TestClient):
        # 1. Start extraction
        payload = {
            "urls": ["https://example.com/page1", "https://example.com/page2"],
            "extraction_strategy": "trafilatura",
            "request_timeout": 10.0,
            "max_concurrency": 2,
        }
        start_res = client.post("/api/extraction/start", json=payload)
        assert start_res.status_code == 200
        start_data = start_res.json()
        assert start_data["success"] is True
        job_id = start_data["data"]["job_id"]

        # 2. Check extraction status
        status_res = client.get(f"/api/extraction/status/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["job_id"] == job_id
        assert status_data["job_type"] == "extraction"

        # 3. Check export formats
        formats_res = client.get("/api/export/formats")
        assert formats_res.status_code == 200
        formats_data = formats_res.json()
        assert len(formats_data["data"]["formats"]) >= 4

    def test_export_empty_job_fails(self, client: TestClient):
        # Non-existent job
        res = client.post("/api/export/download", json={"job_id": "nonexistent", "format": "json"})
        assert res.status_code == 404

    def test_retry_failed_endpoints(self, client: TestClient):
        # Create a job directly in manager to test retry
        from app.api.routes_extraction import job_manager
        assert job_manager is not None

        test_job = job_manager.create_extraction_job(
            urls=["https://example.com/failed1"],
            config={"urls": ["https://example.com/failed1"]},
        )
        test_job.documents.append({
            "url": "https://example.com/failed1",
            "title": "Failed",
            "status": "failed",
            "quality": "failed",
        })

        retry_res = client.post(f"/api/extraction/retry/{test_job.job_id}")
        assert retry_res.status_code == 200
        retry_data = retry_res.json()
        assert retry_data["success"] is True
        assert retry_data["data"]["retry_count"] == 1


class TestJobsApi:
    def test_list_and_get_job(self, client: TestClient):
        list_res = client.get("/api/jobs/list")
        assert list_res.status_code == 200
        assert isinstance(list_res.json(), list)

        # 404 for unknown job
        unknown_res = client.get("/api/jobs/unknown-id-12345")
        assert unknown_res.status_code == 404


class TestApiContractsAndErrors:
    """Verifies API response envelope, job_id contract, and error formatting."""

    def test_discovery_contract_has_job_id(self, client: TestClient):
        res = client.post("/api/discovery/start", json={"root_url": "https://example.com", "max_depth": 1, "max_pages": 5})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "job_id" in data["data"]
        assert isinstance(data["data"]["job_id"], str)

    def test_extraction_contract_has_job_id(self, client: TestClient):
        res = client.post("/api/extraction/start", json={"urls": ["https://example.com"]})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "job_id" in data["data"]
        assert isinstance(data["data"]["job_id"], str)

    def test_missing_or_invalid_payload_returns_422(self, client: TestClient):
        res = client.post("/api/discovery/start", json={})
        assert res.status_code == 422
        data = res.json()
        assert "detail" in data

    def test_nonexistent_job_returns_404(self, client: TestClient):
        res = client.get("/api/discovery/status/non-existent-uuid-123")
        assert res.status_code == 404
        assert "detail" in res.json()

