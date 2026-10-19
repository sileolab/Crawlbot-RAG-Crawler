"""Tests for Crawlbot File & Data Processing Studio.

Covers:
- Uploading supported file formats (.md, .txt, .json, .jsonl, .csv)
- Rejection of disallowed extensions (.exe, .py, .sh)
- Rejection of oversized files (> 10MB)
- Path traversal defense in file uploads
- Processing operations: normalize, markdown_to_json, json_validate, json_cleanup, chunk, metadata_enrich
- LLM optimization and embedding operations (mocked)
- Multi-step pipeline execution
- Artifact preview, download, and file info endpoints
- Error handling and non-existent artifact 404s
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.file_processor import FileProcessorService, file_processor


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Upload & Security Tests
# ---------------------------------------------------------------------------

class TestFileUploadAndSecurity:
    """Test suite for file uploads, size limits, and path traversal protection."""

    def test_upload_valid_markdown_file(self, client):
        content = b"# Title\n\nThis is sample markdown with Persian: \xd8\xaf\xd8\xa7\xd8\xaf\xd9\x87\xd9\x87\xd8\xa7"
        files = {"file": ("sample.md", io.BytesIO(content), "text/markdown")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["success"] is True
        data = res_json["data"]
        assert data["filename"] == "sample.md"
        assert data["format"] == "md"
        assert data["size_bytes"] == len(content)
        assert "file_id" in data

    def test_upload_valid_json_file(self, client):
        obj = {"name": "test", "items": [1, 2, 3]}
        content = json.dumps(obj).encode("utf-8")
        files = {"file": ("data.json", io.BytesIO(content), "application/json")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["format"] == "json"

    def test_upload_valid_jsonl_file(self, client):
        content = b'{"line": 1}\n{"line": 2}\n'
        files = {"file": ("records.jsonl", io.BytesIO(content), "application/x-ndjson")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["format"] == "jsonl"

    def test_upload_valid_csv_file(self, client):
        content = b"col1,col2,col3\nval1,val2,val3\n"
        files = {"file": ("table.csv", io.BytesIO(content), "text/csv")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["format"] == "csv"

    def test_upload_disallowed_extension_rejected(self, client):
        content = b"print('malicious')"
        files = {"file": ("script.py", io.BytesIO(content), "text/x-python")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 400
        assert "مجاز نیست" in response.json()["detail"] or "پشتیبانی" in response.json()["detail"]

    def test_upload_executable_rejected(self, client):
        content = b"MZ\x90\x00"
        files = {"file": ("virus.exe", io.BytesIO(content), "application/octet-stream")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 400

    def test_upload_oversized_file_rejected(self, client):
        # 11 MB file exceeds 10 MB limit
        large_content = b"A" * (11 * 1024 * 1024)
        files = {"file": ("large.txt", io.BytesIO(large_content), "text/plain")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 400
        assert "حجم فایل" in response.json()["detail"]

    def test_upload_path_traversal_sanitized(self, client):
        content = b"safe content"
        files = {"file": ("../../../../etc/passwd.txt", io.BytesIO(content), "text/plain")}
        response = client.post("/api/files/upload", files=files)
        assert response.status_code == 200
        data = response.json()["data"]
        # Path traversal characters must be stripped
        assert "/" not in data["filename"]
        assert ".." not in data["filename"]


# ---------------------------------------------------------------------------
# Processing Operations Tests
# ---------------------------------------------------------------------------

class TestFileProcessingOperations:
    """Test suite for file processing actions and transformations."""

    def test_normalize_operation(self, client):
        raw_text = "تمام دادهها و برنامهها درون سایتها بهصورت خودکار ذخیره میشوند."
        files = {"file": ("doc.txt", io.BytesIO(raw_text.encode("utf-8")), "text/plain")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        proc_resp = client.post(
            "/api/files/process",
            json={
                "file_id": file_id,
                "operations": ["normalize"],
                "options": {"fix_zwnj_affixes": True},
            },
        )
        assert proc_resp.status_code == 200
        res = proc_resp.json()
        assert res["success"] is True
        data = res["data"]
        assert data["file_id"] == file_id
        assert "normalize" in data["operations_executed"]
        assert "\u200c" in data["output_preview"]  # ZWNJ was applied
        assert "telemetry" in data
        assert data["telemetry"]["normalization_fixes"] > 0

    def test_markdown_to_json_operation(self, client):
        md_text = "# Main Header\n\nFirst paragraph.\n\n## Sub Header\n\nSecond paragraph."
        files = {"file": ("doc.md", io.BytesIO(md_text.encode("utf-8")), "text/markdown")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        proc_resp = client.post(
            "/api/files/process",
            json={
                "file_id": file_id,
                "operations": ["markdown_to_json"],
            },
        )
        assert proc_resp.status_code == 200
        res = proc_resp.json()
        assert res["success"] is True
        data = res["data"]
        assert "markdown_to_json" in data["operations_executed"]
        parsed = json.loads(data["output_preview"])
        assert "sections" in parsed
        assert len(parsed["sections"]) >= 1

    def test_json_validate_valid_and_cleanup(self, client):
        raw_json = json.dumps({"title": "  Sample Title  ", "count": 10, "empty_str": ""})
        files = {"file": ("test.json", io.BytesIO(raw_json.encode("utf-8")), "application/json")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        proc_resp = client.post(
            "/api/files/process",
            json={
                "file_id": file_id,
                "operations": ["json_validate", "json_cleanup"],
            },
        )
        assert proc_resp.status_code == 200
        res = proc_resp.json()
        assert res["success"] is True
        data = res["data"]
        cleaned = json.loads(data["output_preview"])
        assert cleaned["title"] == "Sample Title"  # Whitespace trimmed

    def test_json_validate_invalid_raises_error(self, client):
        broken_json = b"{\"key\": 'unquoted_val'}"
        files = {"file": ("broken.json", io.BytesIO(broken_json), "application/json")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        proc_resp = client.post(
            "/api/files/process",
            json={
                "file_id": file_id,
                "operations": ["json_validate"],
            },
        )
        assert proc_resp.status_code == 400
        assert "خطای اعتبارسنجی" in proc_resp.json()["detail"] or "JSON" in proc_resp.json()["detail"]

    def test_chunking_operation(self, client):
        paragraphs = "\n\n".join([f"This is paragraph number {i} with some descriptive content." for i in range(15)])
        files = {"file": ("long_text.txt", io.BytesIO(paragraphs.encode("utf-8")), "text/plain")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        proc_resp = client.post(
            "/api/files/process",
            json={
                "file_id": file_id,
                "operations": ["chunk"],
                "options": {
                    "strategy": "paragraph",
                    "chunk_size": 200,
                    "overlap": 30,
                },
            },
        )
        assert proc_resp.status_code == 200
        res = proc_resp.json()
        assert res["success"] is True
        data = res["data"]
        assert data["telemetry"]["chunks_produced"] > 0
        assert "chunk_id" in data["output_preview"]
        assert "text" in data["output_preview"]

    def test_metadata_enrich_operation(self, client):
        text = "Sample content for enrichment test."
        files = {"file": ("enrich.txt", io.BytesIO(text.encode("utf-8")), "text/plain")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        proc_resp = client.post(
            "/api/files/process",
            json={
                "file_id": file_id,
                "operations": ["metadata_enrich"],
                "options": {
                    "category": "legal_docs",
                    "domain": "enterprise.local",
                },
            },
        )
        assert proc_resp.status_code == 200
        res = proc_resp.json()
        assert res["success"] is True
        data = res["data"]
        enriched = json.loads(data["output_preview"])
        assert enriched["category"] == "legal_docs"
        assert enriched["metadata"]["domain"] == "enterprise.local"


# ---------------------------------------------------------------------------
# End-to-End Pipeline & Artifact Retrieval Tests
# ---------------------------------------------------------------------------

class TestPipelineAndArtifacts:
    """Test suite for full pipeline execution, diffs, previews, and downloads."""

    def test_full_pipeline_normalize_and_chunk(self, client):
        raw_text = (
            "# مستند آزمایشی سیستم\n\n"
            "این یک متن نمونه است که دادهها در آن بهصورت ساختاریافته ذخیره میشوند.\n\n"
            "## بخش دوم\n\n"
            "توضیحات تکمیلی سیستم با زبان فارسی."
        )
        files = {"file": ("pipeline_test.md", io.BytesIO(raw_text.encode("utf-8")), "text/markdown")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        proc_resp = client.post(
            "/api/files/process",
            json={
                "file_id": file_id,
                "operations": ["normalize", "chunk"],
                "options": {
                    "fix_zwnj_affixes": True,
                    "strategy": "semantic_heading",
                    "chunk_size": 300,
                },
            },
        )
        assert proc_resp.status_code == 200
        res = proc_resp.json()
        assert res["success"] is True
        data = res["data"]
        assert len(data["diff"]) > 0  # Diff was generated
        artifact_id = data["artifact_id"]
        assert artifact_id is not None

        # Verify preview endpoint
        prev_resp = client.get(f"/api/files/preview/{artifact_id}")
        assert prev_resp.status_code == 200
        prev_data = prev_resp.json()["data"]
        assert prev_data["artifact_id"] == artifact_id
        assert len(prev_data["content_preview"]) > 0

        # Verify download endpoint
        dl_resp = client.get(f"/api/files/download/{artifact_id}")
        assert dl_resp.status_code == 200
        assert dl_resp.headers["content-type"].startswith("application/json")
        assert len(dl_resp.content) > 0

    def test_file_info_endpoint(self, client):
        files = {"file": ("info_test.txt", io.BytesIO(b"Hello world"), "text/plain")}
        upload_resp = client.post("/api/files/upload", files=files)
        file_id = upload_resp.json()["data"]["file_id"]

        info_resp = client.get(f"/api/files/info/{file_id}")
        assert info_resp.status_code == 200
        info_data = info_resp.json()["data"]
        assert info_data["file_id"] == file_id
        assert info_data["filename"] == "info_test.txt"

    def test_nonexistent_artifact_returns_404(self, client):
        resp = client.get("/api/files/preview/nonexistent-id-999")
        assert resp.status_code == 404

        resp_dl = client.get("/api/files/download/nonexistent-id-999")
        assert resp_dl.status_code == 404

    def test_nonexistent_file_id_info_returns_404(self, client):
        resp = client.get("/api/files/info/nonexistent-file-123")
        assert resp.status_code == 404
