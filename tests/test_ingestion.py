from pathlib import Path
from unittest.mock import mock_open, patch

import pymupdf
import pytest

from pharmadoc_ai.config import Settings
from pharmadoc_ai.ingestion import (
    InvalidDocumentError,
    detect_file_type,
    inspect_document,
    open_document,
)


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("rx.pdf", b"%PDF-1.7\n", "pdf"),
        ("rx.PDF", b"%PDF-1.7\n", "pdf"),
        ("scan.png", b"\x89PNG\r\n\x1a\n", "png"),
        ("scan.jpg", b"\xff\xd8\xff\xe0", "jpeg"),
        ("scan.jpeg", b"\xff\xd8\xff\xe1", "jpeg"),
    ],
)
def test_matching_signature(tmp_path, name, content, expected):
    path = tmp_path / name
    path.write_bytes(content)
    assert detect_file_type(path) == expected
    assert path.read_bytes() == content


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("scan.png", b"not an image"),
        ("scan.pdf", b"\x89PNG\r\n\x1a\n"),
        ("scan.gif", b"GIF89a"),
        ("scan.exe", b"%PDF-1.7\n"),
        ("empty.pdf", b""),
        ("partial.png", b"\x89PNG"),
    ],
)
def test_invalid_signature(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(InvalidDocumentError):
        detect_file_type(path)


def test_missing_and_directory(tmp_path):
    for path in (tmp_path / "missing.pdf", tmp_path):
        with pytest.raises(InvalidDocumentError):
            detect_file_type(path)


def test_inclusive_limit(tmp_path):
    path = tmp_path / "rx.pdf"
    path.write_bytes(b"%PDF-1.7")
    assert detect_file_type(path, 8) == "pdf"
    with pytest.raises(InvalidDocumentError):
        detect_file_type(path, 7)


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limit(tmp_path, limit):
    with pytest.raises(InvalidDocumentError):
        detect_file_type(tmp_path / "rx.pdf", limit)


def test_reads_only_eight_bytes(tmp_path):
    path = tmp_path / "rx.pdf"
    path.write_bytes(b"%PDF-1.7\n" + b"x" * 100)
    opened = mock_open(read_data=b"%PDF-1.7")
    with patch.object(Path, "open", opened):
        assert detect_file_type(path) == "pdf"
    opened().read.assert_called_once_with(8)


def test_precheck_is_not_decode(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.7\ngarbage")
    assert detect_file_type(path) == "pdf"
    with pytest.raises(InvalidDocumentError):
        inspect_document(path)


def test_pdf_routes_and_page_limit(tmp_path):
    path = tmp_path / "pages.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50, 60), "Patient: Synthetic Person\nPrescription record")
        doc.new_page()
        doc.save(path)
    pages = inspect_document(path)
    assert [p.route for p in pages] == ["digital", "ocr"]
    assert pages[0].words[0].bbox[0] >= 0
    with pytest.raises(InvalidDocumentError, match="page limit"):
        inspect_document(path, Settings(max_pages=1))


def test_reject_encrypted_pdf(tmp_path):
    path = tmp_path / "encrypted.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="synthetic")
    with pytest.raises(InvalidDocumentError, match="Encrypted"):
        with open_document(path, Settings()):
            pass
