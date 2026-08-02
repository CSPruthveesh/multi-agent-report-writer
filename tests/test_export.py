"""Exporting a report to .docx and .pdf.

Both formats walk one parser, so the parser is where the tests are. A block rule that
is wrong is wrong in both files at once — which is the trade for not having two
markdown readers, and it is worth having on purpose rather than by accident.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.export import blocks, pdf_family, to_docx, to_pdf
from api.main import app
from src.common.io import RESULTS

REPORT = RESULTS / "multiagent" / "t1" / "report.md"

SAMPLE = """# A title

An opening paragraph
wrapped across two lines.

## A section

Text with **bold** and a citation [F004] in it.

---

## Known limitations

- A first limitation.
- A second one that wraps
  onto a continuation line.
"""


def test_the_parser_reads_every_kind_the_writer_emits():
    got = blocks(SAMPLE)
    kinds = [k for k, _ in got]
    assert kinds == ["h1", "p", "h2", "p", "rule", "h2", "li", "li"]
    assert got[1][1] == "An opening paragraph wrapped across two lines."
    assert got[-1][1] == "A second one that wraps onto a continuation line.", (
        "a wrapped bullet opened a new item instead of continuing the one above"
    )


def test_crlf_does_not_collapse_the_report_into_one_block():
    """The defect that hit the browser renderer, in the exporter's own words.

    Every block rule here splits on a blank line. \\r\\n\\r\\n is not \\n\\n, so without
    normalising, a report with Windows endings becomes a single paragraph — one wall of
    text in the .pdf, and no headings at all in the .docx.
    """
    assert blocks(SAMPLE.replace("\n", "\r\n")) == blocks(SAMPLE)


@pytest.mark.parametrize(
    ("font", "family"),
    [
        ("Georgia", "Times-Roman"), ("Cambria", "Times-Roman"),
        ("Consolas", "Courier"), ("JetBrains Mono", "Courier"),
        ("Segoe UI", "Helvetica"), ("Archivo", "Helvetica"),
        ("  gEoRgIa ", "Times-Roman"),
        (None, "Helvetica"), ("", "Helvetica"), ("Nonesuch", "Helvetica"),
    ],
)
def test_a_requested_font_maps_onto_one_a_pdf_can_assume(font, family):
    """A PDF embeds nothing here, so it gets the nearest of three — never a guess."""
    assert pdf_family(font) == family


def test_both_formats_produce_a_file_that_opens():
    md = SAMPLE
    docx, pdf = to_docx(md, title="t"), to_pdf(md, title="t")
    assert docx[:2] == b"PK", "not a zip, so not a .docx"
    assert pdf[:5] == b"%PDF-"
    assert len(docx) > 5_000 and len(pdf) > 1_000


@pytest.mark.skipif(not REPORT.exists(), reason="no recorded report")
def test_a_real_report_survives_both_renderers():
    md = REPORT.read_text(encoding="utf-8")
    kinds = {k for k, _ in blocks(md)}
    assert {"h1", "h2", "p", "li"} <= kinds
    assert to_docx(md, title="t1", font="Georgia")[:2] == b"PK"
    assert to_pdf(md, title="t1", font="Georgia")[:5] == b"%PDF-"


class TestEndpoint:
    client = TestClient(app)

    def test_docx_download_names_the_file_and_keeps_the_font(self):
        r = self.client.post("/api/export/docx",
                             json={"report": SAMPLE, "title": "Grid storage!! 2026",
                                   "font": "Georgia"})
        assert r.status_code == 200
        assert r.content[:2] == b"PK"
        # The slug is bounded and alphanumeric because it lands in a header.
        assert 'filename="grid-storage-2026.docx"' in r.headers["content-disposition"]
        assert r.headers["X-Font-Applied"] == "Georgia"

    def test_pdf_reports_the_family_it_actually_used(self):
        r = self.client.post("/api/export/pdf",
                             json={"report": SAMPLE, "font": "Candara"})
        assert r.status_code == 200
        assert r.content[:5] == b"%PDF-"
        assert r.headers["X-Font-Applied"] == "Helvetica", (
            "the caller must be told it did not get the font it asked for"
        )

    def test_an_unknown_format_is_refused(self):
        assert self.client.post("/api/export/rtf", json={"report": "x"}).status_code == 400

    # The size, not the string. pytest writes the test id into PYTEST_CURRENT_TEST, and
    # Windows caps an environment variable at 32767 characters — a 200k parameter fails
    # in teardown with an error about the environment, nowhere near the actual test.
    @pytest.mark.parametrize("size", [0, 200_001], ids=["empty", "oversized"])
    def test_empty_and_oversized_reports_are_refused(self, size):
        r = self.client.post("/api/export/pdf", json={"report": "x" * size})
        assert r.status_code == 422

    def test_a_titleless_report_still_gets_a_filename(self):
        r = self.client.post("/api/export/pdf", json={"report": SAMPLE})
        assert 'filename="report.pdf"' in r.headers["content-disposition"]
