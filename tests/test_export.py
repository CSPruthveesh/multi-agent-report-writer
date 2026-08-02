"""Exporting a report to .docx and .pdf.

Both formats walk one parser, so the parser is where the tests are. A block rule that
is wrong is wrong in both files at once — which is the trade for not having two
markdown readers, and it is worth having on purpose rather than by accident.
"""

from __future__ import annotations

import io
import re

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from api.export import blocks, font_index, pdf_family, resolve_pdf_font, to_docx, to_pdf
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
    """The fallback mapping, used only when the font file cannot be found.

    Embedding made this the exception rather than the rule, but it still has to be a
    near match: a serif must not fall back to a sans.
    """
    assert pdf_family(font) == family


def test_both_formats_produce_a_file_that_opens():
    md = SAMPLE
    docx, pdf = to_docx(md, title="t"), to_pdf(md, title="t")
    assert docx[:2] == b"PK", "not a zip, so not a .docx"
    assert pdf[:5] == b"%PDF-"
    assert len(docx) > 5_000 and len(pdf) > 1_000


def test_the_pdf_says_what_the_report_says():
    """Magic bytes prove a PDF exists, not that the report is in it.

    A renderer that dropped every paragraph would still emit a valid three-page file
    starting %PDF-, which is what the check above was really asserting.
    """
    reader = PdfReader(io.BytesIO(to_pdf(SAMPLE, title="t")))
    text = "\n".join(p.extract_text() for p in reader.pages)
    assert "A title" in text
    assert "Known limitations" in text
    assert "[F004]" in text, "the citation was lost, which is the only reason to trust it"
    assert "A second one that wraps onto a continuation line." in text.replace("\n", " ")


# reportlab's canvas registers Helvetica as its initial font on every page, used or
# not — a Times-only document still lists it as a resource. Verified by generating a
# report with no bullets and no bold: it is there either way. So it is excluded here
# rather than asserted against, which is the difference between a test that describes
# the library and one that describes this code.
CANVAS_DEFAULT = {"Helvetica"}
SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")


def _fonts_in(pdf: bytes) -> set[str]:
    """Every font resource named in the file, with subset prefixes stripped."""
    used = set()
    for page in PdfReader(io.BytesIO(pdf)).pages:
        for res in (page.get("/Resources", {}).get("/Font", {}) or {}).values():
            name = str(res.get_object()["/BaseFont"]).lstrip("/")
            used.add(SUBSET_PREFIX.sub("", name))
    return used


def _installed(family: str) -> bool:
    return family.lower() in font_index()


@pytest.mark.parametrize("family", ["Georgia", "Consolas", "Palatino Linotype"])
def test_an_installed_font_is_embedded_rather_than_substituted(family):
    """The glyphs travel inside the file, so it reads the same on any machine.

    Skipped where the family is not on this one — asserting that a CI box has Georgia
    would be testing the box.
    """
    if not _installed(family):
        pytest.skip(f"{family} is not installed here")

    assert resolve_pdf_font(family)[2] == family, "reported as substituted"
    pdf = to_pdf(SAMPLE, font=family)
    flat = family.replace(" ", "")
    assert any(flat.lower() in f.lower() for f in _fonts_in(pdf)), (
        f"{family} is not among the file's font resources: {_fonts_in(pdf)}"
    )
    assert len(pdf) > 20_000, "no font data in the file — nothing was embedded"


def test_a_font_this_machine_lacks_falls_back_to_a_near_match():
    """Archivo and JetBrains Mono are webfonts. The browser has them; the OS does not."""
    for family, expected in [("Archivo", "Helvetica"), ("JetBrains Mono", "Courier")]:
        if _installed(family):
            continue
        assert resolve_pdf_font(family)[2] == expected
        assert expected in _fonts_in(to_pdf(SAMPLE, font=family))


def test_no_family_leaks_across_the_fallback_mapping():
    """A Courier request must not produce Times, embedded or otherwise."""
    used = _fonts_in(to_pdf(SAMPLE, font="Nonesuch Font"))
    assert used <= {"Helvetica", "Helvetica-Bold"} | CANVAS_DEFAULT, used


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
        """Whatever happened, the header says which family the file is set in."""
        r = self.client.post("/api/export/pdf",
                             json={"report": SAMPLE, "font": "Candara"})
        assert r.status_code == 200
        assert r.content[:5] == b"%PDF-"
        expected = "Candara" if _installed("Candara") else "Helvetica"
        assert r.headers["X-Font-Applied"] == expected

    def test_a_font_nothing_can_supply_is_reported_as_the_substitute(self):
        r = self.client.post("/api/export/pdf",
                             json={"report": SAMPLE, "font": "Nonesuch Font"})
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
