"""Turn a finished report into a .docx or a .pdf.

ONE PARSER, TWO RENDERERS
-------------------------
The markdown the Writer emits is small and known: one H1, H2 sections, paragraphs,
bold, horizontal rules, bullets under "Known limitations", and [F001] citations. So
`blocks()` reads it once into a flat list and each renderer walks that list.

The alternative — a markdown reader inside each exporter — is the mistake the front end
already made and fixed: the trace was rendered by two copies of the same code and they
drifted until the pages disagreed about what a run looked like. A report that paginates
differently depending on which button was pressed is the same defect with a longer
feedback loop.

FONTS
-----
The page lets a reader put the report in any font their machine has, and the export
honours that as far as each format honestly can.

  .docx  carries the family by name. Word resolves it on whatever machine opens the
         file, which is exactly the right behaviour for a document that travels.
  .pdf   embeds nothing, so it is limited to the three families reportlab ships:
         Times, Helvetica and Courier. A requested font is mapped onto the nearest of
         those and the mapping is reported back to the caller in a header, rather than
         silently producing a document in a font nobody asked for.

Registering the machine's TTFs with reportlab would remove that limit and tie the
server to a Windows font directory. Not worth it for a demo that has to run anywhere.
"""

from __future__ import annotations

import io
import re
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.platypus import Paragraph as RLParagraph

BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
CITE_RE = re.compile(r"\[((?:F\d{3})(?:,\s*F\d{3})*)\]")

# What the page offers, grouped by what a PDF can actually do with it. Anything not
# named here falls back to Helvetica, which is also what an unknown font should do.
_SERIF = {
    "georgia", "times new roman", "cambria", "constantia", "palatino linotype",
    "book antiqua", "garamond", "sylfaen", "charter", "serif",
}
_MONO = {"consolas", "courier new", "jetbrains mono", "monospace"}


def pdf_family(font: str | None) -> str:
    """The reportlab family a requested font maps onto."""
    name = (font or "").strip().lower()
    if name in _MONO:
        return "Courier"
    if name in _SERIF:
        return "Times-Roman"
    return "Helvetica"


def blocks(md: str) -> list[tuple[str, str]]:
    """Flatten the report into (kind, text) pairs.

    Kinds: h1, h2, rule, li, p. Line endings are normalised first — the same \\r\\n
    assumption that collapsed the whole report into one block in the browser renderer
    would collapse it into one paragraph here.
    """
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"\n{2,}", md.replace("\r\n", "\n").replace("\r", "\n")):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("# "):
            out.append(("h1", chunk[2:].strip()))
        elif chunk.startswith("## "):
            out.append(("h2", chunk[3:].strip()))
        elif re.fullmatch(r"-{3,}", chunk):
            out.append(("rule", ""))
        elif re.match(r"^\s*[-*]\s", chunk):
            # A wrapped bullet continues the item above it rather than opening an
            # empty one — the same rule the browser renderer applies.
            for line in chunk.split("\n"):
                m = re.match(r"^\s*[-*]\s+(.*)$", line)
                if m:
                    out.append(("li", m.group(1).strip()))
                elif out and out[-1][0] == "li" and line.strip():
                    out[-1] = ("li", f"{out[-1][1]} {line.strip()}")
        else:
            out.append(("p", " ".join(chunk.split())))
    return out


def _runs(text: str) -> list[tuple[str, bool]]:
    """Split on **bold** into (text, is_bold) pairs."""
    parts, last = [], 0
    for m in BOLD_RE.finditer(text):
        if m.start() > last:
            parts.append((text[last:m.start()], False))
        parts.append((m.group(1), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))
    return parts or [(text, False)]


# ------------------------------------------------------------------------ docx
def to_docx(md: str, *, title: str = "", font: str | None = None) -> bytes:
    doc = Document()
    if font:
        # The Normal style, so every paragraph inherits it rather than each one
        # carrying its own run-level override.
        doc.styles["Normal"].font.name = font
    doc.styles["Normal"].font.size = Pt(11)

    if title:
        doc.core_properties.title = title

    for kind, text in blocks(md):
        if kind == "rule":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run("* * *")
            continue
        if kind == "h1":
            doc.add_heading(text, level=0)
            continue
        if kind == "h2":
            doc.add_heading(text, level=1)
            continue

        p = doc.add_paragraph(style="List Bullet" if kind == "li" else None)
        # Citations get their own runs so they can be coloured. They are the reason to
        # trust the document, and in a printed report they are the only thing that
        # links a claim back to a source.
        for chunk, bold in _runs(text):
            for i, piece in enumerate(CITE_RE.split(chunk)):
                if not piece:
                    continue
                run = p.add_run(f"[{piece}]" if i % 2 else piece)
                run.bold = bold
                if i % 2:
                    run.font.color.rgb = RGBColor(0x0F, 0x6E, 0x6B)
                    run.font.size = Pt(9)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------------------- pdf
def _xml(text: str) -> str:
    """Escape for reportlab's mini-HTML, then put the bold tags back."""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return BOLD_RE.sub(r"<b>\1</b>", safe)


def to_pdf(md: str, *, title: str = "", font: str | None = None) -> bytes:
    fam = pdf_family(font)
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "body", parent=base["BodyText"], fontName=fam, fontSize=10.5, leading=15.5,
        alignment=TA_JUSTIFY, spaceAfter=9,
    )
    h1 = ParagraphStyle(
        "h1", parent=base["Title"], fontName=f"{fam}-Bold" if fam != "Times-Roman"
        else "Times-Bold", fontSize=19, leading=23, spaceAfter=14, alignment=0,
    )
    h2 = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName=f"{fam}-Bold" if fam != "Times-Roman"
        else "Times-Bold", fontSize=12.5, leading=16, spaceBefore=13, spaceAfter=5,
    )

    flow: list[Any] = []
    bullets: list[ListItem] = []

    def flush() -> None:
        if bullets:
            flow.append(ListFlowable(list(bullets), bulletType="bullet",
                                     leftIndent=14, bulletFontSize=7))
            flow.append(Spacer(1, 7))
            bullets.clear()

    for kind, text in blocks(md):
        if kind != "li":
            flush()
        if kind == "h1":
            flow.append(RLParagraph(_xml(text), h1))
        elif kind == "h2":
            flow.append(RLParagraph(_xml(text), h2))
        elif kind == "rule":
            flow.append(Spacer(1, 5))
            flow.append(HRFlowable(width="100%", thickness=0.6, color="#C3CACC"))
            flow.append(Spacer(1, 9))
        elif kind == "li":
            bullets.append(ListItem(RLParagraph(_xml(text), body), leftIndent=14))
        else:
            flow.append(RLParagraph(_xml(text), body))
    flush()

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=24 * mm, rightMargin=24 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title=title or "Report", author="Multi-Agent Report Writer",
    ).build(flow)
    return buf.getvalue()
