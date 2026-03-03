#!/usr/bin/env python
"""
Generate a formatted PDF project report for the Crime Scene Reconstruction project.
Follows academic report template structure: Title, Abstract, TOC, Chapters 1-5, References, Appendix.
Uses reportlab for professional PDF generation with Times New Roman, tables, and embedded images.

Fixes applied:
- Table cells wrapped in Paragraph for proper bold/italic rendering
- Unicode subscript replaced with plain text (w1, w2, ...)
- Page numbers in Table of Contents
- Page border on every page
- Major chapters start on new pages
- Appendix II removed
- Title page matches sample template (name, roll number, signature fields)
"""

import os
import sys
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image as RLImage, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.colors import Color

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PDF = PROJECT_ROOT / "Crime_Scene_Reconstruction_Project_Report.pdf"

# ── Latest run directory ──────────────────────────────────────────────────
RUN_DIR = PROJECT_ROOT / "outputs" / "run_20260302_025355"  # latest run with RV5.1
IMAGES_DIR = RUN_DIR / "images"
DEPTH_DIR = RUN_DIR / "depth_maps"
GRAPH_DIR = RUN_DIR / "scene_graphs"

# ── Colors ────────────────────────────────────────────────────────────────
HEADER_BG = HexColor("#D5E8F0")
TABLE_BORDER = HexColor("#444444")

WIDTH, HEIGHT = A4  # 595.27, 841.89 points

# ── Cell styles for tables ────────────────────────────────────────────────
CELL_STYLE = ParagraphStyle(
    "CellNormal", fontName="Times-Roman", fontSize=11, leading=14,
    alignment=TA_CENTER,
)
CELL_BOLD = ParagraphStyle(
    "CellBold", fontName="Times-Bold", fontSize=11, leading=14,
    alignment=TA_CENTER,
)
CELL_HEADER = ParagraphStyle(
    "CellHeader", fontName="Times-Bold", fontSize=11, leading=14,
    alignment=TA_CENTER,
)
CELL_LEFT = ParagraphStyle(
    "CellLeft", fontName="Times-Roman", fontSize=11, leading=14,
    alignment=TA_LEFT,
)


# ═══════════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════════
def build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "TitlePage", fontName="Times-Bold", fontSize=22, leading=28,
        alignment=TA_CENTER, spaceAfter=12, textColor=black,
    ))
    ss.add(ParagraphStyle(
        "SubTitle", fontName="Times-Roman", fontSize=14, leading=18,
        alignment=TA_CENTER, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "SubTitleBold", fontName="Times-Bold", fontSize=14, leading=18,
        alignment=TA_CENTER, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "ChapterTitle", fontName="Times-Bold", fontSize=16, leading=22,
        spaceBefore=18, spaceAfter=12, alignment=TA_LEFT,
    ))
    ss.add(ParagraphStyle(
        "Section", fontName="Times-Bold", fontSize=14, leading=18,
        spaceBefore=12, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "SubSection", fontName="Times-Bold", fontSize=13, leading=17,
        spaceBefore=10, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "BodyText14", fontName="Times-Roman", fontSize=12, leading=18,
        alignment=TA_JUSTIFY, spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "Caption", fontName="Times-Italic", fontSize=11, leading=14,
        alignment=TA_CENTER, spaceAfter=10,
    ))
    ss.add(ParagraphStyle(
        "CodeBlock", fontName="Courier", fontSize=9, leading=12,
        spaceBefore=6, spaceAfter=6, leftIndent=24,
        backColor=HexColor("#F0F0F0"),
    ))
    ss.add(ParagraphStyle(
        "TOCHeading", fontName="Times-Bold", fontSize=16, leading=22,
        spaceBefore=12, spaceAfter=12, alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        "BulletBody", fontName="Times-Roman", fontSize=12, leading=18,
        alignment=TA_JUSTIFY, leftIndent=36, bulletIndent=18, spaceAfter=4,
    ))
    ss.add(ParagraphStyle(
        "SignatureField", fontName="Times-Roman", fontSize=12, leading=16,
        alignment=TA_LEFT, spaceAfter=2,
    ))
    return ss


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════
def make_table(data, col_widths=None):
    """Create a nicely formatted table. All cells are Paragraph-wrapped."""
    wrapped = []
    for r_idx, row in enumerate(data):
        new_row = []
        for cell in row:
            if isinstance(cell, Paragraph):
                new_row.append(cell)
            elif r_idx == 0:
                new_row.append(Paragraph(str(cell), CELL_HEADER))
            else:
                new_row.append(Paragraph(str(cell), CELL_STYLE))
        wrapped.append(new_row)

    t = Table(wrapped, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]
    t.setStyle(TableStyle(style))
    return t


def safe_image(path, width=4.5 * inch, max_height=3.5 * inch):
    """Return an Image flowable if the file exists, else a placeholder paragraph."""
    p = Path(path)
    if p.exists():
        try:
            img = RLImage(str(p), width=width)
            ratio = img.imageWidth / img.imageHeight
            h = width / ratio
            if h > max_height:
                h = max_height
                w = h * ratio
                img = RLImage(str(p), width=w, height=h)
            else:
                img = RLImage(str(p), width=width, height=h)
            return img
        except Exception:
            pass
    return Paragraph(f"<i>[Image: {p.name} -- not available]</i>",
                     getSampleStyleSheet()["Normal"])


def bullet_list(items, style):
    """Return a list of bullet-pointed paragraphs."""
    return [Paragraph(f"&bull;  {item}", style) for item in items]


# ═══════════════════════════════════════════════════════════════════════════
# Page callbacks -- border + page number
# ═══════════════════════════════════════════════════════════════════════════
BORDER_MARGIN = 0.5 * inch

def add_page_border_and_number(canvas, doc):
    """Draw a rectangular border on every page and add page number."""
    canvas.saveState()

    # Border
    canvas.setStrokeColor(black)
    canvas.setLineWidth(1.5)
    bx = BORDER_MARGIN
    by = BORDER_MARGIN
    bw = WIDTH - 2 * BORDER_MARGIN
    bh = HEIGHT - 2 * BORDER_MARGIN
    canvas.rect(bx, by, bw, bh)

    # Page number
    page_num = canvas.getPageNumber()
    canvas.setFont("Times-Roman", 11)
    canvas.drawCentredString(WIDTH / 2, 0.35 * inch, str(page_num))

    canvas.restoreState()


# ═══════════════════════════════════════════════════════════════════════════
# Build content
# ═══════════════════════════════════════════════════════════════════════════
def build_story(ss):
    story = []
    body = ss["BodyText14"]
    caption = ss["Caption"]
    bullet = ss["BulletBody"]

    # ──────────────────────────────────────────────────────────────────────
    # TITLE PAGE (matching sample template exactly)
    # ──────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph("PROJECT REPORT", ss["TitlePage"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("on", ss["SubTitle"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Probabilistic Multi-View Crime Scene Reconstruction<br/>"
        "from Natural Language Descriptions Using<br/>"
        "Unified Multi-Objective Scoring",
        ParagraphStyle("TitleMain", fontName="Times-Bold", fontSize=18, leading=24,
                        alignment=TA_CENTER, spaceAfter=8)
    ))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Submitted by", ss["SubTitle"]))
    story.append(Spacer(1, 0.15 * inch))

    # Student info block
    story.append(Paragraph("Name: <b>Pranav</b>", ss["SignatureField"]))
    story.append(Paragraph("Roll Number: <b>[Your Roll Number]</b>", ss["SignatureField"]))
    story.append(Spacer(1, 0.5 * inch))

    story.append(Paragraph("Under the Guidance of", ss["SubTitle"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("<b>[Guide Name]</b>, <i>[Designation]</i>", ss["SubTitle"]))
    story.append(Spacer(1, 0.6 * inch))

    # Signature block
    sig_data = [
        [Paragraph("Signature of Student", CELL_STYLE),
         Paragraph("Signature of Guide", CELL_STYLE)],
        [Paragraph("Name: Pranav", CELL_STYLE),
         Paragraph("Name: [Guide Name]", CELL_STYLE)],
        [Paragraph("Roll Number: [Your Roll Number]", CELL_STYLE),
         Paragraph("Designation: [Designation]", CELL_STYLE)],
        [Paragraph("Date:", CELL_STYLE),
         Paragraph("Date:", CELL_STYLE)],
    ]
    sig_table = Table(sig_data, colWidths=[3.0 * inch, 3.0 * inch], hAlign="CENTER")
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_table)

    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Department of Computer Science", ss["SubTitleBold"]))
    story.append(Paragraph("2026", ss["SubTitle"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("<i>Note: Please add more if applicable.</i>",
                           ParagraphStyle("Note", fontName="Times-Italic", fontSize=10,
                                          alignment=TA_CENTER)))
    story.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────────
    # ACKNOWLEDGMENT
    # ──────────────────────────────────────────────────────────────────────
    story.append(Paragraph("ACKNOWLEDGMENT", ss["ChapterTitle"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "We would like to express our sincere gratitude to all those who have contributed "
        "to the successful completion of this project on Probabilistic Multi-View Crime "
        "Scene Reconstruction from Natural Language Descriptions.",
        body
    ))
    story.append(Paragraph(
        "First and foremost, we extend our heartfelt thanks to our project guide, "
        "Dr. Anand Kakarla, Assistant Dean - Academic Affairs, for their invaluable guidance, "
        "continuous support, and constructive feedback throughout the duration of this project. "
        "Their expertise in deep learning and computer vision has been instrumental in shaping this work.",
        body
    ))
    story.append(Paragraph(
        "We would also like to thank the open-source community -- Stability AI, Hugging Face, "
        "the Visual Genome project, and the COCO and CLEVR dataset teams -- for providing "
        "the foundational models and datasets that made this work possible.",
        body
    ))
    story.append(Paragraph(
        "Our sincere thanks to our peers and colleagues who provided valuable insights and "
        "suggestions during various phases of this project.",
        body
    ))
    story.append(Paragraph(
        "Finally, we are deeply grateful to our families for their unwavering support and "
        "encouragement throughout our academic journey.",
        body
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Pranav Adusumilli - 22WU0104088", body))
    story.append(Paragraph("Pranav Nihar - 22WU0104089", body))
    story.append(Paragraph("Kesa Vivek - 22WU0104060", body))
    story.append(Paragraph("Yeddula Deva Harsha - 22WU0105029", body))
    story.append(Paragraph("Swamy Sharvana - 22WU0104117", body))
    story.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────────
    # ABSTRACT
    # ──────────────────────────────────────────────────────────────────────
    story.append(Paragraph("ABSTRACT", ss["ChapterTitle"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "This project presents a research-grade unified multi-objective probabilistic "
        "framework for reconstructing crime scenes from natural language descriptions. "
        "The system implements a 12-stage pipeline that transforms textual witness "
        "accounts into photorealistic scene reconstructions using controlled diffusion "
        "models. The pipeline integrates Natural Language Processing (spaCy), scene "
        "graph construction (NetworkX), probabilistic spatial layout estimation, "
        "depth-conditioned image generation (ControlNet + Realistic Vision v5.1), and "
        "a novel 7-component unified scoring function S(R) that evaluates semantic "
        "alignment, spatial consistency, physical plausibility, visual realism, "
        "probabilistic prior likelihood, multi-view consistency, and perceptual "
        "believability.",
        body
    ))
    story.append(Paragraph(
        "The system further incorporates energy-based optimization (Simulated Annealing "
        "and Evolutionary Strategies), closed-loop correction, and an ablation study "
        "framework for systematic evaluation. Experiments on bedroom and kitchen crime "
        "scenes demonstrate a best unified score of S(R) = 0.739, with component scores "
        "reaching 0.999 for spatial consistency and 0.923 for physical plausibility. "
        "The entire system operates on a consumer-grade NVIDIA RTX 3060 (6 GB VRAM) "
        "using sequential CPU offloading.",
        body
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "<b>Keywords:</b> Diffusion Models, ControlNet, Crime Scene Reconstruction, "
        "Scene Graphs, Multi-Objective Scoring, Spatial Layout, CLIP Evaluation, "
        "Probabilistic Inference, Generative AI",
        body
    ))
    story.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────────
    # TABLE OF CONTENTS (with page numbers)
    # ──────────────────────────────────────────────────────────────────────
    story.append(Paragraph("TABLE OF CONTENTS", ss["TOCHeading"]))
    story.append(Spacer(1, 0.2 * inch))

    toc_data = [
        [Paragraph("Acknowledgment", CELL_LEFT), Paragraph("2", CELL_STYLE)],
        [Paragraph("Abstract", CELL_LEFT), Paragraph("3", CELL_STYLE)],
        [Paragraph("Table of Contents", CELL_LEFT), Paragraph("4", CELL_STYLE)],
        [Paragraph("List of Tables", CELL_LEFT), Paragraph("5", CELL_STYLE)],
        [Paragraph("List of Figures", CELL_LEFT), Paragraph("6", CELL_STYLE)],
        [Paragraph("<b>1. Introduction</b>", CELL_LEFT), Paragraph("<b>7</b>", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.1 Background", CELL_LEFT), Paragraph("7", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.2 Motivation", CELL_LEFT), Paragraph("7", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.3 Problem Statement", CELL_LEFT), Paragraph("8", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.4 Objectives", CELL_LEFT), Paragraph("8", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.5 Dataset Overview", CELL_LEFT), Paragraph("9", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.6 Scope and Limitations", CELL_LEFT), Paragraph("9", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.7 Organization of the Report", CELL_LEFT), Paragraph("10", CELL_STYLE)],
        [Paragraph("<b>2. Literature Review</b>", CELL_LEFT), Paragraph("<b>11</b>", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.1 Diffusion Models for Image Generation", CELL_LEFT), Paragraph("11", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.2 Controlled Image Generation", CELL_LEFT), Paragraph("11", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.3 Scene Understanding and Scene Graphs", CELL_LEFT), Paragraph("11", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.4 Crime Scene Reconstruction", CELL_LEFT), Paragraph("12", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.5 Multi-Objective Scoring", CELL_LEFT), Paragraph("12", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.6 Spatial Layout Estimation", CELL_LEFT), Paragraph("12", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.7 Evaluation Metrics for Generated Images", CELL_LEFT), Paragraph("13", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.8 Research Gap", CELL_LEFT), Paragraph("13", CELL_STYLE)],
        [Paragraph("<b>3. Methodology</b>", CELL_LEFT), Paragraph("<b>14</b>", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.1 System Overview", CELL_LEFT), Paragraph("14", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.2 NLP and Scene Graph Construction", CELL_LEFT), Paragraph("14", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.3 Spatial Layout and Depth Estimation", CELL_LEFT), Paragraph("15", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.4 Image Generation Pipeline", CELL_LEFT), Paragraph("15", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.5 Unified Scoring Function S(R)", CELL_LEFT), Paragraph("16", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.6 Energy-Based Optimization", CELL_LEFT), Paragraph("17", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.7 Closed-Loop Correction", CELL_LEFT), Paragraph("17", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.8 Implementation Details", CELL_LEFT), Paragraph("18", CELL_STYLE)],
        [Paragraph("<b>4. Results and Discussion</b>", CELL_LEFT), Paragraph("<b>19</b>", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.1 Experimental Setup", CELL_LEFT), Paragraph("19", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.2 Scoring Component Analysis", CELL_LEFT), Paragraph("19", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.3 Model Comparison", CELL_LEFT), Paragraph("20", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.4 Optimization Analysis", CELL_LEFT), Paragraph("21", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.5 Generated Reconstructions", CELL_LEFT), Paragraph("21", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.6 Error Analysis", CELL_LEFT), Paragraph("23", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.7 Discussion", CELL_LEFT), Paragraph("23", CELL_STYLE)],
        [Paragraph("<b>5. Conclusion and Future Work</b>", CELL_LEFT), Paragraph("<b>25</b>", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.1 Summary of Contributions", CELL_LEFT), Paragraph("25", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.2 Limitations", CELL_LEFT), Paragraph("26", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.3 Future Work", CELL_LEFT), Paragraph("26", CELL_STYLE)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.4 Concluding Remarks", CELL_LEFT), Paragraph("27", CELL_STYLE)],
        [Paragraph("<b>References</b>", CELL_LEFT), Paragraph("<b>28</b>", CELL_STYLE)],
        [Paragraph("<b>Appendix I -- Pipeline Output Screenshots</b>", CELL_LEFT), Paragraph("<b>29</b>", CELL_STYLE)],
    ]

    toc_table = Table(toc_data, colWidths=[5.2 * inch, 0.8 * inch], hAlign="CENTER")
    toc_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, HexColor("#CCCCCC")),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────────
    # LIST OF TABLES
    # ──────────────────────────────────────────────────────────────────────
    story.append(Paragraph("LIST OF TABLES", ss["TOCHeading"]))
    story.append(Spacer(1, 0.15 * inch))
    tables_list = [
        ("Table 1.1: Datasets Used in the Project", "9"),
        ("Table 3.1: Hyperparameter Configuration", "16"),
        ("Table 3.2: Unified Scoring Function Components", "17"),
        ("Table 3.3: Model Specifications", "15"),
        ("Table 4.1: Scoring Component Results (Bedroom Scene)", "19"),
        ("Table 4.2: Model Comparison (SD v1.4 vs Realistic Vision v5.1)", "20"),
        ("Table 4.3: Scoring Weights Configuration", "21"),
    ]
    lot_data = [[Paragraph("<b>Table</b>", CELL_LEFT), Paragraph("<b>Page</b>", CELL_STYLE)]]
    for title, pg in tables_list:
        lot_data.append([Paragraph(title, CELL_LEFT), Paragraph(pg, CELL_STYLE)])
    lot_table = Table(lot_data, colWidths=[5.2 * inch, 0.8 * inch], hAlign="CENTER")
    lot_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
    ]))
    story.append(lot_table)
    story.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────────
    # LIST OF FIGURES
    # ──────────────────────────────────────────────────────────────────────
    story.append(Paragraph("LIST OF FIGURES", ss["TOCHeading"]))
    story.append(Spacer(1, 0.15 * inch))
    figures_list = [
        ("Figure 4.1: Pass 1 Base Generation", "21"),
        ("Figure 4.2: MiDaS Depth Map", "22"),
        ("Figure 4.3: Final Reconstruction (Bedroom Scene)", "22"),
        ("Figure 4.4: Segmentation Layout Map", "22"),
        ("Figure 4.5: Composite Conditioning Image", "23"),
        ("Figure A.1: Scene Graph Visualization", "29"),
        ("Figure A.2: Depth Map Visualization", "29"),
        ("Figure A.3: Layout Preview", "30"),
    ]
    lof_data = [[Paragraph("<b>Figure</b>", CELL_LEFT), Paragraph("<b>Page</b>", CELL_STYLE)]]
    for title, pg in figures_list:
        lof_data.append([Paragraph(title, CELL_LEFT), Paragraph(pg, CELL_STYLE)])
    lof_table = Table(lof_data, colWidths=[5.2 * inch, 0.8 * inch], hAlign="CENTER")
    lof_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ("LEFTPADDING", (0, 0), (0, -1), 8),
    ]))
    story.append(lof_table)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 1 -- INTRODUCTION  (new page)
    # ══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. INTRODUCTION", ss["ChapterTitle"]))

    story.append(Paragraph("1.1 Background", ss["Section"]))
    story.append(Paragraph(
        "Crime scene reconstruction is a critical component of forensic investigation, "
        "enabling investigators to visualize and analyze the spatial arrangement of evidence, "
        "objects, and environmental conditions at a crime scene. Traditional reconstruction "
        "methods rely on physical mockups, 2D sketches, or manual 3D modeling -- all of which "
        "are time-consuming and require specialist expertise. Recent advances in generative AI, "
        "particularly diffusion-based image synthesis models, have opened new possibilities for "
        "automated visual reconstruction from textual descriptions.",
        body
    ))
    story.append(Paragraph(
        "This project leverages state-of-the-art text-to-image diffusion models (Stable Diffusion), "
        "depth-conditioned generation (ControlNet), and multi-modal evaluation (CLIP) to build a "
        "system that takes a natural language description of a crime scene and produces a "
        "photorealistic reconstruction along with comprehensive quality scores.",
        body
    ))

    story.append(Paragraph("1.2 Motivation", ss["Section"]))
    story.append(Paragraph(
        "The motivation for this project stems from several converging factors: (a) the need for "
        "rapid, low-cost crime scene visualization tools for law enforcement, (b) the maturation "
        "of text-to-image diffusion models capable of generating photo-realistic imagery, and "
        "(c) the absence of integrated systems that combine NLP understanding, spatial reasoning, "
        "and multi-objective quality evaluation for forensic scene generation.",
        body
    ))
    story.append(Paragraph(
        "Existing text-to-image systems like DALL-E, Midjourney, and Stable Diffusion generate "
        "compelling images but lack structured spatial reasoning, forensic domain adaptation, and "
        "quantitative evaluation frameworks. This project addresses these gaps by building a "
        "12-stage pipeline with a 7-component unified scoring function.",
        body
    ))

    story.append(Paragraph("1.3 Problem Statement", ss["Section"]))
    story.append(Paragraph(
        "Given a natural language description of a crime scene, the system must automatically:",
        body
    ))
    story.extend(bullet_list([
        "Parse and understand the textual description to extract objects, attributes, and spatial relationships",
        "Construct a structured scene graph representing the semantic content",
        "Generate a probabilistic spatial layout respecting physical constraints (gravity, support, containment)",
        "Produce a photorealistic image conditioned on the spatial layout via depth-controlled diffusion",
        "Evaluate the reconstruction quality using a unified multi-objective scoring function S(R)",
        "Optimize the scene configuration through energy-based methods to maximize S(R)",
    ], bullet))

    story.append(Paragraph("1.4 Objectives", ss["Section"]))
    story.append(Paragraph("The specific objectives of this project are:", body))
    objectives = [
        "To build an automated system that can generate realistic images of crime scenes from written descriptions.",
        "To develop a comprehensive scoring method that measures how accurate and realistic the generated scenes are across multiple quality dimensions.",
        "To use optimization techniques to iteratively improve the quality of the generated reconstructions.",
        "To enable the system to self-correct by identifying and fixing weaknesses in the output.",
        "To evaluate how each component of the system contributes to the overall performance.",
        "To ensure the entire system runs efficiently on affordable, consumer-grade hardware.",
    ]
    for i, obj in enumerate(objectives, 1):
        story.append(Paragraph(f"{i}. {obj}", bullet))

    story.append(Paragraph("1.5 Dataset Overview", ss["Section"]))
    story.append(Paragraph(
        "Three external datasets are used for vocabulary normalization, co-occurrence statistics, "
        "and evaluation benchmarks:",
        body
    ))
    t_data = [
        ["Dataset", "Purpose", "Size"],
        ["Visual Genome", "Object/relationship aliases, co-occurrence priors", "108,077 images, 3.8M relationships"],
        ["COCO 2017", "Object detection vocabulary, caption evaluation", "118K train + 5K val images"],
        ["CLEVR v1.0", "Spatial reasoning evaluation benchmark", "100K train + 15K val images"],
    ]
    story.append(make_table(t_data, col_widths=[1.5 * inch, 2.5 * inch, 2.5 * inch]))
    story.append(Paragraph("<i>Table 1.1: Datasets Used in the Project</i>", caption))

    story.append(Paragraph("1.6 Scope and Limitations", ss["Section"]))
    story.append(Paragraph("The scope of this project includes:", body))
    story.extend(bullet_list([
        "Reconstruction of indoor crime scenes (bedroom, kitchen, living room, bathroom, hallway, garage, basement, office) and selected outdoor scenes (alley, parking lot, street, warehouse)",
        "Single-image and multi-view generation at 512 x 512 resolution",
        "Quantitative evaluation via the 7-component unified score S(R)",
    ], bullet))
    story.append(Paragraph("Limitations include:", body))
    story.extend(bullet_list([
        "GPU VRAM (6 GB) constrains resolution to 512 x 512 and precludes SDXL models",
        "CLIP ViT-B/32 provides limited discriminative power for AI-generated image detection",
        "The system generates static 2D images, not full 3D scene reconstructions",
        "Evaluation is automated; no human evaluator study was conducted",
    ], bullet))

    story.append(Paragraph("1.7 Organization of the Report", ss["Section"]))
    story.append(Paragraph(
        "The remainder of this report is organized as follows: Chapter 2 presents a literature "
        "review of diffusion models, controlled generation, scene graphs, and multi-objective "
        "scoring. Chapter 3 details the methodology, including the 12-stage pipeline, unified "
        "scoring function, optimization, and correction mechanisms. Chapter 4 presents "
        "experimental results with scoring breakdowns, model comparisons, and generated "
        "reconstructions. Chapter 5 concludes the report with contributions, limitations, and "
        "directions for future work.",
        body
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 2 -- LITERATURE REVIEW  (new page)
    # ══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. LITERATURE REVIEW", ss["ChapterTitle"]))
    story.append(Paragraph(
        "This chapter reviews the key research areas that underpin the crime scene "
        "reconstruction system: diffusion-based image generation, controlled synthesis, "
        "scene understanding, and evaluation methodology.",
        body
    ))

    story.append(Paragraph("2.1 Diffusion Models for Image Generation", ss["Section"]))
    story.append(Paragraph(
        "Denoising Diffusion Probabilistic Models (DDPMs), introduced by Ho et al. (2020) [1], "
        "learn to reverse a gradual noising process to generate images from Gaussian noise. "
        "Rombach et al. (2022) [2] proposed Latent Diffusion Models (LDMs), which operate in "
        "a compressed latent space, dramatically reducing computational cost while maintaining "
        "image quality. Stable Diffusion, an open-source LDM, forms the backbone of modern "
        "text-to-image generation systems. This project uses Realistic Vision v5.1, a community "
        "fine-tune of Stable Diffusion v1.5 optimized for photorealistic output [3].",
        body
    ))

    story.append(Paragraph("2.2 Controlled Image Generation", ss["Section"]))
    story.append(Paragraph(
        "ControlNet, proposed by Zhang and Agrawala (2023) [4], adds spatial conditioning to "
        "pre-trained diffusion models by injecting control signals (depth maps, edge maps, "
        "segmentation maps) through trainable copy-branches of the encoder. This enables precise "
        "control over the spatial structure of generated images while preserving the generative "
        "quality of the base model. In this project, depth-conditioned ControlNet guides the "
        "reconstruction layout using MiDaS depth estimation [5].",
        body
    ))

    story.append(Paragraph("2.3 Scene Understanding and Scene Graphs", ss["Section"]))
    story.append(Paragraph(
        "Scene graphs represent images as directed graphs where nodes are objects and edges "
        "are relationships. Johnson et al. (2015) [6] introduced scene graph generation for "
        "image retrieval, while Krishna et al. (2017) [7] created the Visual Genome dataset "
        "containing dense scene graph annotations. In the text-to-scene direction, scene graphs "
        "provide a structured intermediate representation that bridges natural language and "
        "spatial layout. This project constructs scene graphs from parsed text using spaCy NLP "
        "and normalizes vocabulary against Visual Genome aliases.",
        body
    ))

    story.append(Paragraph("2.4 Crime Scene Reconstruction", ss["Section"]))
    story.append(Paragraph(
        "Traditional crime scene reconstruction relies on laser scanning, photogrammetry, and "
        "manual 3D modeling tools (e.g., SketchUp, Blender). Buck et al. (2013) [8] surveyed "
        "virtual crime scene reconstruction techniques. Recent work by forensic AI researchers "
        "has explored automated generation from evidence logs, but no prior system combines "
        "NLP parsing, probabilistic spatial reasoning, controlled diffusion generation, and "
        "multi-objective scoring in a unified pipeline -- which is the contribution of this project.",
        body
    ))

    story.append(Paragraph("2.5 Multi-Objective Scoring", ss["Section"]))
    story.append(Paragraph(
        "Evaluating generated images is a multi-faceted problem. Frechet Inception Distance (FID) "
        "[9] measures distributional similarity but requires large sample sets. CLIP-based "
        "similarity [10] evaluates text-image alignment at the semantic level. This project "
        "introduces a 7-component unified scoring function S(R) that combines semantic alignment "
        "(CLIP), spatial consistency (constraint satisfaction), physical plausibility (gravity, "
        "support), visual realism (aesthetic + noise + sharpness), probabilistic prior (Visual "
        "Genome co-occurrence), multi-view consistency, and perceptual believability (CLIP probe).",
        body
    ))

    story.append(Paragraph("2.6 Spatial Layout Estimation", ss["Section"]))
    story.append(Paragraph(
        "Estimating 2D/3D spatial layouts from scene descriptions involves placing objects "
        "according to spatial relationships while respecting physical constraints. Prior work "
        "includes Neural Turtle Graphics (Liao et al., 2019) [11] for indoor layout generation "
        "and LayoutTransformer (Yang et al., 2021) [12] for autoregressive layout prediction. "
        "This project uses a probabilistic sampling approach with room-type-specific bounding "
        "boxes, gravity-aware vertical placement, and depth ordering based on scene graph "
        "relationships.",
        body
    ))

    story.append(Paragraph("2.7 Evaluation Metrics for Generated Images", ss["Section"]))
    story.append(Paragraph(
        "Standard metrics include FID, IS (Inception Score), CLIP score, and human evaluation. "
        "Hessel et al. (2021) [13] proposed CLIPScore for reference-free image-text evaluation. "
        "This project extends evaluation beyond single-metric approaches by combining seven "
        "orthogonal quality dimensions into a weighted unified score, enabling fine-grained "
        "diagnosis of reconstruction failures.",
        body
    ))

    story.append(Paragraph("2.8 Research Gap", ss["Section"]))
    story.append(Paragraph(
        "Despite advances in text-to-image generation, no existing system provides: (a) an "
        "end-to-end pipeline from natural language to forensic scene reconstruction, (b) a "
        "multi-objective scoring framework that jointly assesses semantic, spatial, physical, "
        "visual, probabilistic, multi-view, and perceptual quality, (c) energy-based optimization "
        "over scene configurations, and (d) closed-loop correction for iterative improvement. "
        "This project addresses all four gaps in a single integrated framework.",
        body
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 3 -- METHODOLOGY  (new page)
    # ══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. METHODOLOGY", ss["ChapterTitle"]))
    story.append(Paragraph(
        "This chapter details the methodology employed in developing the crime scene "
        "reconstruction system. It covers the 12-stage pipeline architecture, the unified "
        "scoring function, optimization strategies, and implementation details.",
        body
    ))

    story.append(Paragraph("3.1 System Overview", ss["Section"]))
    story.append(Paragraph(
        "The system consists of a 12-stage pipeline that progressively transforms a natural "
        "language crime scene description into a photorealistic image reconstruction. The stages "
        "are organized into four phases:",
        body
    ))
    story.extend(bullet_list([
        "<b>Understanding Phase (Stages 1-3):</b> NLP parsing, vocabulary normalization against Visual Genome, and scene graph construction",
        "<b>Planning Phase (Stages 4-6):</b> Multi-hypothesis generation, probabilistic spatial layout, and depth map computation",
        "<b>Generation Phase (Stages 7-8):</b> Two-pass ControlNet + Stable Diffusion generation and multi-view rendering",
        "<b>Evaluation Phase (Stages 9-11):</b> Explainability reporting, unified scoring, and result packaging",
    ], bullet))

    story.append(Paragraph("3.2 NLP and Scene Graph Construction", ss["Section"]))
    story.append(Paragraph(
        "Stage 1 uses spaCy (en_core_web_sm) to parse the input text, extracting noun chunks "
        "as candidate objects and dependency relations as spatial relationships. Stage 2 "
        "normalizes the extracted vocabulary against Visual Genome object and relationship alias "
        "files, mapping synonyms to canonical forms (e.g., 'sofa' to 'couch'). Stage 3 constructs "
        "a directed scene graph using NetworkX, where nodes represent objects with attributes "
        "(position, size, depth) and edges represent spatial relationships (on, near, beside).",
        body
    ))

    story.append(Paragraph("3.3 Spatial Layout and Depth Estimation", ss["Section"]))
    story.append(Paragraph(
        "Stage 4 generates multiple hypothesis configurations by sampling object positions from "
        "room-type-specific prior distributions. Each hypothesis assigns 2D bounding boxes and "
        "depth values to every object. Stage 5 applies physical constraints (gravity alignment, "
        "support relations, non-interpenetration) to refine the layout. Stage 6 generates a "
        "depth map using either room-aware synthetic depth (for controllable layouts) or MiDaS "
        "DPT-Hybrid estimation (for pass-2 refinement from generated images).",
        body
    ))
    story.append(Paragraph("3.3.1 Two-Pass Generation", ss["SubSection"]))
    story.append(Paragraph(
        "The system employs a two-pass generation strategy. Pass 1 generates a base image using "
        "the prompt alone (standard Stable Diffusion). Pass 2 estimates the MiDaS depth map from "
        "the Pass 1 output and feeds it to ControlNet for a depth-conditioned refinement. This "
        "yields images with significantly better spatial coherence.",
        body
    ))

    story.append(Paragraph("3.4 Image Generation Pipeline", ss["Section"]))
    story.append(Paragraph(
        "Image generation uses Realistic Vision v5.1 (SG161222/Realistic_Vision_V5.1_noVAE), "
        "a photorealistic fine-tune of Stable Diffusion v1.5, paired with sd-vae-ft-mse from "
        "Stability AI for improved decoding sharpness. ControlNet depth conditioning uses "
        "lllyasviel/sd-controlnet-depth. All models run in FP16 with sequential CPU offloading "
        "to accommodate the 6 GB VRAM constraint.",
        body
    ))

    model_data = [
        ["Component", "Model", "Parameters"],
        ["Base Generator", "Realistic Vision v5.1", "~860M (UNet)"],
        ["VAE Decoder", "sd-vae-ft-mse", "~83M"],
        ["ControlNet", "sd-controlnet-depth", "~361M"],
        ["Depth Estimator", "MiDaS DPT-Hybrid", "~123M"],
        ["CLIP Evaluator", "ViT-B/32 (OpenAI)", "~151M"],
        ["NLP Parser", "spaCy en_core_web_sm", "~12M"],
    ]
    story.append(make_table(model_data, col_widths=[1.5 * inch, 2.5 * inch, 1.5 * inch]))
    story.append(Paragraph("<i>Table 3.3: Model Specifications</i>", caption))

    story.append(Paragraph("3.4.1 Generation Hyperparameters", ss["SubSection"]))
    hp_data = [
        ["Hyperparameter", "Value"],
        ["Inference Steps", "35"],
        ["Guidance Scale (CFG)", "9.0"],
        ["Resolution", "512 x 512"],
        ["Precision", "FP16"],
        ["ControlNet Conditioning Scale", "0.8"],
        ["Seed", "42"],
        ["Number of Hypotheses", "3"],
        ["CPU Offload", "Sequential"],
    ]
    story.append(make_table(hp_data, col_widths=[3.0 * inch, 2.5 * inch]))
    story.append(Paragraph("<i>Table 3.1: Hyperparameter Configuration</i>", caption))

    story.append(Paragraph("3.5 Unified Scoring Function S(R)", ss["Section"]))
    story.append(Paragraph(
        "The unified scoring function evaluates a reconstruction R as a weighted sum of seven "
        "orthogonal quality components:",
        body
    ))
    story.append(Paragraph(
        "<b>S(R) = w1 x Semantic + w2 x Spatial + w3 x Physical + w4 x Visual + "
        "w5 x Probabilistic + w6 x Multiview + w7 x Human</b>",
        ParagraphStyle("Formula", fontName="Times-Bold", fontSize=11, leading=16,
                        alignment=TA_CENTER, spaceBefore=8, spaceAfter=8)
    ))

    scoring_data = [
        ["Component", "Weight", "Method"],
        ["Semantic Alignment", "0.20", "CLIP similarity + object recall + relationship satisfaction"],
        ["Spatial Consistency", "0.15", "Constraint violations, position/depth ordering errors"],
        ["Physical Plausibility", "0.10", "Gravity alignment, support relations, scale realism"],
        ["Visual Realism", "0.15", "Aesthetic CLIP score, noise residual, sharpness"],
        ["Probabilistic Prior", "0.10", "Visual Genome co-occurrence + spatial relation log-likelihood"],
        ["Multi-View Consistency", "0.10", "CLIP similarity across depth-perturbed views"],
        ["Perceptual Believability", "0.20", "CLIP realism probe, scene coherence, uncanny penalty"],
    ]
    story.append(make_table(scoring_data, col_widths=[1.8 * inch, 0.8 * inch, 3.8 * inch]))
    story.append(Paragraph("<i>Table 3.2: Unified Scoring Function Components</i>", caption))

    story.append(Paragraph("3.6 Energy-Based Optimization", ss["Section"]))
    story.append(Paragraph(
        "The system defines an energy function E(c) = -S(R(c)) over the scene configuration c "
        "and minimizes it using two complementary optimizers:",
        body
    ))
    story.extend(bullet_list([
        "<b>Simulated Annealing (SA):</b> Perturbs object positions and recomputes S(R); accepts worse configurations with decreasing probability as temperature decays from 1.0 to 0.01 over 100 iterations",
        "<b>Evolutionary Strategies (ES):</b> Maintains a population of configurations, applies Gaussian mutations, and selects top-k survivors per generation (population=20, generations=50)",
        "<b>Hybrid (default):</b> Runs SA first to find a good basin, then refines with ES around the SA solution",
    ], bullet))

    story.append(Paragraph("3.7 Closed-Loop Correction", ss["Section"]))
    story.append(Paragraph(
        "The closed-loop correction module iteratively re-generates the image, re-scores it, "
        "and applies targeted fixes based on the weakest scoring component. For example, if "
        "visual realism is the lowest score, the system increases guidance scale and adds "
        "negative prompt terms. If spatial consistency is low, it adjusts object positions in "
        "the layout. The loop runs for a configurable number of iterations (default: 3) or "
        "until the score improvement falls below a threshold.",
        body
    ))

    story.append(Paragraph("3.8 Implementation Details", ss["Section"]))
    story.append(Paragraph(
        "The system is implemented in Python 3.10 using PyTorch 2.7.1 with CUDA 11.8. Key "
        "libraries include Hugging Face Diffusers (pipeline management), Transformers (model "
        "loading), open_clip (CLIP evaluation), spaCy (NLP), NetworkX (scene graphs), and "
        "Pillow (image processing). All experiments were conducted on an NVIDIA RTX 3060 "
        "Laptop GPU with 6 GB VRAM. Memory optimizations include attention slicing, VAE "
        "slicing, and sequential CPU offloading (model weights stream from system RAM to GPU "
        "layer-by-layer during inference).",
        body
    ))
    story.append(Paragraph(
        "The codebase is organized into modular packages: src/stages/ (12 pipeline stages), "
        "src/scoring/ (7 scoring components + unified scorer), src/optimization/ (SA + ES + "
        "weight calibration), src/correction/ (closed-loop), src/conditioning/ (segmentation "
        "layout), src/experiments/ (runner, ablation, research logger), and src/utils/ (config, "
        "memory management, logging). Total: 45+ Python source files.",
        body
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 4 -- RESULTS AND DISCUSSION  (new page)
    # ══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. RESULTS AND DISCUSSION", ss["ChapterTitle"]))
    story.append(Paragraph(
        "This chapter presents experimental results from running the pipeline on crime scene "
        "descriptions, analyzing scoring components, comparing model versions, and discussing "
        "the generated reconstructions.",
        body
    ))

    story.append(Paragraph("4.1 Experimental Setup", ss["Section"]))
    story.append(Paragraph(
        "Experiments were conducted on two crime scene descriptions:", body
    ))
    story.extend(bullet_list([
        '<b>Bedroom scene:</b> "A dimly lit bedroom with a broken window, a bloodstained mattress on the floor, and a knife near the doorway"',
        '<b>Kitchen scene:</b> "Small kitchen with broken glass on the floor, an overturned chair, and a bloodstain on the counter"',
    ], bullet))
    story.append(Paragraph(
        "Each scene was processed through the full 12-stage pipeline with 3 hypothesis "
        "configurations, 35 inference steps, CFG scale 9.0, and seed 42. The best hypothesis "
        "(highest S(R)) was selected for analysis.",
        body
    ))

    # 4.2 Scoring Breakdown -- Table 4.1 with Paragraph-wrapped cells
    story.append(Paragraph("4.2 Scoring Component Analysis", ss["Section"]))
    story.append(Paragraph(
        "Table 4.1 presents the detailed scoring breakdown for the bedroom scene reconstruction "
        "(best hypothesis, after scoring recalibration).",
        body
    ))

    score_data = [
        [Paragraph("Component", CELL_HEADER),
         Paragraph("Score", CELL_HEADER),
         Paragraph("Sub-metrics", CELL_HEADER)],
        [Paragraph("Semantic Alignment", CELL_STYLE),
         Paragraph("0.783", CELL_STYLE),
         Paragraph("CLIP: 0.633, Object Recall: 1.000, Rel. Satisfaction: 0.833", CELL_LEFT)],
        [Paragraph("Spatial Consistency", CELL_STYLE),
         Paragraph("0.999", CELL_STYLE),
         Paragraph("Violations: 0, Position Errors: 0, Depth Errors: 0", CELL_LEFT)],
        [Paragraph("Physical Plausibility", CELL_STYLE),
         Paragraph("0.923", CELL_STYLE),
         Paragraph("Gravity: 1.000, Support: 1.000, Floor: 0.760, Scale: 0.932", CELL_LEFT)],
        [Paragraph("Visual Realism", CELL_STYLE),
         Paragraph("0.515", CELL_STYLE),
         Paragraph("Aesthetic: 0.648, Noise: 0.381, Sharpness: 0.472", CELL_LEFT)],
        [Paragraph("Probabilistic Prior", CELL_STYLE),
         Paragraph("0.816", CELL_STYLE),
         Paragraph("Spatial LL: 3.964, Co-occ LL: -6.000, Mean LL: -0.509", CELL_LEFT)],
        [Paragraph("Multi-View Consistency", CELL_STYLE),
         Paragraph("0.500", CELL_STYLE),
         Paragraph("(Skipped -- insufficient views)", CELL_LEFT)],
        [Paragraph("Perceptual Believability", CELL_STYLE),
         Paragraph("0.654", CELL_STYLE),
         Paragraph("Realism Probe: 0.519, Coherence: 0.630, Uncanny: 0.000", CELL_LEFT)],
        [Paragraph("<b>Unified S(R)</b>", CELL_BOLD),
         Paragraph("<b>0.739</b>", CELL_BOLD),
         Paragraph("<b>Weighted sum with calibrated weights</b>",
                   ParagraphStyle("CellBoldLeft", parent=CELL_BOLD, alignment=TA_LEFT))],
    ]
    score_table = Table(score_data, colWidths=[1.8 * inch, 0.8 * inch, 3.8 * inch], hAlign="CENTER")
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, TABLE_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(score_table)
    story.append(Paragraph("<i>Table 4.1: Scoring Component Results (Bedroom Scene)</i>", caption))

    story.append(Paragraph(
        "Spatial consistency achieved near-perfect scores (0.999) indicating that the probabilistic "
        "layout engine correctly satisfies scene graph constraints. Physical plausibility scored "
        "0.923, with minor floor-contact penalties. Visual realism (0.515) and perceptual "
        "believability (0.654) are limited by the inherent characteristics of diffusion-generated "
        "images at 512 x 512 resolution on a ViT-B/32 CLIP model.",
        body
    ))

    story.append(Paragraph("4.3 Model Comparison", ss["Section"]))
    story.append(Paragraph(
        "Table 4.2 compares the two base models evaluated during development:",
        body
    ))
    model_comp = [
        ["Metric", "SD v1.4", "Realistic Vision v5.1"],
        ["Visual Realism", "0.402", "0.515"],
        ["Perceptual Believability", "0.645", "0.654"],
        ["Probabilistic Prior", "0.421", "0.816"],
        ["Overall S(R)", "0.676", "0.739"],
        ["Pipeline Time", "235s", "217s"],
    ]
    story.append(make_table(model_comp, col_widths=[2.0 * inch, 2.0 * inch, 2.5 * inch]))
    story.append(Paragraph("<i>Table 4.2: Model Comparison (SD v1.4 vs Realistic Vision v5.1)</i>", caption))
    story.append(Paragraph(
        "Realistic Vision v5.1 improved visual realism by +0.113 and overall S(R) by +0.063. "
        "The probabilistic prior improvement (+0.395) is primarily due to scoring recalibration "
        "(Laplace smoothing and co-occurrence log-likelihood clamping) applied alongside the "
        "model upgrade.",
        body
    ))

    # Table 4.3 -- Weights (FIXED: plain text w1..w7 instead of unicode)
    story.append(Paragraph("4.3.1 Scoring Weights", ss["SubSection"]))
    weights_data = [
        ["Component", "Weight"],
        ["Semantic (w1)", "0.20"],
        ["Spatial (w2)", "0.15"],
        ["Physical (w3)", "0.10"],
        ["Visual (w4)", "0.15"],
        ["Probabilistic (w5)", "0.10"],
        ["Multi-View (w6)", "0.10"],
        ["Human/Perceptual (w7)", "0.20"],
    ]
    story.append(make_table(weights_data, col_widths=[2.5 * inch, 2.0 * inch]))
    story.append(Paragraph("<i>Table 4.3: Scoring Weights Configuration</i>", caption))

    story.append(Paragraph("4.4 Optimization Analysis", ss["Section"]))
    story.append(Paragraph(
        "Simulated Annealing optimization ran for 101 iterations in 0.45 seconds, achieving a "
        "best layout energy of -0.324 (layout-only score without image generation). The "
        "optimization converged early (iteration 0), indicating that the initial hypothesis "
        "from the probabilistic sampler was already near-optimal for the bedroom scene. This "
        "validates the effectiveness of the room-type-specific prior distributions used in "
        "hypothesis generation.",
        body
    ))

    story.append(Paragraph("4.5 Generated Reconstructions", ss["Section"]))
    story.append(Paragraph(
        "The following figures show key outputs from the bedroom scene reconstruction pipeline:",
        body
    ))

    story.append(safe_image(IMAGES_DIR / "pass1_base.png"))
    story.append(Paragraph("<i>Figure 4.1: Pass 1 -- Base generation from text prompt (Realistic Vision v5.1)</i>", caption))

    story.append(safe_image(DEPTH_DIR / "midas_depth_h1.png"))
    story.append(Paragraph("<i>Figure 4.2: MiDaS DPT-Hybrid depth map estimated from Pass 1 output</i>", caption))

    story.append(safe_image(IMAGES_DIR / "reconstruction_h1.png"))
    story.append(Paragraph("<i>Figure 4.3: Final reconstruction -- Pass 2 depth-conditioned generation (best hypothesis)</i>", caption))

    story.append(safe_image(IMAGES_DIR / "segmentation_map.png"))
    story.append(Paragraph("<i>Figure 4.4: Segmentation layout map showing object regions</i>", caption))

    story.append(safe_image(IMAGES_DIR / "composite_conditioning.png"))
    story.append(Paragraph("<i>Figure 4.5: Composite conditioning image fed to ControlNet</i>", caption))

    story.append(Paragraph("4.6 Error Analysis", ss["Section"]))
    story.append(Paragraph("Analysis of scoring deficiencies reveals several patterns:", body))
    story.extend(bullet_list([
        "<b>Visual Realism (0.515):</b> Diffusion-generated images at 512 x 512 exhibit subtle artifacts (texture repetition, soft edges) that suppress sharpness and noise scores. Higher resolution (768+) or SDXL would likely improve this.",
        "<b>Perceptual Believability (0.654):</b> CLIP ViT-B/32 has limited ability to distinguish real photographs from AI-generated images. A dedicated AI-detection model or larger CLIP (ViT-L/14) would provide stronger signal.",
        "<b>Multi-View (0.500):</b> Multi-view generation was skipped in the latest run due to time constraints; when enabled, it provides additional consistency signal.",
        "<b>Semantic CLIP (0.633):</b> Crime scene descriptions include forensic-specific terms ('bloodstained', 'dimly lit') that have weak CLIP embeddings. Domain adaptation of CLIP could improve alignment.",
    ], bullet))

    story.append(Paragraph("4.7 Discussion", ss["Section"]))
    story.append(Paragraph(
        "The experimental results validate the effectiveness of the unified multi-objective "
        "approach. Several key observations emerge:", body
    ))
    story.append(Paragraph(
        "First, the 12-stage pipeline successfully transforms natural language into structured "
        "reconstructions. The NLP to scene graph to layout to depth to image chain produces "
        "spatially coherent results, as evidenced by the 0.999 spatial consistency score.",
        body
    ))
    story.append(Paragraph(
        "Second, the unified scoring function S(R) provides fine-grained diagnostics. Unlike "
        "single metrics (FID, CLIP score alone), the 7-component decomposition pinpoints exactly "
        "which quality dimension is deficient, enabling targeted improvement.",
        body
    ))
    story.append(Paragraph(
        "Third, the model upgrade from SD v1.4 to Realistic Vision v5.1 with a tuned VAE "
        "decoder produced measurable improvements across visual metrics, confirming that "
        "domain-specific fine-tuning is beneficial for forensic scene generation.",
        body
    ))
    story.append(Paragraph(
        "Fourth, operating the entire system on a 6 GB consumer GPU demonstrates that "
        "memory-efficient techniques (sequential CPU offload, attention/VAE slicing) make "
        "research-grade generative pipelines accessible without expensive hardware.",
        body
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # CHAPTER 5 -- CONCLUSION AND FUTURE WORK  (new page)
    # ══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("5. CONCLUSION AND FUTURE WORK", ss["ChapterTitle"]))

    story.append(Paragraph("5.1 Summary of Contributions", ss["Section"]))
    story.append(Paragraph(
        "This project successfully developed a research-grade unified multi-objective crime "
        "scene reconstruction system. The key contributions are:", body
    ))
    contributions = [
        "A 12-stage pipeline that transforms natural language crime scene descriptions into photorealistic image reconstructions, integrating NLP parsing, scene graph construction, probabilistic spatial layout, and depth-conditioned diffusion generation.",
        "A novel 7-component unified scoring function S(R) that evaluates semantic alignment, spatial consistency, physical plausibility, visual realism, probabilistic prior, multi-view consistency, and perceptual believability -- enabling fine-grained quality diagnosis.",
        "An energy-based optimization framework (Simulated Annealing + Evolutionary Strategies) that maximizes S(R) over scene configurations.",
        "A closed-loop correction mechanism for iterative score improvement based on weakest-component diagnosis.",
        "An ablation study framework for systematic evaluation of component contributions.",
        "Demonstration that the entire system operates on a consumer-grade NVIDIA RTX 3060 (6 GB VRAM) using sequential CPU offloading, achieving S(R) = 0.739 on realistic crime scene descriptions.",
    ]
    for i, c in enumerate(contributions, 1):
        story.append(Paragraph(f"{i}. {c}", bullet))

    story.append(Paragraph("5.2 Limitations", ss["Section"]))
    story.append(Paragraph("The following limitations are acknowledged:", body))
    story.extend(bullet_list([
        "GPU VRAM (6 GB) limits resolution to 512 x 512 and precludes using SDXL or larger CLIP models (ViT-L/14)",
        "Visual realism scoring is bounded by CLIP ViT-B/32's limited discriminative power for AI-generated images",
        "No human evaluator study was conducted; all evaluation is automated",
        "The system generates static 2D images, not interactive 3D scenes",
        "Crime scene vocabulary is underrepresented in CLIP's training data, limiting semantic alignment",
        "Sequential CPU offloading increases inference time (~2.6s/step vs ~0.8s/step with full GPU loading)",
    ], bullet))

    story.append(Paragraph("5.3 Future Work", ss["Section"]))

    story.append(Paragraph("5.3.1 Higher-Resolution Generation", ss["SubSection"]))
    story.append(Paragraph(
        "Upgrading to SDXL (1024 x 1024) on a GPU with 10+ GB VRAM would substantially improve "
        "visual realism and sharpness scores. The pipeline architecture is model-agnostic and "
        "would require only configuration changes.",
        body
    ))

    story.append(Paragraph("5.3.2 Stronger CLIP Evaluation", ss["SubSection"]))
    story.append(Paragraph(
        "Replacing ViT-B/32 with ViT-L/14 (requires ~1.5 GB VRAM) would improve semantic "
        "alignment measurement and perceptual believability scoring. Alternatively, a dedicated "
        "AI-generated image detector could replace the CLIP-probe approach.",
        body
    ))

    story.append(Paragraph("5.3.3 3D Scene Reconstruction", ss["SubSection"]))
    story.append(Paragraph(
        "Extending the system to generate 3D scene representations (NeRF, Gaussian Splatting) "
        "from multi-view outputs would enable interactive exploration of reconstructed scenes.",
        body
    ))

    story.append(Paragraph("5.3.4 Human Evaluation Study", ss["SubSection"]))
    story.append(Paragraph(
        "Conducting a formal human evaluation with forensic experts would validate the "
        "perceptual quality scores and provide ground truth for calibrating the unified "
        "scoring weights.",
        body
    ))

    story.append(Paragraph("5.3.5 Domain-Adapted CLIP", ss["SubSection"]))
    story.append(Paragraph(
        "Fine-tuning CLIP on forensic imagery and crime scene descriptions would improve "
        "semantic alignment for domain-specific terminology (bloodstain patterns, evidence "
        "markers, trajectory analysis).",
        body
    ))

    story.append(Paragraph("5.3.6 Real-Time Deployment", ss["SubSection"]))
    story.append(Paragraph(
        "Optimizing the pipeline for real-time use through model distillation (e.g., LCM-LoRA "
        "for 4-step inference), TensorRT compilation, and a web-based interface would enable "
        "practical deployment in forensic investigation workflows.",
        body
    ))

    story.append(Paragraph("5.4 Concluding Remarks", ss["Section"]))
    story.append(Paragraph(
        "This project has demonstrated the feasibility of automated crime scene reconstruction "
        "from natural language descriptions using a unified multi-objective generative AI "
        "framework. The 7-component scoring function provides unprecedented diagnostic "
        "granularity for generated scene quality, while the 12-stage pipeline offers a "
        "modular, extensible architecture. The work contributes to the intersection of "
        "generative AI, forensic science, and multi-objective optimization, and establishes "
        "a foundation for future research in AI-assisted criminal investigation.",
        body
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # REFERENCES  (new page)
    # ══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("REFERENCES", ss["ChapterTitle"]))
    story.append(Spacer(1, 0.1 * inch))
    references = [
        "[1] Ho, J., Jain, A., &amp; Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, 6840-6851.",
        "[2] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., &amp; Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF CVPR (pp. 10684-10695).",
        "[3] SG161222. (2023). Realistic Vision V5.1. Hugging Face Model Hub.",
        "[4] Zhang, L., &amp; Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF ICCV (pp. 3836-3847).",
        "[5] Ranftl, R., Bochkovskiy, A., &amp; Koltun, V. (2021). Vision transformers for dense prediction. In Proceedings of the IEEE/CVF ICCV (pp. 12179-12188).",
        "[6] Johnson, J., Krishna, R., Stark, M., et al. (2015). Image retrieval using scene graphs. In Proceedings of the IEEE CVPR (pp. 3668-3678).",
        "[7] Krishna, R., Zhu, Y., Groth, O., et al. (2017). Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 123(1), 32-73.",
        "[8] Buck, U., Naether, S., Rass, B., Jackowski, C., &amp; Thali, M. J. (2013). Accident or homicide -- virtual crime scene reconstruction using 3D methods. Forensic Science International, 225(1-3), 75-84.",
        "[9] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., &amp; Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. NeurIPS, 30.",
        "[10] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. In ICML (pp. 8748-8763). PMLR.",
        "[11] Liao, Y. H., Kar, A., &amp; Fidler, S. (2019). Object completion using neural turtle graphics. NeurIPS.",
        "[12] Yang, J., Zeng, A., Li, T., &amp; Zhao, T. (2021). LayoutTransformer: Layout generation and completion with self-attention. In IEEE/CVF ICCV (pp. 1004-1014).",
        "[13] Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., &amp; Choi, Y. (2021). CLIPScore: A reference-free evaluation metric for image captioning. In EMNLP (pp. 7514-7528).",
    ]
    for ref in references:
        story.append(Paragraph(ref, ParagraphStyle(
            "Ref", fontName="Times-Roman", fontSize=11, leading=15,
            alignment=TA_JUSTIFY, spaceAfter=6, leftIndent=36, firstLineIndent=-36,
        )))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # APPENDIX I -- Screenshots  (new page)
    # ══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("APPENDIX I -- Pipeline Output Screenshots", ss["ChapterTitle"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Scene Graph Visualization", ss["Section"]))
    story.append(safe_image(GRAPH_DIR / "scene_graph.png"))
    story.append(Paragraph("<i>Figure A.1: Scene graph constructed from the bedroom crime scene description</i>", caption))

    story.append(Paragraph("Room-Aware Depth Map", ss["Section"]))
    story.append(safe_image(DEPTH_DIR / "depth_map_h1.png"))
    story.append(Paragraph("<i>Figure A.2: Synthetic room-aware depth map for hypothesis 1</i>", caption))

    story.append(Paragraph("Layout Preview", ss["Section"]))
    story.append(safe_image(IMAGES_DIR / "layout_preview.png"))
    story.append(Paragraph("<i>Figure A.3: 2D spatial layout preview with object bounding boxes</i>", caption))

    return story


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print(f"Generating report: {OUTPUT_PDF}")

    ss = build_styles()
    story = build_story(ss)

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        title="Crime Scene Reconstruction - Project Report",
        author="Pranav",
    )

    doc.build(story,
              onFirstPage=add_page_border_and_number,
              onLaterPages=add_page_border_and_number)
    print(f"Report generated successfully: {OUTPUT_PDF}")
    print(f"File size: {OUTPUT_PDF.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
