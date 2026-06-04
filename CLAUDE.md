# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an **OMR (Optical Mark Recognition) system** for automated answer sheet scanning and recognition. It uses Python + OpenCV + Streamlit to identify multiple choice answers, detect barcodes (student IDs), and crop subjective question regions from scanned answer sheets.

## Commands

```bash
# Install dependencies
cd omr_demo
pip install -r requirements.txt

# Run the Streamlit application
python -m streamlit run app.py
```

## Architecture

### Core Processing (`omr_demo/core/processor.py`)

The `CardProcessor` class is the central engine:

1. **Template System** — JSON templates define bubble positions, barcode regions, and subjective areas for specific answer sheet formats. Templates use reference dimensions (`ref_w`, `ref_h`) and coordinate scaling to adapt to different scan resolutions.

2. **Recognition Methods**:
   - **Differential method** — Compares filled answer sheet against blank reference to detect marks (more accurate)
   - **Fixed threshold method** — Direct binarization and pixel ratio analysis (no blank reference needed)

3. **A/B Page Handling** — Answer sheets have two sides (A and B). Files are paired by naming convention (`xxx01A.jpg`, `xxx01B.jpg`). Processing rules:
   - Template barcode detection on A page
   - OMR bubble recognition on A page
   - Subjective area cropping on both pages

4. **Manual Region Overrides** — User-defined regions can replace template behavior:
   - `选择题` (Multiple Choice) — Region-specific bubble recognition
   - `个人信息` (Personal Info) — Region-specific barcode detection
   - `非选择题` (Non-Choice) — Pure cropping without recognition

### Streamlit UI (`omr_demo/app.py`)

Three main tabs:
- **Tab 1 (模板与参考)** — Load templates, set blank references, configure manual regions, define paper layouts with grid-based bubble generation
- **Tab 2 (批量处理)** — Batch process A/B paired answer sheets
- **Tab 3 (结果核对与导出)** — Review results, apply manual corrections, export Excel score reports

### Paper Layouts (半自动网格)

A newer feature for new paper formats without pre-defined bubbles. Users draw column rectangles on a sample image, specify row/column counts and thresholds, and the system auto-generates a grid of bubble detection points.

## Key Implementation Details

### Coordinate Scaling

Coordinates in templates use reference dimensions. The `scale_coords()` method scales them to actual image sizes:
```python
sx = int(x * img_w / self.ref_w)
sy = int(y * img_h / self.ref_h)
```

Manual regions also scale from their `ref_size` (the dimensions of the image used during calibration).

### Windows Chinese Path Handling

`cv2.imwrite()` has UTF-8 encoding issues on Windows. The codebase uses `cv2.imencode + open(filepath, 'wb')` instead for saving cropped images.

### OMR Threshold

Default `threshold=0.15` (15% dark pixel ratio). Adjust based on:
- Too many missed detections → lower threshold (0.05–0.10)
- Too many false positives → raise threshold (0.20–0.30)

### Template Structure

```json
{
  "name": "template_name",
  "image_size": {"w": 1237, "h": 1741},
  "pages": {
    "A": {
      "barcode": {"x": 730, "y": 220, "w": 280, "h": 80},
      "bubbles": [{"q": 1, "opt": "A", "x": 248, "y": 517, "w": 12, "h": 12}, ...],
      "subjective": {"42": {"x1": 120, "y1": 1100, "x2": 1120, "y2": 1200, "score": 2}, ...}
    },
    "B": { ... }
  }
}
```

## File Pairing Convention

Batch processing expects filenames ending with `A.jpg` (or `A.jpeg`, `A.png`) and `B.jpg` for the two sides of the same answer sheet. The base name is extracted by removing the last character before the extension.
