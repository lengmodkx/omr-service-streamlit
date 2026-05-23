# 答题卡智能处理系统 (OMR Demo)

## Project Overview

This project is a Python-based Optical Mark Recognition (OMR) demo system designed for processing English exam answer cards (答题卡). It provides both a Streamlit web UI and standalone scripts for batch processing scanned A/B side answer card images.

The system supports:
- Automatic multiple-choice bubble recognition using differential method (差分法) with blank reference cards, or fixed threshold fallback.
- Barcode decoding for student ID extraction.
- Subjective question area cropping and export.
- Manual result correction and Excel grade report generation.

All source code comments, UI labels, and documentation are written in **Chinese**.

## Directory Structure

```
.
├── omr_demo/                    # Main application code
│   ├── app.py                   # Streamlit web demo entry point
│   ├── core/
│   │   ├── __init__.py          # (empty)
│   │   └── processor.py         # Core CardProcessor class (OMR logic)
│   ├── templates/
│   │   └── english.json         # Answer card template (bubble coords, barcode ROI, subjective areas)
│   ├── calibrate.py             # Interactive OpenCV calibration tool
│   ├── fit_template.py          # Programmatic template coordinate generator
│   ├── analyze_card.py          # Template-matching bubble detection analysis script
│   ├── test_processor.py        # Standalone processor test
│   ├── test_visual.py           # Visual debugging script for bubble detection
│   ├── requirements.txt         # Python dependencies
│   └── output/                  # Generated subjective crops (created at runtime)
│       └── subjective/
│           └── {student_id}/
├── testPaper/                   # Sample scanned answer card images
│   ├── 911156C_22104651_01A.jpg # A-side (选择题 + 部分主观题)
│   ├── 911156C_22104651_01B.jpg # B-side (主观题)
│   └── ... (paired A/B images)
```

## Technology Stack

- **Python** 3.12+
- **OpenCV** (`opencv-python`) – image processing, thresholding, template matching
- **Streamlit** – web UI for interactive upload, processing, and correction
- **NumPy** – array operations and fill-ratio calculations
- **Pandas** – result aggregation and Excel export
- **pyzbar** – barcode decoding for student IDs
- **Pillow** – image display in Streamlit
- **openpyxl** – Excel file generation engine

## Build and Run Commands

No build step is required. This is a pure Python project.

### Install Dependencies

```bash
cd omr_demo
pip install -r requirements.txt
```

### Run Streamlit Web App

```bash
cd omr_demo
streamlit run app.py
```

### Run Standalone Test Scripts

```bash
cd omr_demo
python test_processor.py   # Test batch processing on sample images
python test_visual.py      # Generate visual_check.jpg for debugging bubble detection
python analyze_card.py     # Analyze answer card with template matching
python calibrate.py        # Launch interactive calibration GUI
python fit_template.py     # Regenerate english.json template from hardcoded coordinates
```

### Batch Process a Folder (No UI)

Use `core/processor.py` directly:

```python
from core.processor import CardProcessor
proc = CardProcessor("templates/english.json")
# optionally set blank reference
proc.set_blank_ref(blank_a_img, blank_b_img)
df = proc.process_folder("../testPaper", output_dir="output")
```

## Code Organization and Module Division

### `core/processor.py` — Core Engine

Contains `CardProcessor`, the single class responsible for all OMR logic:

- `__init__(template_path, blank_ref_path)` – loads JSON template.
- `set_blank_ref(img_a, img_b)` – stores grayscale blank references for differential recognition.
- `preprocess(img)` – converts to grayscale + inverted binary threshold (180).
- `scale_coords(x, y, img_w, img_h)` – maps template coordinates to current image size.
- `detect_barcode(img, page)` – crops barcode ROI and decodes with `pyzbar`.
- `recognize_choices(img, page, threshold)` – main OMR method:
  - If blank reference exists: computes `cv2.subtract(roi, blank_roi)` and measures significant pixel difference ratio.
  - Else: measures black pixel ratio directly.
  - Returns `Dict[q_num → "A" | "B" | "C" | "D" | None | "X(多涂)"]`.
- `crop_subjective(img, student_id, page, output_dir)` – crops subjective areas defined in template and saves JPGs.
- `process_pair(img_a, img_b, student_id, output_dir)` – high-level wrapper that processes an A+B pair.
- `process_folder(input_dir, output_dir)` – batch processes all `*A.jpg` / `*B.jpg` pairs in a directory.

### `app.py` — Streamlit Frontend

Three-tab UI:
1. **模板与参考** – Upload blank A/B cards, load template, preview bubble/subjective overlays.
2. **批量处理** – Upload multiple A+B pairs, run OMR with adjustable threshold slider, view summary table.
3. **结果核对与导出** – Per-student data editor for manual corrections, subjective image preview, Excel export.

Session state keys: `processor`, `blank_a`, `blank_b`, `results`, `standard_answers`, `manual_corrections`.

### Template JSON Format (`templates/english.json`)

```json
{
  "name": "english_2026",
  "image_size": {"w": 1237, "h": 1741},
  "pages": {
    "A": {
      "barcode": {"x": 730, "y": 220, "w": 280, "h": 80},
      "bubbles": [
        {"q": 1, "opt": "A", "x": 248, "y": 517, "w": 12, "h": 12},
        ...
      ],
      "subjective": {
        "42": {"x1": 120, "y1": 1100, "x2": 1120, "y2": 1200, "score": 2}
      }
    },
    "B": {
      "subjective": { ... }
    }
  }
}
```

## Development Conventions

### File Naming Convention for Input Images

The system expects paired scans with strict naming:
- A-side: `*{something}A.jpg`
- B-side: `*{something}B.jpg`
- Examples: `911156C_22104651_01A.jpg` + `911156C_22104651_01B.jpg`
- `process_folder()` derives `student_id` from the stem after the last underscore, e.g. `01`.

### Coordinate System

- All template coordinates are based on a reference image size (`1237×1741` for the current English card).
- `scale_coords()` performs proportional scaling to match the actual input image dimensions.
- Bubble dimensions (`w`, `h`) are also scaled; a minimum size of `8–10 px` is enforced in code.

### OMR Thresholds

- Default recognition threshold: **0.10–0.15** (differential fill ratio).
- Threshold is exposed as a slider in Streamlit (range 0.02–0.30).
- Lower = more sensitive (risk of false positives); higher = stricter (risk of misses).

### Output Directory Layout

```
omr_demo/output/subjective/{student_id}/
├── 42.jpg
├── 43.jpg
├── 44.jpg
├── 45.jpg
├── 46-55.jpg
├── 56-60.jpg
└── writing.jpg
```

## Testing Strategy

There is **no formal unit-test framework** (pytest/unittest) in this project. Testing is done via standalone scripts:

| Script | Purpose |
|--------|---------|
| `test_processor.py` | End-to-end test: loads template, sets blank ref, processes one pair, prints first 15 answers. |
| `test_visual.py` | Generates `visual_check.jpg` with color-coded rectangles (red=filled, yellow=suspicious, green=empty) and threshold sweep output. |
| `analyze_card.py` | Uses OpenCV template matching (`cv2.matchTemplate`) to auto-detect bubble positions; outputs `analyzed_a.jpg` and `detected_bubbles.json`. |

### Recommended Testing Workflow

1. Run `python test_visual.py` to verify template alignment.
2. Adjust coordinates in `fit_template.py` or `calibrate.py` if circles/rectangles are off.
3. Run `python test_processor.py` to verify recognition accuracy on a known blank + filled pair.
4. Use the Streamlit app for interactive threshold tuning before batch processing.

## Deployment Notes

- This is a **local/demo-grade** application. There is no containerization, CI/CD, or production server configuration.
- Streamlit runs in development mode by default. For sharing within a local network, use:
  ```bash
  streamlit run app.py --server.address 0.0.0.0
  ```
- The `output/` directory is created at runtime and is currently tracked in the repository (contains sample crops).

## Security Considerations

- File uploads in Streamlit are handled entirely in-memory and local; no external network calls are made.
- `pyzbar` decodes barcodes without validation — downstream consumers should sanitize the `barcode` / `student_id` fields before using them in databases or filenames.
- No authentication or authorization is implemented in the Streamlit app.
- Path traversal is unlikely because `process_folder()` uses `pathlib.Path` on a configured directory, but avoid exposing `process_folder` to untrusted user input.
