## SAM3 FastAPI Segmentation App

### What this is
A FastAPI-based web application that wraps the SAM3 image model with a custom UI for box-based segmentation, ROI handling, scaling, and per-segment analysis (length and color dominance).

### Key capabilities
- Upload or URL image loading (drag-and-drop supported).
- Box-only prompts with adjustable confidence, mask fill toggle, and box thickness.
- ROI selection: full image remains on the canvas for context; ROI-only output shows the cropped ROI overlay. Hover/tooltip logic respects ROI offsets.
- Scale support: define a scale on the original image; lengths stay valid when segmenting within ROI.
- Per-segment metrics: length (scaled if provided), green/red dominance %, confidence. Table with thumbnails and CSV export (no images in CSV).
- Hover overlay to highlight segments and show metrics; numbers on overlay match the table.

### Files to know
- `app/main.py` — FastAPI app, model inference, ROI handling, overlay rendering, metrics.
- `app/templates/index.html` — UI layout (controls, canvas, output).
- `app/static/app.js` — Frontend logic: load/ROI/scale, form submission, hover overlay, CSV export.
- `app/static/styles.css` — Styling.
- `WORKLOG.md` — Development handoff/notes.

### Running
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000  # adjust host/port as needed
```
Ensure SAM3 checkpoints are available (via HF or `SAM3_CHECKPOINT_PATH`) and dependencies installed (`pip install -e .`).

### Usage flow
1) Load image (upload/drag-drop or URL + Load URL).
2) (Optional) Set scale on full image; mark ROI if desired (ROI box stays on canvas; ROI-only toggles cropped output).
3) Click “Draw box” then click start/end points on the canvas; adjust confidence, mask fill, thickness, ROI-only, show confidence.
4) Click Segment; hover output to inspect lengths/green/red/confidence; export CSV for metrics.

### Notes
- Only box prompts are supported (text/dot removed).
- ROI box is shown only on the canvas; ROI-only output displays the cropped ROI overlay.
- CSV omits segment images (thumbnails stay in the UI).
