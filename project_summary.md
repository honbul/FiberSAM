## Project Summary

This repository hosts a FastAPI-based web application wrapping the SAM3 image model with a custom UI for box-based segmentation, ROI handling, scaling, and per-segment analysis (length and color dominance).

### Core Features
- **Image input**: upload/drag-and-drop or URL load.
- **Box-only prompts**: draw boxes (two clicks) to segment; adjustable confidence, mask fill toggle, and box thickness.
- **ROI workflow**: select an ROI; ROI-only output shows the cropped overlay. The canvas always shows the full image with ROI box for context. Hover and overlays respect ROI offsets.
- **Scale support**: define a scale on the original image; lengths stay valid when segmenting within ROI. Lengths reported as the longest bbox edge (scaled if provided).
- **Per-segment metrics**: length, green/red dominance (%) based on linear coverage (ignoring thickness), confidence. A table shows thumbnails, normalized line visualization (red/green spans), metrics, and supports CSV export (no images in CSV).
- **Hover overlay**: highlights segments with tooltips (length/green/red/confidence). Overlay numbering matches the table.

### Key Files
- `app/main.py` — FastAPI app, model inference, ROI handling, overlay rendering, length/color metrics, normalized line generation.
- `app/templates/index.html` — UI layout (controls, canvas, output, user guide).
- `app/static/app.js` — Frontend logic for loading, ROI/scale, prompts, hover overlay, table rendering, CSV export.
- `app/static/styles.css` — Styling for the custom UI and table thumbnails.
- `WORKLOG.md` — Development handoff/notes.
- `PROJECT_OVERVIEW.md` — High-level project overview.

### Running
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Ensure SAM3 checkpoints are available (via HF or `SAM3_CHECKPOINT_PATH`) and dependencies installed (`pip install -e .`).

### Usage Flow
1) Load image (upload/drag-drop or URL + Load URL).
2) (Optional) Set scale on the full image; mark ROI if desired (ROI box stays on canvas; ROI-only shows cropped output).
3) Click “Draw box”, then click start/end points on the canvas; adjust confidence, fill masks, box thickness, ROI-only, show confidence.
4) Click Segment; hover the output to inspect lengths/green/red/confidence; table shows thumbnails and normalized line; export CSV for metrics.

### Notes
- Only box prompts are supported (text/dot removed).
- ROI box is displayed only on the canvas; ROI-only output displays the cropped ROI overlay.
- Green/red ratios use linear coverage along the major axis (thickness ignored). Normalized line column visualizes red/green spans for each segment.

### Recent Updates (Dec 11, 2025)
- **Refactoring**: Extracted color dominance analysis and normalized line generation into helper functions (`analyze_color_dominance`, `generate_normalized_line_b64`) in `app/main.py` to fix variable scope issues and improve maintainability.
- **Bug Fix**: Fixed an issue where the "Normalized segment" column was empty and caused subsequent table columns to shift. Updated `app/static/app.js` to strictly enforce column structure and added cache busting to `app/templates/index.html`.
- **Logic Improvement**: Updated `analyze_color_dominance` to include an intensity threshold (value > 30), preventing dark/black background pixels from being incorrectly classified as red or green.