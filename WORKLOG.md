## SAM3 FastAPI Web App – Worklog / Handoff

### What’s built
- FastAPI backend serving a custom UI for SAM3 image segmentation.
- Box-only prompt workflow (text/dot removed).
- ROI support: ROI box drawn on the canvas; ROI-only mode returns a cropped overlay for the ROI while keeping full-image context on the canvas.
- Scale support: scale set on the full image remains valid for ROI segmentation; lengths are reported (longest bbox edge) in scaled units if provided.
- Per-segment metrics: length, green/red dominance (%) using channel dominance heuristic, confidence. Table shows thumbnails and metrics; CSV export available.
- Overlay options: toggle confidence labels, mask fill, adjust box thickness, ROI-only view.
- Hover overlay: highlight segments with tooltip (length/green/red/confidence). Hover respects ROI offsets.
- Drag-and-drop + file/URL loading; status overlay for load/errors.

### Key files
- `app/main.py` — FastAPI app, ROI handling, overlay rendering, metrics computation, CSV data.
- `app/static/app.js` — UI logic, ROI offsets, hover overlay, form submission, CSV export.
- `app/static/styles.css` — UI styling.
- `app/templates/index.html` — Layout (left controls, canvas, output), user guide, controls.

### Current UI layout
- Left column: Image source (file/URL + Load URL), Scale (length/unit + mark/clear), ROI controls (mark/clear/revert), Prompt (box instructions, confidence slider, display toggles, box thickness), action buttons (Draw box, Clear prompt, Segment).
- Right column: Canvas with status/log overlay; full image with ROI box for context.
- Bottom: Segmentation output (ROI-only shows cropped ROI overlay), stats, table with export CSV.

### ROI behavior
- ROI box is sent to backend; inference crops to ROI.
- ROI-only output uses the cropped ROI overlay; ROI box is *not* drawn on the ROI-only output. Canvas retains full image + ROI box for context.
- Hover math accounts for ROI offsets; boxes/tooltips align on both full and ROI-only displays.

### Measurement
- Length = max(width, height) of bbox; scaled by provided scale (pixels per unit) else pixels.
- Green/Red %: fraction of mask pixels where a channel dominates the others by 10%.

### Known user-facing instructions
- Draw box: click “Draw box” then click start/end (UI hint says click twice).
- ROI: mark ROI, optional ROI-only toggle to view cropped overlay.
- Scale: set on full image before ROI if needed; remains valid after ROI.
- Export: CSV button above table (no images in CSV).

### Recent bug fixes
- Removed uploadBtn reference; single set of toggle refs.
- ROI-only output now shows cropped ROI overlay; no ROI box drawn there.
- Hover/tooltips offset by ROI when displaying ROI-only.
- Box-only mode enforced.

### Potential follow-ups
- Ensure “Draw box” button explicitly arms a two-click flow in JS (currently still supports drag).
- Validate CORS/host binding for external access (`uvicorn ... --host 0.0.0.0` + optional CORS middleware).
- If ROI alignment issues persist, recheck coordinate offsets in `handleResultHover` and backend padding logic.
