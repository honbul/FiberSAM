## SAM3 FastAPI Web App

Minimal FastAPI server plus custom frontend for promptable image segmentation with the SAM3 image model. Users can upload a file (including TIFF) or point to an image URL, then run segmentation via box prompts.

### Quickstart

1) Install the package (installs FastAPI + uvicorn via `pyproject.toml`):
```bash
pip install -e .
```

2) Ensure the SAM3 checkpoint is reachable (Hugging Face auth or local file). To force a local file, set `SAM3_CHECKPOINT_PATH=/path/to/ckpt.pth`.

3) Launch the server from the repo root:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Then open `http://localhost:8000`.

### Endpoints
- `GET /` - serves the custom UI.
- `POST /segment` - multipart form for image + prompt. Supports:
  - `prompt_type`: `box`
  - Box fields: `box_x0`, `box_y0`, `box_x1`, `box_y1` (pixel coords in the source image), `box_label` (boolean)
  - `confidence`: float threshold (0-1)
  - Optional scale for measurements: `scale_length` (real-world length for the scale line), `scale_px` (pixel distance of the scale line), `scale_unit` (string unit). When provided, the API returns per-mask lengths.
- `GET /health` - basic health probe.

### UX Behavior
- **TIFF Support**: TIFF files are supported; for multi-page TIFFs, only the first frame is processed.
- **Box Interaction**: Click once to start a box, move the mouse, and click again to finish. Press `Esc` to cancel an in-progress box.
- **Inference**: Only box-based prompts are currently supported in the web app.

### Notes
- Model device is auto-selected (`cuda` if available). Heavy inference will be slow on CPU.
- URL images are downloaded server-side with a short timeout; uploads are kept in-memory.
