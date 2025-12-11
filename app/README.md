## SAM3 FastAPI Web App

Minimal FastAPI server plus custom frontend for promptable image segmentation with the SAM3 image model. Users can upload a file or point to an image URL, then run segmentation via text, box, or dot prompts.

### Quickstart

1) Install the package (installs FastAPI + uvicorn via `pyproject.toml`):
```bash
pip install -e .
```

2) Ensure the SAM3 checkpoint is reachable (Hugging Face auth or local file). To force a local file, set `SAM3_CHECKPOINT_PATH=/path/to/ckpt.pth`.

3) Launch the server from the repo root:
```bash
uvicorn app.main:app --reload --port 8000
```
Then open `http://localhost:8000`.

### Endpoints
- `GET /` - serves the custom UI.
- `POST /segment` - multipart form for image + prompt. Supports:
- `prompt_type`: `text`, `box`, or `dot`
- `text_prompt`: required for text prompts
- Box fields: `box_x0`, `box_y0`, `box_x1`, `box_y1` (pixel coords in the source image), `box_label` (boolean)
- Dot fields: `dot_x`, `dot_y` (pixel coords), `dot_size` (px radius mapped to a tiny box), `dot_label` (boolean)
- `confidence`: float threshold (0-1)
- Optional ROI crop: `roi_x0`, `roi_y0`, `roi_x1`, `roi_y1` (pixel coords). If provided, the image is cropped before prompting.
- Optional scale for measurements: `scale_length` (real-world length for the scale line), `scale_px` (pixel distance of the scale line), `scale_unit` (string unit). When provided, the API returns per-mask areas in `area_unit`.
- `GET /health` - basic health probe.

### Notes
- Model device is auto-selected (`cuda` if available). Heavy inference will be slow on CPU.
- URL images are downloaded server-side with a short timeout; uploads are kept in-memory.
- Dot prompts are converted into a small geometric box centered on the click so they flow through SAM3's geometric prompt path.
