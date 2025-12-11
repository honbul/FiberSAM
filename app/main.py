import asyncio
import base64
import io
import os
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

app = FastAPI(title="SAM3 Image Segmentation")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def _resolve_device():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Torch is required to run the SAM3 web app but was not found."
        ) from exc

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> Tuple:
    """Load the SAM3 image model and processor once."""
    device = _resolve_device()
    checkpoint = os.environ.get("SAM3_CHECKPOINT_PATH")
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint,
        load_from_HF=checkpoint is None,
    )
    processor = Sam3Processor(model, device=device)
    return model, processor


MODEL, PROCESSOR = _load_model()
MODEL_LOCK = asyncio.Lock()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the UI shell."""
    with open("app/templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _load_image(image_file: Optional[UploadFile], image_url: Optional[str]) -> Image.Image:
    """Load an image from an upload or URL."""
    if image_file is None and not image_url:
        raise HTTPException(status_code=400, detail="Provide an upload or an image URL.")

    if image_file is not None:
        data = image_file.file.read()
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # pragma: no cover - input validation
            raise HTTPException(status_code=400, detail="Unsupported image format.") from exc

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - runtime fetch
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {exc}") from exc


def _normalize_box(box_xyxy: Dict[str, float], width: int, height: int) -> list:
    """Convert xyxy pixel box to cxcywh normalized for the processor."""
    x0, y0, x1, y1 = (
        float(box_xyxy["x0"]),
        float(box_xyxy["y0"]),
        float(box_xyxy["x1"]),
        float(box_xyxy["y1"]),
    )
    cx = (x0 + x1) / 2.0 / width
    cy = (y0 + y1) / 2.0 / height
    w = abs(x1 - x0) / width
    h = abs(y1 - y0) / height
    return [cx, cy, w, h]


def analyze_color_dominance(base_np, mask_np, x0, y0, axis_len, axis_is_x):
    mask_pixels = base_np[mask_np]

    red_ratio = 0.0
    green_ratio = 0.0
    red_pos = np.zeros(axis_len, dtype=bool)
    green_pos = np.zeros(axis_len, dtype=bool)

    if mask_pixels.shape[0] == 0:
        return red_ratio, green_ratio, red_pos, green_pos

    r = mask_pixels[:, 0].astype(np.float32)
    g = mask_pixels[:, 1].astype(np.float32)
    b = mask_pixels[:, 2].astype(np.float32)

    # Ignore dark pixels (e.g. black background)
    threshold = 30.0
    
    red_dom_mask = (r > g * 1.1) & (r > b * 1.1) & (r > threshold)
    green_dom_mask = (g > r * 1.1) & (g > b * 1.1) & (g > threshold)

    pixel_coords = np.argwhere(mask_np)  # (y, x)

    red_bins = np.zeros(axis_len, dtype=np.int64)
    if red_dom_mask.any():
        red_pixel_coords = pixel_coords[red_dom_mask]
        val = red_pixel_coords[:, 1] if axis_is_x else red_pixel_coords[:, 0]
        offset = x0 if axis_is_x else y0
        indices = val - offset
        valid_idx = (indices >= 0) & (indices < axis_len)
        indices = indices[valid_idx]
        if indices.size > 0:
            bins = np.bincount(indices, minlength=axis_len)
            if bins.shape[0] > axis_len:
                bins = bins[:axis_len]
            red_bins = bins

    green_bins = np.zeros(axis_len, dtype=np.int64)
    if green_dom_mask.any():
        green_pixel_coords = pixel_coords[green_dom_mask]
        val = green_pixel_coords[:, 1] if axis_is_x else green_pixel_coords[:, 0]
        offset = x0 if axis_is_x else y0
        indices = val - offset
        valid_idx = (indices >= 0) & (indices < axis_len)
        indices = indices[valid_idx]
        if indices.size > 0:
            bins = np.bincount(indices, minlength=axis_len)
            if bins.shape[0] > axis_len:
                bins = bins[:axis_len]
            green_bins = bins

    red_pos = (red_bins > green_bins) & (red_bins > 0)
    green_pos = (green_bins > red_bins) & (green_bins > 0)

    red_cov = int(red_pos.sum())
    green_cov = int(green_pos.sum())
    total_cov = red_cov + green_cov

    if total_cov > 0:
        red_ratio = (red_cov / total_cov) * 100.0
        green_ratio = (green_cov / total_cov) * 100.0

    return red_ratio, green_ratio, red_pos, green_pos


def generate_normalized_line_b64(axis_len, red_pos, green_pos):
    line_w, line_h = 200, 14
    source_line = np.full((1, axis_len, 3), 220, dtype=np.uint8)

    if red_pos.any():
        source_line[0, red_pos] = np.array([255, 80, 80], dtype=np.uint8)
    if green_pos.any():
        source_line[0, green_pos] = np.array([80, 255, 160], dtype=np.uint8)

    line_img = Image.fromarray(source_line)
    line_img = line_img.resize((line_w, line_h), resample=Image.NEAREST)

    line_buff = io.BytesIO()
    line_img.save(line_buff, format="PNG")
    return base64.b64encode(line_buff.getvalue()).decode("ascii")


def _draw_overlay(
    image: Image.Image,
    state: Dict,
    requested_box: Optional[Dict[str, float]],
    requested_dot: Optional[Dict[str, float]],
    include_box: bool,
    show_scores: bool,
    number_labels: Optional[list] = None,
    fill_masks: bool = True,
    box_thickness: float = 3.0,
):
    """Blend predicted masks and prompts onto the source image."""
    base = image.convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)

    masks = state.get("masks", [])
    boxes = state.get("boxes", [])
    scores = state.get("scores", [])
    colors = [
        (255, 99, 71),
        (66, 135, 245),
        (34, 197, 94),
        (235, 160, 5),
        (168, 85, 247),
        (16, 185, 129),
        (244, 114, 182),
    ]

    thickness = max(1, int(box_thickness))

    for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask_np = mask[0].cpu().numpy()
        color = colors[idx % len(colors)]
        if fill_masks:
            mask_img = Image.new("RGBA", base.size, (*color, 110))
            mask_layer.paste(
                mask_img,
                (0, 0),
                Image.fromarray((mask_np > 0.5).astype(np.uint8) * 255),
            )

        if include_box:
            x0, y0, x1, y1 = box.cpu().numpy()
            draw.rectangle([x0, y0, x1, y1], outline=color, width=thickness)
            label_y = max(y0 - 24, 0)
            label_x = x0
            if show_scores:
                draw.rectangle(
                    [label_x, label_y, label_x + 90, y0], fill=(*color, 200)
                )
                draw.text((label_x + 4, label_y + 2), f"{score:.2f}", fill="white")
            if number_labels:
                number_str = str(number_labels[idx])
                draw.rectangle([x0, y0, x0 + 26, y0 + 18], fill=(*color, 230))
                draw.text((x0 + 4, y0 + 2), number_str, fill="white")

    # Show user prompts back on the image
    if requested_box:
        draw.rectangle(
            [requested_box["x0"], requested_box["y0"], requested_box["x1"], requested_box["y1"]],
            outline=(255, 255, 255),
            width=2,
        )
    if requested_dot:
        r = 6
        x, y = requested_dot["x"], requested_dot["y"]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))

    blended = Image.alpha_composite(base, mask_layer)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _segment(
    image: Image.Image,
    prompt_type: str,
    text_prompt: Optional[str],
    box_payload: Optional[Dict[str, float]],
    dot_payload: Optional[Dict[str, float]],
    confidence: float,
    roi: Optional[Dict[str, float]],
    show_scores: bool,
    scale_length: Optional[float],
    scale_px: Optional[float],
    scale_unit: Optional[str],
    fill_masks: bool,
    box_thickness: float,
    roi_only: bool,
):
    original_image = image
    if roi:
        x0, y0, x1, y1 = (
            int(roi["x0"]),
            int(roi["y0"]),
            int(roi["x1"]),
            int(roi["y1"]),
        )
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, image.width), min(y1, image.height)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise HTTPException(status_code=400, detail="ROI is invalid or too small.")
        crop_offset = (x0, y0)
        image = image.crop((x0, y0, x1, y1))
    else:
        crop_offset = None

    width, height = image.size
    orig_width, orig_height = original_image.size
    state = PROCESSOR.set_image(image, state={})
    PROCESSOR.set_confidence_threshold(confidence)

    if prompt_type == "text":
        if not text_prompt:
            raise HTTPException(status_code=400, detail="Text prompt is required.")
        state = PROCESSOR.set_text_prompt(text_prompt, state)
        requested_box = None
        requested_dot = None
    elif prompt_type == "box":
        if not box_payload:
            raise HTTPException(status_code=400, detail="Box prompt is required.")
        label = bool(box_payload.get("label", True))
        norm_box = _normalize_box(box_payload, width, height)
        state = PROCESSOR.add_geometric_prompt(norm_box, label, state)
        requested_box = box_payload
        requested_dot = None
    elif prompt_type == "dot":
        if not dot_payload:
            raise HTTPException(status_code=400, detail="Dot prompt is required.")
        label = bool(dot_payload.get("label", True))
        size_px = float(dot_payload.get("size", 12))
        norm_box = [
            dot_payload["x"] / width,
            dot_payload["y"] / height,
            size_px / width,
            size_px / height,
        ]
        state = PROCESSOR.add_geometric_prompt(norm_box, label, state)
        requested_box = None
        requested_dot = dot_payload
    else:
        raise HTTPException(status_code=400, detail="Unknown prompt type.")

    scores = [float(s.item()) for s in state.get("scores", [])]
    num_labels = list(range(1, len(scores) + 1))
    lengths = []
    length_unit = None
    px_per_unit = None
    if scale_length and scale_px and scale_length > 0 and scale_px > 0:
        px_per_unit = scale_px / scale_length
        length_unit = scale_unit or "unit"

    segments = []
    masks = state.get("masks", [])
    boxes = state.get("boxes", [])

    # If cropped and not ROI-only, pad masks/boxes back to original coords
    if crop_offset and (not roi_only) and isinstance(masks, np.ndarray) is False:
        try:
            import torch
        except Exception:
            torch = None
        ox, oy = crop_offset
        if torch and isinstance(masks, torch.Tensor):
            n, c, h, w = masks.shape
            padded = masks.new_zeros((n, c, orig_height, orig_width))
            padded[:, :, oy : oy + h, ox : ox + w] = masks
            masks = padded
            state["masks"] = masks
        elif isinstance(masks, list) and len(masks) > 0 and hasattr(masks[0], "shape"):
            padded_list = []
            for m in masks:
                h, w = m.shape[-2:]
                blank = np.zeros((1, orig_height, orig_width), dtype=m.dtype)
                blank[:, oy : oy + h, ox : ox + w] = m
                padded_list.append(blank)
            masks = padded_list
            state["masks"] = masks

        if boxes is not None:
            if torch and isinstance(boxes, torch.Tensor):
                offset = torch.tensor([ox, oy, ox, oy], device=boxes.device, dtype=boxes.dtype)
                boxes = boxes + offset
                state["boxes"] = boxes
            elif isinstance(boxes, np.ndarray):
                boxes = boxes + np.array([ox, oy, ox, oy])
                state["boxes"] = boxes

    base_image = image if (roi_only and crop_offset) else original_image
    base_rgba = base_image.convert("RGBA")
    base_np = np.array(base_rgba)
    masks = state.get("masks", [])
    for idx, (mask, score) in enumerate(zip(masks, scores), start=1):
        mask_np = mask[0].cpu().numpy() > 0.5
        if not mask_np.any():
            continue
        ys, xs = np.nonzero(mask_np)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        width_px = (x1 - x0 + 1)
        height_px = (y1 - y0 + 1)
        length_px = max(width_px, height_px)
        if px_per_unit:
            lengths.append(length_px / px_per_unit)
            length_val = length_px / px_per_unit
        else:
            lengths.append(float(length_px))
            length_val = float(length_px)

        # Color proportion along major axis (ignore thickness; use per-axis dominance)
        axis_is_x = width_px >= height_px
        axis_len = max(1, length_px)

        red_ratio, green_ratio, red_pos, green_pos = analyze_color_dominance(
            base_np, mask_np, x0, y0, axis_len, axis_is_x
        )

        normalized_line_b64 = generate_normalized_line_b64(axis_len, red_pos, green_pos)

        seg_rgba = base_np.copy()
        alpha = np.where(mask_np, 255, 0).astype(np.uint8)
        seg_rgba[..., 3] = alpha
        seg_crop = seg_rgba[y0 : y1 + 1, x0 : x1 + 1, :]
        seg_img = Image.fromarray(seg_crop)
        buff = io.BytesIO()
        seg_img.save(buff, format="PNG")
        seg_b64 = base64.b64encode(buff.getvalue()).decode("ascii")

        segments.append(
            {
                "number": idx,
                "length": length_val,
                "confidence": float(score),
                "image": seg_b64,
                "box": [int(x0), int(y0), int(x1), int(y1)],
                "length_unit": length_unit,
                "green_ratio": green_ratio,
                "red_ratio": red_ratio,
                "normalized_line": normalized_line_b64,
            }
        )

    overlay = _draw_overlay(
        base_image,
        state,
        requested_box,
        requested_dot,
        include_box=True,
        show_scores=show_scores,
        number_labels=num_labels,
        fill_masks=fill_masks,
        box_thickness=box_thickness,
    )

    out_w, out_h = base_image.size if roi_only and crop_offset else (orig_width, orig_height)

    return overlay, len(scores), scores, out_w, out_h, lengths, length_unit, segments


@app.post("/segment")
async def segment(
    image: Optional[UploadFile] = File(default=None),
    image_url: Optional[str] = Form(default=None),
    prompt_type: str = Form(...),
    text_prompt: Optional[str] = Form(default=None),
    box_x0: Optional[float] = Form(default=None),
    box_y0: Optional[float] = Form(default=None),
    box_x1: Optional[float] = Form(default=None),
    box_y1: Optional[float] = Form(default=None),
    box_label: Optional[bool] = Form(default=True),
    dot_x: Optional[float] = Form(default=None),
    dot_y: Optional[float] = Form(default=None),
    dot_size: Optional[float] = Form(default=12),
    dot_label: Optional[bool] = Form(default=True),
    confidence: float = Form(0.5),
    roi_x0: Optional[float] = Form(default=None),
    roi_y0: Optional[float] = Form(default=None),
    roi_x1: Optional[float] = Form(default=None),
    roi_y1: Optional[float] = Form(default=None),
    show_scores: bool = Form(default=True),
    roi_only: bool = Form(default=False),
    scale_length: Optional[float] = Form(default=None),
    scale_px: Optional[float] = Form(default=None),
    scale_unit: Optional[str] = Form(default=None),
    fill_masks: bool = Form(default=True),
    box_thickness: float = Form(default=3.0),
):
    """Perform segmentation based on the selected prompt."""
    loaded_image = _load_image(image, image_url)

    box_payload = None
    if all(v is not None for v in [box_x0, box_y0, box_x1, box_y1]):
        box_payload = {
            "x0": box_x0,
            "y0": box_y0,
            "x1": box_x1,
            "y1": box_y1,
            "label": box_label,
        }

    dot_payload = None
    if dot_x is not None and dot_y is not None:
        dot_payload = {"x": dot_x, "y": dot_y, "size": dot_size, "label": dot_label}

    roi_payload = None
    if all(v is not None for v in [roi_x0, roi_y0, roi_x1, roi_y1]):
        roi_payload = {"x0": roi_x0, "y0": roi_y0, "x1": roi_x1, "y1": roi_y1}

    async with MODEL_LOCK:
        (
            overlay,
            count,
            scores,
            out_w,
            out_h,
            lengths,
            length_unit,
            segments_table,
        ) = await asyncio.to_thread(
            _segment,
            loaded_image,
            prompt_type,
            text_prompt,
            box_payload,
            dot_payload,
            confidence,
            roi_payload,
            show_scores,
            scale_length,
            scale_px,
            scale_unit,
            fill_masks,
            box_thickness,
            roi_only,
        )

    # For ROI-only, overlay already cropped; for full-image we keep original sizing

    return JSONResponse(
        {
            "overlay": overlay,
            "num_masks": count,
            "scores": scores,
            "width": out_w,
            "height": out_h,
            "lengths": lengths,
            "length_unit": length_unit,
            "segments": segments_table,
            "roi": roi_payload,
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
