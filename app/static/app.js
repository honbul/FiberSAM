const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const previewImage = document.getElementById("previewImage");
const fileInput = document.getElementById("fileInput");
const urlInput = document.getElementById("urlInput");
const loadUrlBtn = document.getElementById("loadUrl");
const segmentBtn = document.getElementById("segmentBtn");
const clearPromptBtn = document.getElementById("clearPrompt");
const textPromptInput = document.getElementById("textPrompt");
const boxLabelInput = document.getElementById("boxLabel");
const dotLabelInput = document.getElementById("dotLabel");
const dotSizeInput = document.getElementById("dotSize");
const confidenceInput = document.getElementById("confidence");
const confValue = document.getElementById("confValue");
const resultImg = document.getElementById("resultImage");
const resultMeta = document.getElementById("resultMeta");
const maskCount = document.getElementById("maskCount");
const areaList = document.getElementById("areaList");
const lengthList = document.getElementById("areaList"); // reuse same DOM id for lengths
const scoreList = document.getElementById("scoreList");
const runtimeEl = document.getElementById("runtime");
const canvasStatus = document.getElementById("canvasStatus");
const roiStartBtn = document.getElementById("roiStart");
const roiClearBtn = document.getElementById("roiClear");
const roiRevertBtn = document.getElementById("roiRevert");
const scoreToggle = document.getElementById("scoreToggle");
const roiOnlyToggle = document.getElementById("roiOnlyToggle");
const scaleLengthInput = document.getElementById("scaleLength");
const scaleUnitInput = document.getElementById("scaleUnit");
const scaleMarkBtn = document.getElementById("scaleMark");
const scaleClearBtn = document.getElementById("scaleClear");
const segmentsBody = document.getElementById("segmentsBody");
const resultOverlay = document.getElementById("resultOverlay");
const resultOverlayShell = document.querySelector(".result-overlay-shell");
const resultTooltip = document.getElementById("resultTooltip");
const dropZone = document.querySelector(".canvas-shell") || document.body;
const canvasLog = document.getElementById("canvasLog");
const maskFillToggle = document.getElementById("maskFillToggle");
const boxThicknessInput = document.getElementById("boxThickness");
const exportCsvBtn = document.getElementById("exportCsv");

function updateCanvasLogVisibility(message) {
  if (!canvasLog) return;
  const msg = (message || "").toLowerCase();
  const isError = msg.includes("failed") || msg.includes("error");
  const hasImage = state.naturalWidth > 0 && state.naturalHeight > 0;
  const loaded =
    msg.includes("image loaded") ||
    msg.includes("ready to draw") ||
    msg.includes("done. source");
  // Hide overlay once an image is available unless we hit an error
  const show = (!hasImage && !loaded) || isError;
  canvasLog.style.display = show ? "flex" : "none";
}

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

const state = {
  promptType: "box",
  naturalWidth: 0,
  naturalHeight: 0,
  scaleX: 1,
  scaleY: 1,
  box: null,
  dot: null,
  drawing: false,
  start: null,
  imageFile: null,
  originalDataURL: null,
  roi: null,
  roiDraft: null,
  roiMode: false,
  roiApplied: false,
  scaleMode: false,
  scaleLine: null,
  scalePixels: null,
  resultSegs: [],
  resultNatural: { w: 0, h: 0 },
};

function setStatus(message) {
  canvasStatus.textContent = message;
  if (canvasLog) {
    canvasLog.textContent = message;
    updateCanvasLogVisibility(message);
  }
}

function togglePromptRows() {
  document.getElementById("boxPromptRow").style.display = "flex";
}

function getPointerPosition(evt) {
  const rect = canvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) * state.scaleX;
  const y = (evt.clientY - rect.top) * state.scaleY;
  return {
    x: clamp(x, 0, state.naturalWidth),
    y: clamp(y, 0, state.naturalHeight),
  };
}

function drawCanvas(previewBox = null) {
  if (!state.naturalWidth || !state.naturalHeight) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(previewImage, 0, 0, state.naturalWidth, state.naturalHeight);

  ctx.lineWidth = 2;

  if (state.scaleLine) {
    ctx.strokeStyle = "#80ffea";
    ctx.lineWidth = 4;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(state.scaleLine.x0, state.scaleLine.y0);
    ctx.lineTo(state.scaleLine.x1, state.scaleLine.y1);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#80ffea";
    ctx.beginPath();
    ctx.arc(state.scaleLine.x0, state.scaleLine.y0, 6, 0, Math.PI * 2);
    ctx.arc(state.scaleLine.x1, state.scaleLine.y1, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
  }

  const roiBox = state.roiDraft;
  if (roiBox) {
    ctx.strokeStyle = "#ffd166";
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(
      roiBox.x0,
      roiBox.y0,
      roiBox.x1 - roiBox.x0,
      roiBox.y1 - roiBox.y0
    );
    ctx.setLineDash([]);
  }

  const boxToDraw = previewBox || state.box;
  if (boxToDraw) {
    ctx.strokeStyle = boxLabelInput.checked ? "#7df3ff" : "#ff7e6b";
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(
      boxToDraw.x0,
      boxToDraw.y0,
      boxToDraw.x1 - boxToDraw.x0,
      boxToDraw.y1 - boxToDraw.y0
    );
    ctx.setLineDash([]);
  }

  if (state.dot) {
    ctx.fillStyle = dotLabelInput.checked ? "#9dffb0" : "#ff8fa3";
    ctx.beginPath();
    ctx.arc(state.dot.x, state.dot.y, 6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function setCanvasSize() {
  const maxWidth = 900;
  const ratio = Math.min(1, maxWidth / state.naturalWidth);
  if (!state.naturalWidth || !state.naturalHeight || ratio <= 0) return;
  canvas.width = state.naturalWidth;
  canvas.height = state.naturalHeight;
  canvas.style.width = `${state.naturalWidth * ratio}px`;
  canvas.style.height = `${state.naturalHeight * ratio}px`;
  state.scaleX = state.naturalWidth / (state.naturalWidth * ratio);
  state.scaleY = state.naturalHeight / (state.naturalHeight * ratio);
  if (resultOverlayShell) {
    syncResultOverlay();
  }
}

function resetPrompts() {
  state.box = null;
  state.dot = null;
  drawCanvas();
}

function handleFileChange(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  loadLocalFile(file);
}

function loadLocalFile(file) {
  state.imageFile = file;
  urlInput.value = "";
  state.roiApplied = false;
  state.roi = null;
  state.roiDraft = null;
  state.resultSegs = [];
  state.originalDataURL = null;
  setStatus(`Loading local image: ${file.name}...`);

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    state.originalDataURL = e.target.result;
    if (previewImage.complete && previewImage.naturalWidth > 0) {
      previewImage.onload();
    }
  };
  reader.onerror = () => {
    setStatus("Failed to read file.");
  };
  reader.readAsDataURL(file);
}

function handleUrlLoad() {
  const url = urlInput.value.trim();
  if (!url) {
    alert("Enter an image URL to load.");
    return;
  }
  setStatus("Downloading image...");
  fetch(url)
    .then((res) => {
      if (!res.ok) throw new Error("Failed to fetch image");
      return res.blob();
    })
    .then((blob) => {
      const objectUrl = URL.createObjectURL(blob);
      previewImage.crossOrigin = "anonymous";
      previewImage.src = objectUrl;
      state.imageFile = new File([blob], "remote_image", { type: blob.type });
      state.originalDataURL = objectUrl;
      state.roiApplied = false;
      state.roi = null;
      state.roiDraft = null;
      setStatus("URL loaded; ready to draw");
    })
    .catch((err) => {
      setStatus(`URL load failed: ${err.message}`);
      alert("Could not load image from that URL.");
    });
}

previewImage.onload = () => {
  state.naturalWidth = previewImage.naturalWidth;
  state.naturalHeight = previewImage.naturalHeight;
  if (!state.naturalWidth || !state.naturalHeight) {
    setStatus("Image failed to load (no dimensions).");
    return;
  }
  if (!state.originalDataURL) {
    state.originalDataURL = previewImage.src;
  }
  state.roi = null;
  state.roiDraft = null;
  state.roiApplied = false;
  state.roiMode = false;
  state.scaleLine = null;
  state.scalePixels = null;
  state.scaleMode = false;
  state.resultSegs = [];
  clearResultOverlay();
  setCanvasSize();
  resetPrompts();
  drawCanvas();
  canvas.style.visibility = "visible";
  setStatus(`Image loaded (${state.naturalWidth}x${state.naturalHeight})`);
  if (canvasLog) {
    canvasLog.style.display = "none";
  }
};

previewImage.onerror = () => {
  setStatus("Image failed to load. Please try another file or URL.");
};

function handlePromptTypeChange(evt) {
  state.promptType = evt.target.value;
  resetPrompts();
  togglePromptRows();
}

function startRoiMode() {
  if (!state.naturalWidth) {
    alert("Load an image first.");
    return;
  }
  state.roiApplied = false;
  state.roiMode = true;
  state.roiDraft = null;
  state.scaleMode = false;
  setStatus("Draw ROI on the canvas to apply crop.");
}

function applyRoiCrop() {
  if (!state.roiDraft) {
    alert("Draw an ROI box first.");
    return;
  }
  state.roi = { ...state.roiDraft };
  state.roiMode = false;
  state.roiApplied = true;
  state.roiDraft = null;
  drawCanvas();
  setStatus(
    `ROI applied (${(state.roi.x1 - state.roi.x0).toFixed(0)}x${(state.roi.y1 - state.roi.y0).toFixed(0)}).`
  );
}

function clearRoi() {
  revertImage();
  setStatus("ROI cleared and original view restored.");
}

function revertImage() {
  if (!state.originalDataURL) return;
  previewImage.src = state.originalDataURL;
  state.roi = null;
  state.roiDraft = null;
  state.roiMode = false;
  state.roiApplied = false;
  state.scaleLine = null;
  state.scalePixels = null;
  resetPrompts();
  setStatus("Reverted to original image.");
}

function startScaleMode() {
  const scaleLen = parseFloat(scaleLengthInput.value);
  if (!state.naturalWidth) {
    alert("Load an image first.");
    return;
  }
  if (Number.isNaN(scaleLen) || scaleLen <= 0) {
    alert("Enter a positive scale length first.");
    return;
  }
  state.scaleMode = true;
  state.scaleLine = null;
  state.scalePixels = null;
  setStatus("Click two points on the canvas to set scale.");
}

function clearScale() {
  state.scaleMode = false;
  state.scaleLine = null;
  state.scalePixels = null;
  drawCanvas();
  setStatus("Scale cleared.");
}

canvas.addEventListener("mousedown", (evt) => {
  if (!state.naturalWidth) return;
  const pt = getPointerPosition(evt);
  state.drawing = true;
  state.start = pt;
  if (state.roiMode) {
    state.roiDraft = { x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y };
  }
});

canvas.addEventListener("mousemove", (evt) => {
  if (!state.drawing) return;
  const current = getPointerPosition(evt);
  if (state.roiMode) {
    state.roiDraft = {
      x0: clamp(Math.min(state.start.x, current.x), 0, state.naturalWidth),
      y0: clamp(Math.min(state.start.y, current.y), 0, state.naturalHeight),
      x1: clamp(Math.max(state.start.x, current.x), 0, state.naturalWidth),
      y1: clamp(Math.max(state.start.y, current.y), 0, state.naturalHeight),
    };
    drawCanvas();
    return;
  }
  if (state.promptType !== "box") return;
  drawCanvas({
    x0: state.start.x,
    y0: state.start.y,
    x1: current.x,
    y1: current.y,
  });
});

canvas.addEventListener("mouseup", (evt) => {
  if (!state.drawing) return;
  state.drawing = false;
  const end = getPointerPosition(evt);

  if (state.roiMode) {
    state.roiDraft = {
      x0: clamp(Math.min(state.start.x, end.x), 0, state.naturalWidth),
      y0: clamp(Math.min(state.start.y, end.y), 0, state.naturalHeight),
      x1: clamp(Math.max(state.start.x, end.x), 0, state.naturalWidth),
      y1: clamp(Math.max(state.start.y, end.y), 0, state.naturalHeight),
    };
    drawCanvas();
    applyRoiCrop();
    return;
  }

  if (state.promptType !== "box") return;
  const box = {
    x0: clamp(Math.min(state.start.x, end.x), 0, state.naturalWidth),
    y0: clamp(Math.min(state.start.y, end.y), 0, state.naturalHeight),
    x1: clamp(Math.max(state.start.x, end.x), 0, state.naturalWidth),
    y1: clamp(Math.max(state.start.y, end.y), 0, state.naturalHeight),
  };
  state.box = box;
  drawCanvas();
});

canvas.addEventListener("click", (evt) => {
  if (state.roiMode || !state.naturalWidth) return;

  if (state.scaleMode) {
    const pt = getPointerPosition(evt);
    if (!state.scaleLine) {
      state.scaleLine = { x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y };
      drawCanvas();
      return;
    }
    state.scaleLine = { ...state.scaleLine, x1: pt.x, y1: pt.y };
    const dx = state.scaleLine.x1 - state.scaleLine.x0;
    const dy = state.scaleLine.y1 - state.scaleLine.y0;
    state.scalePixels = Math.sqrt(dx * dx + dy * dy);
    state.scaleMode = false;
    drawCanvas();
    setStatus(`Scale set: ${state.scalePixels.toFixed(2)} px for ${scaleLengthInput.value} ${scaleUnitInput.value || "units"}.`);
    return;
  }

  if (state.promptType !== "dot") return;
  const pt = getPointerPosition(evt);
  state.dot = { x: pt.x, y: pt.y };
  drawCanvas();
});

confidenceInput.addEventListener("input", () => {
  confValue.textContent = Number(confidenceInput.value).toFixed(2);
});

fileInput.addEventListener("change", handleFileChange);
loadUrlBtn.addEventListener("click", handleUrlLoad);
clearPromptBtn.addEventListener("click", resetPrompts);
roiStartBtn.addEventListener("click", startRoiMode);
roiClearBtn.addEventListener("click", clearRoi);
roiRevertBtn.addEventListener("click", revertImage);
scaleMarkBtn.addEventListener("click", startScaleMode);
scaleClearBtn.addEventListener("click", clearScale);

async function segment() {
  if (!state.naturalWidth || (!state.imageFile && !urlInput.value.trim())) {
    alert("Load an image first.");
    return;
  }

  const fd = new FormData();
  if (state.imageFile) {
    fd.append("image", state.imageFile);
  } else if (urlInput.value.trim()) {
    fd.append("image_url", urlInput.value.trim());
  }

  fd.append("prompt_type", state.promptType);
  fd.append("confidence", confidenceInput.value);

  if (!state.box) {
    alert("Draw a box on the canvas.");
    return;
  }
  const box = state.roi
    ? {
        x0: state.box.x0 - state.roi.x0,
        y0: state.box.y0 - state.roi.y0,
        x1: state.box.x1 - state.roi.x0,
        y1: state.box.y1 - state.roi.y0,
      }
    : state.box;
  box.x0 = clamp(box.x0, 0, state.naturalWidth);
  box.y0 = clamp(box.y0, 0, state.naturalHeight);
  box.x1 = clamp(box.x1, 0, state.naturalWidth);
  box.y1 = clamp(box.y1, 0, state.naturalHeight);
  fd.append("box_x0", box.x0);
  fd.append("box_y0", box.y0);
  fd.append("box_x1", box.x1);
  fd.append("box_y1", box.y1);
  fd.append("box_label", boxLabelInput.checked);

  if (state.roi) {
    fd.append("roi_x0", state.roi.x0);
    fd.append("roi_y0", state.roi.y0);
    fd.append("roi_x1", state.roi.x1);
    fd.append("roi_y1", state.roi.y1);
  }
  const scaleLen = parseFloat(scaleLengthInput.value);
  if (!Number.isNaN(scaleLen) && scaleLen > 0 && state.scalePixels) {
    fd.append("scale_length", scaleLen);
    fd.append("scale_px", state.scalePixels);
    if (scaleUnitInput.value) {
      fd.append("scale_unit", scaleUnitInput.value);
    }
  }
  fd.append("fill_masks", maskFillToggle ? maskFillToggle.checked : true);
  fd.append(
    "box_thickness",
    boxThicknessInput && boxThicknessInput.value ? boxThicknessInput.value : 3
  );
  fd.append("show_scores", scoreToggle.checked);
  const roiOnly = roiOnlyToggle.checked && !state.roiApplied;
  fd.append("roi_only", roiOnly);

  resultMeta.textContent = "Running segmentation...";
  const start = performance.now();
  try {
    const res = await fetch("/segment", {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      let detail = "Request failed";
      try {
        const err = await res.json();
        detail = err.detail || detail;
      } catch (e) {
        const txt = await res.text();
        detail = txt || detail;
      }
      throw new Error(detail);
    }
    const data = await res.json();
    const duration = ((performance.now() - start) / 1000).toFixed(2);
    resultImg.src = `data:image/png;base64,${data.overlay}`;
    resultMeta.textContent = `Done. Source ${data.width}x${data.height}`;
    maskCount.textContent = data.num_masks;
    scoreList.textContent =
      data.scores && data.scores.length
        ? data.scores.map((s) => s.toFixed(2)).join(", ")
        : "no masks";
    lengthList.textContent =
      data.lengths && data.lengths.length
        ? data.lengths.map((l) => l.toFixed(2)).join(", ") + (data.length_unit ? ` ${data.length_unit}` : "")
        : "-";
    runtimeEl.textContent = `${duration}s`;
    state.resultSegs = data.segments || [];
    state.resultNatural = {
      w: data.width,
      h: data.height,
      length_unit: data.length_unit,
      roi: data.roi || null,
    };
    syncResultOverlay();

    segmentsBody.innerHTML = "";
    if (data.segments && data.segments.length) {
      data.segments.forEach((seg) => {
        const tr = document.createElement("tr");
        
        // 1. Number
        const tdNum = document.createElement("td");
        tdNum.textContent = seg.number;
        tr.appendChild(tdNum);

        // 2. Thumbnail
        const tdImg = document.createElement("td");
        const img = document.createElement("img");
        img.src = `data:image/png;base64,${seg.image}`;
        img.className = "segment-thumb";
        tdImg.appendChild(img);
        tr.appendChild(tdImg);

        // 3. Normalized Line
        const tdNorm = document.createElement("td");
        if (seg.normalized_line) {
          const normImg = document.createElement("img");
          normImg.src = `data:image/png;base64,${seg.normalized_line}`;
          normImg.className = "line-thumb";
          tdNorm.appendChild(normImg);
        } else {
          tdNorm.textContent = "-";
        }
        tr.appendChild(tdNorm);

        // 4. Length
        const tdSize = document.createElement("td");
        const unit = seg.length_unit || data.length_unit || "";
        tdSize.textContent = `${seg.length.toFixed(2)}${unit ? " " + unit : ""}`;
        tr.appendChild(tdSize);

        // 5. Green
        const tdGreen = document.createElement("td");
        tdGreen.textContent = `${seg.green_ratio?.toFixed(1) ?? "0.0"}`;
        tr.appendChild(tdGreen);

        // 6. Red
        const tdRed = document.createElement("td");
        tdRed.textContent = `${seg.red_ratio?.toFixed(1) ?? "0.0"}`;
        tr.appendChild(tdRed);

        // 7. Confidence
        const tdConf = document.createElement("td");
        tdConf.textContent = seg.confidence.toFixed(2);
        tr.appendChild(tdConf);

        segmentsBody.appendChild(tr);
      });
    }
  } catch (err) {
    resultMeta.textContent = `Error: ${err.message}`;
  }
}

function exportCsv() {
  if (!state.resultSegs.length) {
    alert("No segments to export yet.");
    return;
  }
  const rows = [["number", "length", "green", "red", "confidence"]];
  const unit = state.resultNatural.length_unit ? ` ${state.resultNatural.length_unit}` : "";
  state.resultSegs.forEach((seg) => {
    rows.push([
      seg.number,
      `${seg.length.toFixed(2)}${unit}`.trim(),
      seg.green_ratio?.toFixed(1) ?? "0.0",
      seg.red_ratio?.toFixed(1) ?? "0.0",
      seg.confidence.toFixed(2),
    ]);
  });
  const csv = rows.map((r) => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "segments.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function syncResultOverlay() {
  if (!resultOverlay || !resultOverlayShell) return;
  const rect = resultOverlayShell.getBoundingClientRect();
  resultOverlay.width = rect.width;
  resultOverlay.height = rect.height;
  clearResultOverlay();
}

function clearResultOverlay() {
  if (!resultOverlay) return;
  const ctx2 = resultOverlay.getContext("2d");
  ctx2.clearRect(0, 0, resultOverlay.width, resultOverlay.height);
  resultTooltip.style.display = "none";

  // draw ROI outline if available
  const roi = state.resultNatural.roi;
  if (!roi || !state.resultNatural.w || !state.resultNatural.h) return;
  const rect = resultOverlayShell.getBoundingClientRect();
  const scaleX = state.resultNatural.w / rect.width;
  const scaleY = state.resultNatural.h / rect.height;
  const roiW = roi.x1 - roi.x0;
  const roiH = roi.y1 - roi.y0;
  const showingCrop = Math.abs(roiW - state.resultNatural.w) < 1e-3 && Math.abs(roiH - state.resultNatural.h) < 1e-3;
  if (showingCrop) return; // when displaying ROI-only, no need to redraw ROI box
  const offsetX = showingCrop ? roi.x0 : 0;
  const offsetY = showingCrop ? roi.y0 : 0;
  const drawX = (roi.x0 - offsetX) / scaleX;
  const drawY = (roi.y0 - offsetY) / scaleY;
  const drawW = roiW / scaleX;
  const drawH = roiH / scaleY;
  ctx2.strokeStyle = "#ffd166";
  ctx2.setLineDash([6, 4]);
  ctx2.lineWidth = 2;
  ctx2.strokeRect(drawX, drawY, drawW, drawH);
  ctx2.setLineDash([]);
}

function handleResultHover(evt) {
  if (!state.resultSegs.length || !state.resultNatural.w || !state.resultNatural.h) return;
  const rect = resultOverlayShell.getBoundingClientRect();
  const xDisplay = evt.clientX - rect.left;
  const yDisplay = evt.clientY - rect.top;
  if (xDisplay < 0 || yDisplay < 0 || xDisplay > rect.width || yDisplay > rect.height) {
    clearResultOverlay();
    return;
  }
  const scaleX = state.resultNatural.w / rect.width;
  const scaleY = state.resultNatural.h / rect.height;
  const roi = state.resultNatural.roi;
  let roiOffsetX = 0;
  let roiOffsetY = 0;
  if (roi) {
    const roiW = roi.x1 - roi.x0;
    const roiH = roi.y1 - roi.y0;
    const showingCrop = Math.abs(roiW - state.resultNatural.w) < 1e-3 && Math.abs(roiH - state.resultNatural.h) < 1e-3;
    if (showingCrop) {
      roiOffsetX = roi.x0;
      roiOffsetY = roi.y0;
    }
  }
  const x = xDisplay * scaleX + roiOffsetX;
  const y = yDisplay * scaleY + roiOffsetY;

  let hit = null;
  for (const seg of state.resultSegs) {
    const [x0, y0, x1, y1] = seg.box || [];
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
      hit = seg;
      break;
    }
  }

  clearResultOverlay();
  if (!hit) return;

  const ctx2 = resultOverlay.getContext("2d");
  const [x0, y0, x1, y1] = hit.box;
  const drawX0 = (x0 - roiOffsetX) / scaleX;
  const drawY0 = (y0 - roiOffsetY) / scaleY;
  const drawW = (x1 - x0) / scaleX;
  const drawH = (y1 - y0) / scaleY;
  ctx2.fillStyle = "rgba(255,255,255,0.15)";
  ctx2.fillRect(drawX0, drawY0, drawW, drawH);
  ctx2.strokeStyle = "#7df3ff";
  ctx2.lineWidth = 2;
  ctx2.strokeRect(drawX0, drawY0, drawW, drawH);

  resultTooltip.style.display = "block";
  const tooltipX = Math.min(rect.width - 160, Math.max(0, xDisplay + 10));
  const tooltipY = Math.min(rect.height - 60, Math.max(0, yDisplay + 10));
  resultTooltip.style.left = `${tooltipX}px`;
  resultTooltip.style.top = `${tooltipY}px`;
  const unitStr = hit.length_unit
    ? ` ${hit.length_unit}`
    : state.resultNatural.length_unit
      ? ` ${state.resultNatural.length_unit}`
      : "";
  resultTooltip.textContent = `#${hit.number}  length: ${hit.length.toFixed(2)}${unitStr}  conf: ${hit.confidence.toFixed(2)}  green: ${hit.green_ratio?.toFixed(1) ?? "0.0"}%  red: ${hit.red_ratio?.toFixed(1) ?? "0.0"}%`;
}

segmentBtn.addEventListener("click", segment);
resultImg.addEventListener("load", syncResultOverlay);
exportCsvBtn.addEventListener("click", exportCsv);
togglePromptRows();
setStatus("Load an image to start drawing.");

window.addEventListener("resize", syncResultOverlay);
if (resultOverlayShell) {
  resultOverlayShell.addEventListener("mousemove", handleResultHover);
  resultOverlayShell.addEventListener("mouseleave", clearResultOverlay);
}

// Drag and drop support
["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => e.preventDefault());
  document.body.addEventListener(evt, (e) => e.preventDefault());
});

dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) {
    loadLocalFile(file);
  }
});
