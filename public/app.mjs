const el = (id) => document.getElementById(id);
const modelNames = {
  S: "Supervised",
  same: "Same-condition KD",
  cross: "Cross-condition KD",
};
const sampleRate = 2;
const fileInput = el("file");
const modelSelect = el("model");
const opacity = el("opacity");
const canvas = el("canvas");
const context = canvas.getContext("2d");
const inputCanvas = document.createElement("canvas");
const inputContext = inputCanvas.getContext("2d", {willReadFrequently: true});
const overlayCanvas = document.createElement("canvas");
const overlayContext = overlayCanvas.getContext("2d");
let manifest;
let frames = [];
let selectedFrame = 0;
let playbackTimer;
let isBusy = false;

function setStatus(message, isError = false) {
  el("status").textContent = message;
  el("status").dataset.error = String(isError);
}

function setBusy(value) {
  isBusy = value;
  fileInput.disabled = value;
  modelSelect.disabled = value;
  opacity.disabled = value;
  el("sample").disabled = value;
  el("run").disabled = value || !fileInput.files.length;
  el("save-video").disabled = value;
  el("save-frame").disabled = value;
  el("seek").disabled = value;
  el("play").disabled = value;
}

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status}).`);
  return response.json();
}

function populateRows(id, rows) {
  const body = el(id);
  for (const row of rows) {
    const tr = body.insertRow();
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = row[0];
    tr.append(label);
    for (const value of row.slice(1)) tr.insertCell().textContent = value;
  }
}

function drawLegend() {
  for (const [index, name] of manifest.classes.entries()) {
    const item = document.createElement("span");
    const chip = document.createElement("i");
    chip.style.backgroundColor = `rgb(${manifest.palette[index].join(",")})`;
    item.append(chip, document.createTextNode(name));
    el("legend").append(item);
  }
}

async function loadResults() {
  const report = await getJson("results.json");
  const pct = (value) => (100 * value).toFixed(2);
  populateRows("weather", Object.entries(report.study).map(([key, row]) => [
    modelNames[key], pct(row.overall_miou),
    ...["fog", "night", "rain", "snow"].map((weather) => pct(row.per_condition_miou[weather])),
    pct(row.worst_condition_miou),
  ]));
  populateRows("visibility", Object.entries(report.visibility).map(([key, row]) => [
    modelNames[key], pct(row.clear), pct(row.ambiguous),
  ]));
  populateRows("benchmark", report.benchmark.map((row) => [
    row.resolution, `${row.median.toFixed(1)} / ${row.p95.toFixed(1)} ms`, `${row.seconds.toFixed(1)} s`,
  ]));
}

async function infer(source, modelKeys) {
  const {input} = manifest;
  inputContext.clearRect(0, 0, input.width, input.height);
  inputContext.drawImage(source, 0, 0, input.width, input.height);
  const image = inputContext.getImageData(0, 0, input.width, input.height);
  const params = new URLSearchParams();
  for (const key of modelKeys) params.append("model", key);
  const response = await fetch(`api/segment?${params}`, {
    method: "POST",
    headers: {"Content-Type": "application/octet-stream"},
    body: image.data,
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error || `Render returned HTTP ${response.status}. Please try again.`);
  }
  const maskBuffer = await response.arrayBuffer();
  const length = input.width * input.height;
  if (maskBuffer.byteLength !== length * modelKeys.length) {
    throw new Error("Render returned an unexpected mask size.");
  }
  const ratio = (source.videoWidth || source.naturalWidth) / (source.videoHeight || source.naturalHeight);
  return {
    pixels: image.data,
    masks: modelKeys.map((_, index) => new Uint8Array(maskBuffer, index * length, length)),
    titles: ["Input", ...modelKeys.map((key) => modelNames[key])],
    width: Math.max(1, Math.round(Math.min(640, 360 * ratio))),
    height: Math.max(1, Math.round(Math.min(360, 640 / ratio))),
  };
}

function showFrame(index) {
  const frame = frames[index];
  if (!frame) return;
  selectedFrame = index;
  el("seek").value = index;
  const {height, width, masks, pixels, titles} = frame;
  const header = 34;
  const strength = Number(opacity.value) / 100;
  const grid = manifest.input;
  if (canvas.width !== width * titles.length || canvas.height !== height + header) {
    canvas.width = width * titles.length;
    canvas.height = height + header;
  }
  context.fillStyle = "#0a1017";
  context.fillRect(0, 0, canvas.width, canvas.height);
  overlayContext.putImageData(new ImageData(pixels, grid.width, grid.height), 0, 0);
  context.drawImage(overlayCanvas, 0, header, width, height);
  for (const [panel, mask] of masks.entries()) {
    const rendered = new Uint8ClampedArray(pixels);
    for (let i = 0; i < mask.length; i++) {
      const color = manifest.palette[mask[i]];
      for (let c = 0; c < 3; c++) {
        const offset = 4 * i + c;
        rendered[offset] = pixels[offset] * (1 - strength) + color[c] * strength;
      }
    }
    overlayContext.putImageData(new ImageData(rendered, grid.width, grid.height), 0, 0);
    context.drawImage(overlayCanvas, width * (panel + 1), header, width, height);
  }
  context.font = "600 14px system-ui, sans-serif";
  context.fillStyle = "#edf3fa";
  for (const [panel, title] of titles.entries()) {
    context.fillText(title, panel * width + 8, 23, Math.max(1, width - 16));
  }
  canvas.hidden = false;
}

function stopPlayback() {
  clearInterval(playbackTimer);
  playbackTimer = undefined;
  el("play").textContent = "Play";
}

function revealExports() {
  el("exports").hidden = false;
  el("playback").hidden = frames.length < 2;
  el("save-video").hidden = frames.length < 2 || !recordingType();
  el("seek").max = Math.max(0, frames.length - 1);
}

function waitForMedia(media, event) {
  return new Promise((resolve, reject) => {
    const finish = (error) => {
      clearTimeout(timer);
      media.removeEventListener(event, success);
      media.removeEventListener("error", failed);
      if (error) reject(error);
      else resolve();
    };
    const success = () => finish();
    const failed = () => finish(new Error("This browser could not decode the selected media."));
    const timer = setTimeout(() => finish(new Error("Media decoding timed out.")), 20000);
    media.addEventListener(event, success, {once: true});
    media.addEventListener("error", failed, {once: true});
  });
}

async function seek(video, seconds) {
  if (Math.abs(video.currentTime - seconds) < .001) return;
  const seeked = waitForMedia(video, "seeked");
  video.currentTime = seconds;
  await seeked;
}

async function segment({url, videoFile = false, sample = false}) {
  if (isBusy) return;
  stopPlayback();
  setBusy(true);
  const keys = modelSelect.value === "compare" ? ["S", "cross"] : [modelSelect.value];
  let video;
  try {
    setStatus("Preparing media and loading model…");
    let image;
    if (videoFile || sample) {
      video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.preload = "auto";
      const loaded = waitForMedia(video, "loadeddata");
      video.src = url;
      await loaded;
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        throw new Error("This recording does not have a readable duration.");
      }
    } else {
      image = new Image();
      image.src = url;
      await image.decode();
    }
    const duration = videoFile ? Math.min(video.duration, 12) : 0;
    const count = videoFile ? Math.max(1, Math.ceil(duration * sampleRate)) : 1;
    const results = [];
    const progress = el("progress");
    progress.max = count;
    progress.value = 0;
    progress.hidden = false;
    for (let i = 0; i < count; i++) {
      if (video) await seek(video, sample ? 7 : (i + .5) * duration / count);
      setStatus(`Segmenting frame ${i + 1} of ${count}…`);
      results.push(await infer(video || image, keys));
      progress.value = i + 1;
    }
    frames = results;
    showFrame(0);
    revealExports();
    setStatus(videoFile ? `${count} frames sampled from ${duration.toFixed(1)} seconds.` : "Segmentation ready.");
  } catch (error) {
    setStatus(error.message || "Segmentation failed.", true);
  } finally {
    video?.removeAttribute("src");
    video?.load();
    el("progress").hidden = true;
    setBusy(false);
  }
}

function download(blob, filename) {
  if (!blob) return;
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function recordingType() {
  if (!window.MediaRecorder || !canvas.captureStream) return "";
  return ["video/webm;codecs=vp8", "video/webm", "video/mp4"]
    .find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function exportVideo() {
  const type = recordingType();
  if (!type || frames.length < 2 || isBusy) return;
  stopPlayback();
  setBusy(true);
  const stream = canvas.captureStream(sampleRate);
  try {
    const recorder = new MediaRecorder(stream, {mimeType: type});
    const parts = [];
    const finished = new Promise((resolve, reject) => {
      recorder.ondataavailable = ({data}) => { if (data.size) parts.push(data); };
      recorder.onstop = () => resolve(new Blob(parts, {type}));
      recorder.onerror = ({error}) => reject(error);
    });
    // Observe recorder errors even while the timed render loop is active.
    finished.catch(() => {});
    showFrame(0);
    recorder.start();
    for (let i = 0; i < frames.length && recorder.state === "recording"; i++) {
      showFrame(i);
      setStatus(`Encoding frame ${i + 1} of ${frames.length}…`);
      await new Promise((resolve) => setTimeout(resolve, 1000 / sampleRate));
    }
    if (recorder.state === "recording") recorder.stop();
    const video = await finished;
    if (!video.size) throw new Error("The browser returned an empty video.");
    download(video, type.startsWith("video/mp4") ? "segmentation.mp4" : "segmentation.webm");
    setStatus("Video ready.");
  } catch (error) {
    setStatus(error.message || "The browser could not encode the clip.", true);
  } finally {
    stream.getTracks().forEach((track) => track.stop());
    setBusy(false);
  }
}

async function initialize() {
  manifest = await getJson("models.json");
  const {width, height} = manifest.input;
  inputCanvas.width = overlayCanvas.width = width;
  inputCanvas.height = overlayCanvas.height = height;
  drawLegend();
  fileInput.addEventListener("change", () => {
    el("run").disabled = !fileInput.files.length;
  });
  opacity.addEventListener("input", () => {
    el("opacity-value").textContent = `${opacity.value}%`;
    showFrame(selectedFrame);
  });
  el("run").addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    if (file.size > 100 * 1024 * 1024) {
      setStatus("Choose a file smaller than 100 MB.", true);
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      await segment({url, videoFile: file.type.startsWith("video/") || /\.(mp4|webm|mov|m4v|ogv)$/i.test(file.name)});
    } finally {
      URL.revokeObjectURL(url);
    }
  });
  el("sample").addEventListener("click", () => segment({url: "media/dashcam.mp4", sample: true}));
  el("seek").addEventListener("input", () => {
    stopPlayback();
    showFrame(Number(el("seek").value));
  });
  el("play").addEventListener("click", () => {
    if (playbackTimer) stopPlayback();
    else {
      el("play").textContent = "Pause";
      playbackTimer = setInterval(() => showFrame((selectedFrame + 1) % frames.length), 1000 / sampleRate);
    }
  });
  el("save-frame").addEventListener("click", () => canvas.toBlob((image) => download(image, "segmentation.png")));
  el("save-video").addEventListener("click", exportVideo);
  await loadResults();
}

initialize().catch((error) => setStatus(error.message, true));
