const API_URL = window.VOCALISE_API_URL || "http://127.0.0.1:5000/predict";
const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"];
const audioInput = document.getElementById("audioInput");
const dropZone = document.getElementById("dropZone");
const predictButton = document.getElementById("predictButton");
const selectedFile = document.getElementById("selectedFile");
const audioMeta = document.getElementById("audioMeta");
const audioPlayer = document.getElementById("audioPlayer");
const errorMessage = document.getElementById("errorMessage");
let selectedAudioFile = null;

const canvasElements = {
    waveform: document.getElementById("waveformCanvas"),
    emphasis: document.getElementById("emphasisCanvas"),
    spectrogram: document.getElementById("spectrogramCanvas"),
    mfcc: document.getElementById("mfccCanvas")
};

audioInput.addEventListener("change", () => {
    if (audioInput.files.length) selectAudioFile(audioInput.files[0]);
});

dropZone.addEventListener("dragover", event => {
    event.preventDefault();
    dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", event => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("audio/")) selectAudioFile(file);
    else showError("Please select an audio file.");
});

audioPlayer.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(audioPlayer.duration)) {
        audioMeta.textContent = `${formatDuration(audioPlayer.duration)} preview duration`;
    }
});

function selectAudioFile(file) {
    selectedAudioFile = file;
    selectedFile.textContent = `${file.name} · ${formatBytes(file.size)}`;
    audioPlayer.src = URL.createObjectURL(file);
    audioPlayer.hidden = false;
    audioMeta.textContent = "Loading preview...";
    resetResults();
    clearError();
}

predictButton.addEventListener("click", async () => {
    if (!selectedAudioFile) {
        showError("Select an audio recording before analyzing.");
        return;
    }
    clearError();
    predictButton.disabled = true;
    predictButton.querySelector("span").textContent = "Analyzing both models...";
    document.getElementById("agreementBadge").textContent = "Running analysis";
    document.getElementById("agreementBadge").className = "agreement-badge pending";
    const formData = new FormData();
    formData.append("audio", selectedAudioFile);
    try {
        const response = await fetch(API_URL, { method: "POST", body: formData });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "The analysis could not be completed.");
        renderModel("traditional", result.traditional, result.errors?.traditional);
        renderModel("distilhubert", result.distilhubert, result.errors?.distilhubert);
        renderComparison(result);
        if (result.analysis) renderAnalysis(result.analysis);
        if (result.errors && Object.keys(result.errors).length) {
            showError(Object.entries(result.errors).map(([name, message]) => `${name}: ${message}`).join(" "));
        }
        document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        showError(error.message);
        document.getElementById("agreementBadge").textContent = "Analysis failed";
    } finally {
        predictButton.disabled = false;
        predictButton.querySelector("span").textContent = "Analyze recording";
    }
});

function renderModel(modelName, result, error) {
    const prefix = modelName === "traditional" ? "traditional" : "distilhubert";
    const emotion = document.getElementById(`${prefix}Emotion`);
    const confidence = document.getElementById(`${prefix}Confidence`);
    const bar = document.getElementById(`${prefix}ConfidenceBar`);
    const time = document.getElementById(`${prefix}Time`);
    const container = document.getElementById(`${prefix}Probabilities`);
    if (!result) {
        emotion.textContent = "Unavailable";
        confidence.textContent = "—";
        bar.style.width = "0";
        time.textContent = "Error";
        container.innerHTML = `<span class="empty-state">${escapeHtml(error || "This model did not return a result.")}</span>`;
        return;
    }
    emotion.textContent = result.emotion;
    confidence.textContent = `${(result.confidence * 100).toFixed(2)}%`;
    bar.style.width = `${clamp(result.confidence * 100, 0, 100)}%`;
    time.textContent = `${result.inference_time_ms ?? "—"} ms`;
    renderProbabilities(container, result.probabilities, result.emotion);
}

function renderProbabilities(container, probabilities, predicted) {
    container.innerHTML = "";
    EMOTIONS.forEach(emotion => {
        const probability = Number(probabilities?.[emotion] ?? 0);
        const row = document.createElement("div");
        row.className = `probability-row${emotion === predicted ? " predicted" : ""}`;
        row.innerHTML = `<div class="probability-header"><span class="probability-name">${emotion}</span><span class="probability-value">${(probability * 100).toFixed(2)}%</span></div><div class="probability-track"><div class="probability-fill"></div></div>`;
        container.appendChild(row);
        requestAnimationFrame(() => row.querySelector(".probability-fill").style.width = `${clamp(probability * 100, 0, 100)}%`);
    });
}

function renderComparison(result) {
    const traditional = result.traditional;
    const distilhubert = result.distilhubert;
    const badge = document.getElementById("agreementBadge");
    const note = document.getElementById("comparisonNote");
    document.getElementById("tableTraditional").textContent = traditional ? `${traditional.emotion} (${(traditional.confidence * 100).toFixed(2)}%)` : "Unavailable";
    document.getElementById("tableDistilhubert").textContent = distilhubert ? `${distilhubert.emotion} (${(distilhubert.confidence * 100).toFixed(2)}%)` : "Unavailable";
    document.getElementById("traditionalExplanation").textContent = traditional ? `The prediction is based on handcrafted acoustic characteristics extracted from the speech signal. The current recording was classified as ${traditional.emotion}.` : "The traditional DSP model did not return a prediction.";
    document.getElementById("distilhubertExplanation").textContent = distilhubert ? `The prediction is based on learned speech representations from the pretrained model. The current recording was classified as ${distilhubert.emotion}.` : "The DistilHuBERT model did not return a prediction.";
    if (!traditional || !distilhubert) {
        badge.textContent = "Partial result";
        badge.className = "agreement-badge pending";
        note.hidden = false;
        note.textContent = "One model returned a result while the other was unavailable. The available prediction is still shown for inspection.";
        return;
    }
    const agreement = result.comparison?.agreement ?? traditional.emotion === distilhubert.emotion;
    badge.textContent = agreement ? "✓ Models agree" : "↔ Models disagree";
    badge.className = `agreement-badge ${agreement ? "agree" : "disagree"}`;
    note.hidden = false;
    note.textContent = agreement ? "Both models identify the same dominant emotion for this recording." : "The models produce different predictions for this recording, highlighting how handcrafted acoustic features and learned speech representations can diverge.";
}

function renderAnalysis(analysis) {
    document.getElementById("sampleRate").textContent = `${analysis.sample_rate} Hz`;
    document.getElementById("duration").textContent = `${Number(analysis.duration).toFixed(2)} s`;
    document.getElementById("frameLength").textContent = `${analysis.frame_length_ms} ms`;
    document.getElementById("hopLength").textContent = `${analysis.hop_length_ms} ms`;
    document.getElementById("mfccCount").textContent = analysis.mfcc_count;
    document.getElementById("waveformInfo").textContent = `${analysis.sample_rate} Hz`;
    drawWaveform(canvasElements.waveform, analysis.waveform);
    drawWaveform(canvasElements.emphasis, analysis.pre_emphasized_waveform);
    drawHeatmap(canvasElements.spectrogram, analysis.spectrogram);
    drawHeatmap(canvasElements.mfcc, analysis.mfcc);
}

function prepareCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { context, width: rect.width, height: rect.height };
}

function drawWaveform(canvas, data) {
    if (!data?.length) return;
    const { context, width, height } = prepareCanvas(canvas);
    context.fillStyle = "#252321";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "#46413d";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
    context.stroke();
    context.strokeStyle = "#e87532";
    context.lineWidth = 1.5;
    context.beginPath();
    data.forEach((value, index) => {
        const x = (index / Math.max(data.length - 1, 1)) * width;
        const y = height / 2 - value * height * .42;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
}

function drawHeatmap(canvas, matrix) {
    if (!matrix?.length || !matrix[0]?.length) return;
    const { context, width, height } = prepareCanvas(canvas);
    const rows = matrix.length;
    const cols = matrix[0].length;
    const cellWidth = width / cols;
    const cellHeight = height / rows;
    for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
            const value = clamp(Number(matrix[row][col]) || 0, 0, 1);
            context.fillStyle = `rgb(${Math.floor(35 + value * 210)}, ${Math.floor(28 + value * 105)}, ${Math.floor(23 + value * 50)})`;
            context.fillRect(col * cellWidth, row * cellHeight, Math.ceil(cellWidth) + 1, Math.ceil(cellHeight) + 1);
        }
    }
}

function resetResults() {
    ["traditional", "distilhubert"].forEach(prefix => {
        document.getElementById(`${prefix}Emotion`).textContent = "—";
        document.getElementById(`${prefix}Confidence`).textContent = "—";
        document.getElementById(`${prefix}ConfidenceBar`).style.width = "0";
        document.getElementById(`${prefix}Time`).textContent = "—";
        document.getElementById(`${prefix}Probabilities`).innerHTML = `<span class="empty-state">Run an analysis to see the model output.</span>`;
    });
    document.getElementById("agreementBadge").textContent = "Awaiting analysis";
    document.getElementById("agreementBadge").className = "agreement-badge pending";
    document.getElementById("comparisonNote").hidden = true;
    document.getElementById("tableTraditional").textContent = "—";
    document.getElementById("tableDistilhubert").textContent = "—";
}
function showError(message) { errorMessage.textContent = message; errorMessage.hidden = false; }
function clearError() { errorMessage.textContent = ""; errorMessage.hidden = true; }
function clamp(value, min, max) { return Math.min(Math.max(value, min), max); }
function formatBytes(bytes) { if (bytes < 1024) return `${bytes} B`; if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`; return `${(bytes / 1048576).toFixed(1)} MB`; }
function formatDuration(seconds) { const minutes = Math.floor(seconds / 60); return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`; }
function escapeHtml(value) { const element = document.createElement("div"); element.textContent = value; return element.innerHTML; }
