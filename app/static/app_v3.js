/* app.js — DPCT Sepsis Detection Webapp */

const resultBox     = document.getElementById("resultBox");
const resultCards   = document.getElementById("resultCards");
const statusMessage = document.getElementById("statusMessage");
const modelStateBadge = document.getElementById("modelStateBadge");
const thresholdBadge  = document.getElementById("thresholdBadge");

const predictCsvForm   = document.getElementById("predictCsvForm");
const predictFileInput = document.getElementById("predictCsv");
const predictFolderInput = document.getElementById("predictFolder");
const manualForm       = document.getElementById("manualForm");

// ── Status helpers ─────────────────────────────────────────────────────────

function setStatusMessage(message, isError = false) {
    statusMessage.textContent = message;
    statusMessage.style.color = isError ? "#ef4444" : "#64748b";
}

function updateStatusStrip(status) {
    if (!status || !status.ready) {
        modelStateBadge.textContent = "Model Not Loaded";
        modelStateBadge.style.background = "#ef4444";
        thresholdBadge.textContent = "τ = N/A";
        return;
    }
    modelStateBadge.textContent = "DPCT Ready";
    modelStateBadge.style.background = "#22c55e";
    thresholdBadge.textContent = `τ = ${status.threshold}`;
}

async function refreshStatus() {
    try {
        const resp    = await fetch("/api/status");
        const payload = await resp.json();
        updateStatusStrip(payload);
    } catch (_) {}
}

// ── Result rendering ────────────────────────────────────────────────────────

function renderCards(predictions) {
    resultCards.innerHTML = "";
    resultCards.classList.remove("hidden");

    for (const p of predictions) {
        if (p.error) {
            const card = document.createElement("div");
            card.className = "result-card";
            card.innerHTML = `
                <div class="card-patient-id">${escHtml(p.patient_id)}</div>
                <div class="card-prob" style="color:#ef4444">Error</div>
                <div class="card-meta">${escHtml(p.error)}</div>`;
            resultCards.appendChild(card);
            continue;
        }

        const prob      = p.probability ?? 0;
        const pct       = (prob * 100).toFixed(1);
        const riskClass = p.risk_band ?? "Low";
        const label     = p.predicted_label === 1 ? "⚠ Sepsis Likely" : "✓ No Sepsis";
        const peakHour  = p.attention_peak_hour != null ? `Hour ${p.attention_peak_hour}` : "—";

        const flagsHtml = (p.flags || []).map(f =>
            `<span class="flag-pill">⚠ ${escHtml(f)}</span>`
        ).join("");

        const card = document.createElement("div");
        card.className = `result-card risk-${riskClass}`;
        
        // Pass context to RAG modal
        const contextObj = {
            patient_id: p.patient_id,
            probability: p.probability,
            risk_band: riskClass,
            peak_hour: p.attention_peak_hour,
            clinical_flags: p.flags || []
        };
        const contextStr = encodeURIComponent(JSON.stringify(contextObj));

        card.innerHTML = `
            <div class="card-patient-id">${escHtml(String(p.patient_id))}</div>
            <div class="card-prob">${pct}%</div>
            <span class="card-risk-label">${riskClass} Risk — ${label}</span>
            <div class="card-meta">
                <span>🎯 Threshold: ${window.__INITIAL_STATUS__?.threshold ?? "0.87"}</span>
                <span>🔍 DPCT Attention Peak: ${peakHour}</span>
            </div>
            ${flagsHtml ? `<div class="card-flags">${flagsHtml}</div>` : ""}
            <div class="card-actions">
                <button class="btn btn-secondary btn-sm" onclick="openRagModal('${contextStr}')">💬 Ask AI Assistant</button>
            </div>`;

        resultCards.appendChild(card);
    }
}

function renderJson(payload) {
    resultBox.textContent = JSON.stringify(payload, null, 2);
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

// ── API helpers ─────────────────────────────────────────────────────────────

async function parseApiResponse(response) {
    const payload = await response.json().catch(() => ({ detail: "Unable to read response." }));
    if (!response.ok) {
        if (Array.isArray(payload.detail)) {
            const msgs = payload.detail.map(e => {
                const loc = e.loc.length > 0 ? e.loc[e.loc.length-1] : 'field';
                return `${loc}: ${e.msg}`;
            });
            throw new Error("Validation Error - " + msgs.join(' | '));
        }
        throw new Error(payload.detail || payload.message || "Request failed.");
    }
    return payload;
}

function setButtonLoading(btn, loading) {
    if (loading) {
        btn.disabled = true;
        btn.dataset.originalText = btn.textContent;
        btn.innerHTML = `<span class="spinner"></span>Running…`;
    } else {
        btn.disabled = false;
        btn.textContent = btn.dataset.originalText || btn.textContent;
    }
}

function buildFolderFormData(folderInput, fieldName) {
    const files = Array.from(folderInput?.files || []);
    const formData = new FormData();
    let count = 0;
    for (const file of files) {
        if (!/\.(csv|psv|zip)$/i.test(file.name)) continue;
        formData.append(fieldName, file, file.webkitRelativePath || file.name);
        count++;
    }
    return { formData, count };
}

// ── Batch CSV prediction ────────────────────────────────────────────────────

async function handleCsvPredictSubmit(event) {
    event.preventDefault();
    const btn = document.getElementById("batchPredictBtn");
    let endpoint = "/api/predict/csv";
    let formData;

    const folderPayload = buildFolderFormData(predictFolderInput, "datasets");
    if (folderPayload.count > 0) {
        endpoint = "/api/predict/files";
        formData = folderPayload.formData;
        setStatusMessage(`Running batch prediction for ${folderPayload.count} file(s)…`);
    } else if (predictFileInput?.files?.length) {
        formData = new FormData();
        formData.append("dataset", predictFileInput.files[0]);
        setStatusMessage("Running batch prediction…");
    } else {
        setStatusMessage("Please select an inference file (.csv/.psv/.zip) or a folder.", true);
        return;
    }

    setButtonLoading(btn, true);
    resultCards.classList.add("hidden");

    try {
        const response = await fetch(endpoint, { method: "POST", body: formData });
        const payload  = await parseApiResponse(response);
        const preds    = payload.predictions || [];
        renderJson(payload);
        if (preds.length) renderCards(preds);
        setStatusMessage(`✓ Batch prediction completed — ${preds.length} patient(s).`);
    } catch (error) {
        setStatusMessage(`Batch prediction failed: ${error.message}`, true);
        renderJson({ error: error.message });
        resultCards.classList.add("hidden");
    } finally {
        setButtonLoading(btn, false);
    }
}

// ── Manual (bedside) prediction ─────────────────────────────────────────────

async function handleManualPredictSubmit(event) {
    event.preventDefault();
    const btn = document.getElementById("manualPredictBtn");
    setStatusMessage("Running single-patient DPCT inference…");
    setButtonLoading(btn, true);
    resultCards.classList.add("hidden");

    const rows = document.querySelectorAll(".measurement-row");
    const measurements = [];

    rows.forEach(row => {
        const rowData = {};
        const inputs = row.querySelectorAll("input");
        inputs.forEach(input => {
            const key = input.name;
            const value = input.value;
            if (key === "gender") { rowData[key] = value; return; }
            if (value === "")    { rowData[key] = null; return; }
            const num = Number(value);
            rowData[key] = Number.isNaN(num) ? value : num;
        });
        measurements.push(rowData);
    });

    const payload = { measurements };

    try {
        const response = await fetch("/api/predict/manual", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload),
        });
        const data = await parseApiResponse(response);
        const pred = data.prediction ?? data;
        renderJson(data);
        if (pred && pred.probability != null) renderCards([pred]);
        setStatusMessage("✓ Single-patient prediction completed.");
    } catch (error) {
        setStatusMessage(`Manual prediction failed: ${error.message}`, true);
        renderJson({ error: error.message });
        resultCards.classList.add("hidden");
    } finally {
        setButtonLoading(btn, false);
    }
}

// ── Event listeners ─────────────────────────────────────────────────────────

predictCsvForm.addEventListener("submit", handleCsvPredictSubmit);
manualForm.addEventListener("submit", handleManualPredictSubmit);

// ── Multi-Row Form Logic ────────────────────────────────────────────────────

function addMeasurementRow() {
    const container = document.getElementById("measurementsContainer");
    const rows = container.querySelectorAll(".measurement-row");
    const newIndex = rows.length;
    const lastRow = rows[rows.length - 1];
    
    const clone = lastRow.cloneNode(true);
    clone.dataset.index = newIndex;
    clone.querySelector(".hour-label").textContent = newIndex;
    clone.querySelector(".btn-remove-row").classList.remove("hidden");
    
    // Copy the actual typed values from the last row to the new row
    const lastInputs = lastRow.querySelectorAll("input");
    const cloneInputs = clone.querySelectorAll("input");
    cloneInputs.forEach((input, i) => {
        input.value = lastInputs[i].value;
    });
    
    container.appendChild(clone);
}

function removeMeasurementRow(btn) {
    btn.closest(".measurement-row").remove();
    // Update indices
    const rows = document.querySelectorAll(".measurement-row");
    rows.forEach((row, idx) => {
        row.dataset.index = idx;
        row.querySelector(".hour-label").textContent = idx;
    });
}

// ── RAG UI Logic ────────────────────────────────────────────────────────────

let currentPatientContext = null;
let chatHistory = [];

function openRagModal(contextStr) {
    currentPatientContext = JSON.parse(decodeURIComponent(contextStr));
    chatHistory = [];
    document.getElementById("ragModal").classList.remove("hidden");
    document.getElementById("chatLog").innerHTML = `<div class="chat-bubble system-bubble">Hello! I'm the DPCT Clinical Assistant. Ask me anything about this patient's prediction or sepsis guidelines.</div>`;
}

function closeRagModal() {
    document.getElementById("ragModal").classList.add("hidden");
}

document.getElementById("ragForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const queryInput = document.getElementById("ragQuery");
    const query = queryInput.value.trim();
    if (!query) return;

    queryInput.value = "";
    
    const chatLog = document.getElementById("chatLog");
    chatLog.innerHTML += `<div class="chat-bubble user-bubble">${escHtml(query)}</div>`;
    chatLog.scrollTop = chatLog.scrollHeight;
    
    const loadingId = "loading-" + Date.now();
    chatLog.innerHTML += `<div id="${loadingId}" class="chat-bubble system-bubble">Thinking...</div>`;
    chatLog.scrollTop = chatLog.scrollHeight;

    try {
        const response = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                patient_context: currentPatientContext,
                history: chatHistory
            })
        });
        const data = await parseApiResponse(response);
        document.getElementById(loadingId).remove();
        
        chatHistory.push({ role: "user", content: query });
        chatHistory.push({ role: "assistant", content: data.explanation });
        
        // Parse markdown style returns into basic html
        let text = escHtml(data.explanation).replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
        chatLog.innerHTML += `<div class="chat-bubble system-bubble">${text}</div>`;
    } catch (err) {
        document.getElementById(loadingId).remove();
        chatLog.innerHTML += `<div class="chat-bubble system-bubble" style="color:#ef4444">Error: ${err.message}</div>`;
    }
    chatLog.scrollTop = chatLog.scrollHeight;
});

if (window.__INITIAL_STATUS__) {
    updateStatusStrip(window.__INITIAL_STATUS__);
}

refreshStatus();
