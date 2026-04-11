const resultBox = document.getElementById("resultBox");
const statusMessage = document.getElementById("statusMessage");
const modelStateBadge = document.getElementById("modelStateBadge");
const thresholdBadge = document.getElementById("thresholdBadge");
const featureCountBadge = document.getElementById("featureCountBadge");

const trainForm = document.getElementById("trainForm");
const trainFileInput = document.getElementById("trainCsv");
const trainFolderInput = document.getElementById("trainFolder");
const predictCsvForm = document.getElementById("predictCsvForm");
const predictFileInput = document.getElementById("predictCsv");
const predictFolderInput = document.getElementById("predictFolder");
const manualForm = document.getElementById("manualForm");

function setStatusMessage(message, isError = false) {
    statusMessage.textContent = message;
    statusMessage.style.color = isError ? "#a61b1b" : "#334e68";
}

function renderJson(payload) {
    resultBox.textContent = JSON.stringify(payload, null, 2);
}

function updateStatusStrip(status) {
    if (!status || !status.ready) {
        modelStateBadge.textContent = "Model Not Trained";
        modelStateBadge.style.background = "#a61b1b";
        thresholdBadge.textContent = "Threshold: N/A";
        featureCountBadge.textContent = "Features: N/A";
        return;
    }

    modelStateBadge.textContent = "Model Ready";
    modelStateBadge.style.background = "#1a7f37";
    thresholdBadge.textContent = `Threshold: ${status.threshold}`;
    featureCountBadge.textContent = `Features: ${status.feature_count}`;
}

async function parseApiResponse(response) {
    const payload = await response.json().catch(() => ({ detail: "Unable to read response." }));
    if (!response.ok) {
        const message = payload.detail || payload.message || "Request failed.";
        throw new Error(message);
    }
    return payload;
}

async function refreshStatus() {
    try {
        const response = await fetch("/api/status");
        const payload = await parseApiResponse(response);
        updateStatusStrip(payload);
    } catch (error) {
        setStatusMessage(`Status check failed: ${error.message}`, true);
    }
}

function buildFolderFormData(folderInput, fieldName) {
    const files = Array.from(folderInput?.files || []);
    const formData = new FormData();
    const supportedPattern = /\.(csv|psv|zip)$/i;
    let count = 0;

    for (const file of files) {
        if (!supportedPattern.test(file.name)) {
            continue;
        }

        const relativeName = file.webkitRelativePath || file.name;
        formData.append(fieldName, file, relativeName);
        count += 1;
    }

    return { formData, count };
}

async function handleTrainSubmit(event) {
    event.preventDefault();
    let endpoint = "/api/train";
    let formData;

    const folderPayload = buildFolderFormData(trainFolderInput, "datasets");
    if (folderPayload.count > 0) {
        endpoint = "/api/train/files";
        formData = folderPayload.formData;
        setStatusMessage(`Training from ${folderPayload.count} folder file(s)... this can take a few minutes.`);
    } else if (trainFileInput?.files?.length) {
        formData = new FormData();
        formData.append("dataset", trainFileInput.files[0]);
        setStatusMessage("Training in progress... this can take a few minutes.");
    } else {
        const message = "Please select a training file (.csv/.psv/.zip) or a folder of .psv files.";
        setStatusMessage(message, true);
        renderJson({ error: message });
        return;
    }

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });
        const payload = await parseApiResponse(response);
        renderJson(payload);
        updateStatusStrip(payload.status);
        setStatusMessage("Training completed.");
    } catch (error) {
        setStatusMessage(`Training failed: ${error.message}`, true);
        renderJson({ error: error.message });
    }
}

async function handleCsvPredictSubmit(event) {
    event.preventDefault();
    let endpoint = "/api/predict/csv";
    let formData;

    const folderPayload = buildFolderFormData(predictFolderInput, "datasets");
    if (folderPayload.count > 0) {
        endpoint = "/api/predict/files";
        formData = folderPayload.formData;
        setStatusMessage(`Running folder batch prediction for ${folderPayload.count} file(s)...`);
    } else if (predictFileInput?.files?.length) {
        formData = new FormData();
        formData.append("dataset", predictFileInput.files[0]);
        setStatusMessage("Running batch prediction...");
    } else {
        const message = "Please select an inference file (.csv/.psv/.zip) or a folder of .psv files.";
        setStatusMessage(message, true);
        renderJson({ error: message });
        return;
    }

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });
        const payload = await parseApiResponse(response);
        renderJson(payload);
        setStatusMessage(`Batch prediction completed for ${payload.count} patient(s).`);
    } catch (error) {
        setStatusMessage(`Batch prediction failed: ${error.message}`, true);
        renderJson({ error: error.message });
    }
}

async function handleManualPredictSubmit(event) {
    event.preventDefault();
    setStatusMessage("Running single-patient prediction...");

    const formData = new FormData(manualForm);
    const payload = {};

    for (const [key, value] of formData.entries()) {
        if (key === "gender") {
            payload[key] = value;
            continue;
        }

        if (value === "") {
            payload[key] = null;
            continue;
        }

        const numericValue = Number(value);
        payload[key] = Number.isNaN(numericValue) ? value : numericValue;
    }

    try {
        const response = await fetch("/api/predict/manual", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
        const data = await parseApiResponse(response);
        renderJson(data);
        setStatusMessage("Single-patient prediction completed.");
    } catch (error) {
        setStatusMessage(`Manual prediction failed: ${error.message}`, true);
        renderJson({ error: error.message });
    }
}

trainForm.addEventListener("submit", handleTrainSubmit);
predictCsvForm.addEventListener("submit", handleCsvPredictSubmit);
manualForm.addEventListener("submit", handleManualPredictSubmit);

if (window.__INITIAL_STATUS__) {
    updateStatusStrip(window.__INITIAL_STATUS__);
}

refreshStatus();
