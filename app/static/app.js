// @ts-check
// Front-end logic for the flower classifier page.
// Types are written as JSDoc comments: VS Code checks them because of the line above,
// and the browser runs this file as plain JavaScript (no build step).

/**
 * @typedef {{ name: string, probability: number }} Prediction
 * @typedef {{ predictions: Prediction[] }} PredictResponse
 */

/**
 * Find an element by id and check that it has the expected type.
 * @template {HTMLElement} T
 * @param {string} id
 * @param {new () => T} type
 * @returns {T}
 */
function getElement(id, type) {
  const element = document.getElementById(id);
  if (!(element instanceof type)) {
    throw new Error(`Element #${id} is missing or is not a ${type.name}`);
  }
  return element;
}

const form = getElement("predict-form", HTMLFormElement);
const dropzone = getElement("dropzone", HTMLLabelElement);
const fileInput = getElement("file-input", HTMLInputElement);
const preview = getElement("preview", HTMLImageElement);
const topKSelect = getElement("top-k", HTMLSelectElement);
const submitButton = getElement("submit", HTMLButtonElement);
const message = getElement("message", HTMLParagraphElement);
const resultsEmpty = getElement("results-empty", HTMLParagraphElement);
const resultsList = getElement("results", HTMLOListElement);

/** URL of the image currently shown in the preview (so we can release it later). */
let previewUrl = "";

/**
 * Show a status or error message under the form.
 * @param {string} text
 * @param {boolean} isError
 */
function showMessage(text, isError) {
  message.textContent = text;
  message.classList.toggle("is-error", isError);
  message.hidden = false;
}

function hideMessage() {
  message.hidden = true;
  message.textContent = "";
}

/**
 * Show the selected image in the dropzone and enable the button.
 * @param {File} file
 */
function showPreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropzone.classList.add("has-image");
  submitButton.disabled = false;
}

/**
 * Check that the chosen file looks like an image and preview it.
 * The server does the real validation; this is only a quick, friendly check.
 * @param {File | undefined} file
 */
function handleFile(file) {
  hideMessage();
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showMessage("Please choose an image file (JPG or PNG).", true);
    return;
  }
  showPreview(file);
}

/**
 * Turn an error response from the API into a readable sentence.
 * FastAPI returns {"detail": "text"} for our own errors and
 * {"detail": [{"msg": "..."}]} for validation errors.
 * @param {Response} response
 * @returns {Promise<string>}
 */
async function readError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail) && data.detail.length > 0) return data.detail[0].msg;
  } catch {
    // The body was not JSON: fall through to the generic message
  }
  return `The request failed (error ${response.status}).`;
}

/**
 * Send the image to the API and return its predictions.
 * @param {File} file
 * @param {number} topK
 * @returns {Promise<Prediction[]>}
 */
async function classify(file, topK) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`/predict?top_k=${topK}`, { method: "POST", body });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  /** @type {PredictResponse} */
  const data = await response.json();
  return data.predictions;
}

/**
 * Draw the ranked list of predictions.
 * Text is inserted with textContent (never innerHTML), so a name can't inject HTML.
 * @param {Prediction[]} predictions
 */
function renderResults(predictions) {
  resultsList.replaceChildren();

  predictions.forEach((prediction, index) => {
    const percent = prediction.probability * 100;

    const name = document.createElement("span");
    name.className = "result__name";
    name.textContent = prediction.name;

    const value = document.createElement("span");
    value.className = "result__value";
    value.textContent = `${percent.toFixed(1)}%`;

    const head = document.createElement("div");
    head.className = "result__head";
    head.append(name, value);

    const bar = document.createElement("div");
    bar.className = "result__bar";
    bar.style.width = `${percent}%`;

    const track = document.createElement("div");
    track.className = "result__track";
    track.append(bar);

    const item = document.createElement("li");
    item.className = index === 0 ? "result result--top" : "result";
    item.append(head, track);
    resultsList.append(item);
  });

  resultsEmpty.hidden = true;
  resultsList.hidden = false;
}

/**
 * Switch the button between its normal and "working" state.
 * @param {boolean} isLoading
 */
function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Classifying…" : "Classify flower";
}

// ---------- Events ----------

fileInput.addEventListener("change", () => handleFile(fileInput.files?.[0]));

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault(); // required, otherwise the browser refuses the drop
  dropzone.classList.add("is-dragging");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragging"));

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("is-dragging");
  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    fileInput.files = files; // keep the input in sync with what was dropped
    handleFile(files[0]);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    showMessage("Choose an image first.", true);
    return;
  }

  hideMessage();
  setLoading(true);
  try {
    const predictions = await classify(file, Number(topKSelect.value));
    renderResults(predictions);
  } catch (error) {
    const text = error instanceof Error ? error.message : "Something went wrong.";
    showMessage(text, true);
  } finally {
    setLoading(false);
  }
});