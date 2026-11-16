const classes = ["plastic", "paper", "glass", "metal"];
const imageInput = document.querySelector("#image-input");
const preview = document.querySelector("#preview");
const classifyButton = document.querySelector("#classify-button");
const modelStatus = document.querySelector("#model-status");
const message = document.querySelector("#message");
const resultCard = document.querySelector("#result-card");
const prediction = document.querySelector("#prediction");
const confidence = document.querySelector("#confidence");
const probabilities = document.querySelector("#probabilities");
const modelMetrics = document.querySelector("#model-metrics");
const classMetrics = document.querySelector("#class-metrics");

let model;
let selectedImage;
let evaluationMetrics;

function setStatus(text, kind) {
  modelStatus.textContent = text;
  modelStatus.className = `status status-${kind}`;
}

async function loadModel() {
  try {
    const response = await fetch("/model/model.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`model.json request failed (${response.status})`);
    model = await tf.loadGraphModel("/model/model.json");
    setStatus("Model ready", "ready");
    message.textContent = "Add a clear image with one main object.";
  } catch (error) {
    setStatus("Model unavailable", "error");
    message.textContent = `The browser could not load the CNN model: ${error.message}`;
    console.error("Classifier model is unavailable:", error);
  }
}

function displayProbabilities(values) {
  probabilities.replaceChildren();
  values.forEach(({ label, value }) => {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `<span>${label}</span><div class="bar"><span style="width:${value * 100}%"></span></div><strong>${value.toFixed(4)} <small>(${Math.round(value * 100)}%)</small></strong>`;
    probabilities.append(row);
  });
}

function displayEvaluationMetrics(metrics) {
  evaluationMetrics = metrics;
  const summary = [
    ["Accuracy", metrics.accuracy],
    ["Macro precision", metrics["macro avg"]["precision"]],
    ["Macro recall", metrics["macro avg"]["recall"]],
    ["Macro F1-score", metrics["macro avg"]["f1-score"]]
  ];
  modelMetrics.replaceChildren();
  summary.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "metric-item";
    item.innerHTML = `<strong>${(value * 100).toFixed(2)}%</strong><span>${label}</span>`;
    modelMetrics.append(item);
  });
  classMetrics.replaceChildren();
  classes.forEach((label) => {
    const values = metrics[label];
    const row = document.createElement("div");
    row.className = "class-metric-row";
    row.innerHTML = `<strong>${label}</strong><span>Precision ${(values.precision * 100).toFixed(2)}%</span><span>Recall ${(values.recall * 100).toFixed(2)}%</span><span>F1 ${(values["f1-score"] * 100).toFixed(2)}%</span><span>n=${values.support}</span>`;
    classMetrics.append(row);
  });
}

async function loadEvaluationMetrics() {
  const response = await fetch("/model/metrics.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`metrics.json request failed (${response.status})`);
  displayEvaluationMetrics(await response.json());
}

async function classify() {
  if (!model || !selectedImage) return;
  classifyButton.disabled = true;
  classifyButton.textContent = "Classifying...";
  try {
    const tensor = tf.browser.fromPixels(selectedImage)
      .resizeBilinear([224, 224])
      .toFloat()
      .div(255)
      .expandDims(0);
    const output = model.execute(tensor);
    const probabilitiesTensor = tf.softmax(output);
    const values = Array.from(await probabilitiesTensor.data());
    tensor.dispose();
    output.dispose();
    probabilitiesTensor.dispose();
    const ranked = classes.map((label, index) => ({ label, value: values[index] || 0 }))
      .sort((a, b) => b.value - a.value);
    prediction.textContent = ranked[0].label;
    confidence.textContent = `${Math.round(ranked[0].value * 100)}%`;
    displayProbabilities(ranked);
    resultCard.hidden = false;
  } catch (error) {
    message.textContent = `Classification failed: ${error.message}`;
  } finally {
    classifyButton.disabled = false;
    classifyButton.textContent = "Classify waste";
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > 5 * 1024 * 1024) {
    message.textContent = "Choose a JPG, PNG, or WebP image smaller than 5 MB.";
    imageInput.value = "";
    return;
  }
  selectedImage = new Image();
  selectedImage.onload = () => {
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    classifyButton.disabled = !model;
    message.textContent = model ? "Ready to classify." : "The model is not installed yet.";
  };
  selectedImage.src = URL.createObjectURL(file);
});

classifyButton.addEventListener("click", classify);
loadModel();
loadEvaluationMetrics().catch((error) => {
  modelMetrics.textContent = `Evaluation metrics unavailable: ${error.message}`;
});
