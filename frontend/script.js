// Single source of truth for the backend URL. Change this if your API
// runs somewhere other than the default `uvicorn backend.main:app --port 8002`.
const API_BASE_URL = "http://127.0.0.1:8002";

document.getElementById("apiBaseLabel").textContent = API_BASE_URL;

// ---------- Tabs ----------
const tabCompare = document.getElementById("tabCompare");
const tabGenerate = document.getElementById("tabGenerate");
const panelCompare = document.getElementById("panelCompare");
const panelGenerate = document.getElementById("panelGenerate");

function activateTab(which) {
  const isCompare = which === "compare";
  tabCompare.classList.toggle("active", isCompare);
  tabGenerate.classList.toggle("active", !isCompare);
  tabCompare.setAttribute("aria-selected", String(isCompare));
  tabGenerate.setAttribute("aria-selected", String(!isCompare));
  panelCompare.hidden = !isCompare;
  panelGenerate.hidden = isCompare;
}
tabCompare.addEventListener("click", () => activateTab("compare"));
tabGenerate.addEventListener("click", () => activateTab("generate"));

// ---------- Health check ----------
async function checkApiStatus() {
  const el = document.getElementById("apiStatus");
  try {
    const res = await fetch(`${API_BASE_URL}/`);
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    el.textContent = data.gemini_available
      ? "backend online · AI generation enabled"
      : "backend online · AI generation disabled (no GEMINI_API_KEY)";
    el.classList.remove("offline");
    el.classList.add("online");
  } catch (e) {
    el.textContent = "backend unreachable — start it with uvicorn";
    el.classList.remove("online");
    el.classList.add("offline");
  }
}
checkApiStatus();

// ---------- Shared helpers ----------
function showError(el, message) {
  el.textContent = message;
  el.hidden = false;
}
function hideError(el) {
  el.hidden = true;
  el.textContent = "";
}

async function postJSON(path, body) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (networkErr) {
    throw new Error(
      "Could not reach the backend. Is it running at " + API_BASE_URL + "?"
    );
  }

  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    // no JSON body
  }

  if (!res.ok) {
    const detail = data && data.detail ? data.detail : `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return data;
}

function renderResultCard(container, result) {
  const aPct = result.response_a_probability;
  const bPct = result.response_b_probability;
  const aWins = result.prediction === 0;

  container.innerHTML = `
    <div class="result-winner">
      Winner: <span class="winner-tag">${escapeHtml(result.winner)}</span>
    </div>
    <div class="balance-bar" role="img" aria-label="Option A ${aPct}%, Option B ${bPct}%">
      <div class="balance-a" style="width:${aPct}%"></div>
      <div class="balance-b" style="width:${bPct}%"></div>
    </div>
    <div class="result-rows">
      <div class="result-row ${aWins ? "is-winner" : ""}">
        <div class="result-label"><span class="dot dot-a"></span>Response A</div>
        <div class="result-pct">${aPct}%</div>
      </div>
      <div class="result-row ${!aWins ? "is-winner" : ""}">
        <div class="result-label"><span class="dot dot-b"></span>Response B</div>
        <div class="result-pct">${bPct}%</div>
      </div>
    </div>
  `;
  container.hidden = false;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------- MODE 1: Compare my options ----------
const compareForm = document.getElementById("compareForm");
const compareBtn = document.getElementById("compareBtn");
const compareError = document.getElementById("compareError");
const compareLoading = document.getElementById("compareLoading");
const compareResult = document.getElementById("compareResult");

compareForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError(compareError);
  compareResult.hidden = true;

  const prompt = document.getElementById("cPrompt").value.trim();
  const optionA = document.getElementById("cOptionA").value.trim();
  const optionB = document.getElementById("cOptionB").value.trim();

  if (!prompt || !optionA || !optionB) {
    showError(compareError, "Please fill in the situation and both options.");
    return;
  }

  compareBtn.disabled = true;
  compareLoading.hidden = false;
  try {
    const result = await postJSON("/compare", {
      prompt,
      option_a: optionA,
      option_b: optionB,
    });
    renderResultCard(compareResult, result);
  } catch (err) {
    showError(compareError, err.message);
  } finally {
    compareBtn.disabled = false;
    compareLoading.hidden = true;
  }
});

document.getElementById("compareReset").addEventListener("click", () => {
  compareForm.reset();
  hideError(compareError);
  compareResult.hidden = true;
});

// ---------- MODE 2: Let AI suggest options ----------
const generateForm = document.getElementById("generateForm");
const generateBtn = document.getElementById("generateBtn");
const generateAndCompareBtn = document.getElementById("generateAndCompareBtn");
const generateError = document.getElementById("generateError");
const generateLoading = document.getElementById("generateLoading");
const generatedOptions = document.getElementById("generatedOptions");
const genOptionA = document.getElementById("genOptionA");
const genOptionB = document.getElementById("genOptionB");
const compareGeneratedWrap = document.getElementById("compareGeneratedWrap");
const compareGeneratedBtn = document.getElementById("compareGeneratedBtn");
const generateCompareLoading = document.getElementById("generateCompareLoading");
const generateResult = document.getElementById("generateResult");

let lastGenerated = null; // { prompt, option_a, option_b }

function resetGeneratedState() {
  generatedOptions.hidden = true;
  compareGeneratedWrap.hidden = true;
  generateResult.hidden = true;
  lastGenerated = null;
}

async function runGenerate() {
  const prompt = document.getElementById("gPrompt").value.trim();
  if (!prompt) {
    showError(generateError, "Please describe a situation first.");
    return null;
  }
  hideError(generateError);
  resetGeneratedState();
  generateLoading.hidden = false;
  try {
    const data = await postJSON("/generate", { prompt });
    lastGenerated = data;
    genOptionA.textContent = data.option_a;
    genOptionB.textContent = data.option_b;
    generatedOptions.hidden = false;
    compareGeneratedWrap.hidden = false;
    return data;
  } catch (err) {
    showError(generateError, err.message);
    return null;
  } finally {
    generateLoading.hidden = true;
  }
}

generateForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  generateBtn.disabled = true;
  try {
    await runGenerate();
  } finally {
    generateBtn.disabled = false;
  }
});

compareGeneratedBtn.addEventListener("click", async () => {
  if (!lastGenerated) return;
  hideError(generateError);
  generateResult.hidden = true;
  compareGeneratedBtn.disabled = true;
  generateCompareLoading.hidden = false;
  try {
    const result = await postJSON("/compare", {
      prompt: lastGenerated.prompt,
      option_a: lastGenerated.option_a,
      option_b: lastGenerated.option_b,
    });
    renderResultCard(generateResult, result);
  } catch (err) {
    showError(generateError, err.message);
  } finally {
    compareGeneratedBtn.disabled = false;
    generateCompareLoading.hidden = true;
  }
});

generateAndCompareBtn.addEventListener("click", async () => {
  const prompt = document.getElementById("gPrompt").value.trim();
  if (!prompt) {
    showError(generateError, "Please describe a situation first.");
    return;
  }
  hideError(generateError);
  resetGeneratedState();
  generateAndCompareBtn.disabled = true;
  generateLoading.hidden = false;
  try {
    const data = await postJSON("/generate-and-compare", { prompt });
    lastGenerated = { prompt: data.prompt, option_a: data.option_a, option_b: data.option_b };
    genOptionA.textContent = data.option_a;
    genOptionB.textContent = data.option_b;
    generatedOptions.hidden = false;
    compareGeneratedWrap.hidden = false;
    renderResultCard(generateResult, data);
  } catch (err) {
    showError(generateError, err.message);
  } finally {
    generateAndCompareBtn.disabled = false;
    generateLoading.hidden = true;
  }
});

document.getElementById("generateReset").addEventListener("click", () => {
  generateForm.reset();
  hideError(generateError);
  resetGeneratedState();
});
