// ════════════════════════════════════════════════════════════
//  YT Summarizer Pro — app.js
// ════════════════════════════════════════════════════════════

// ── CONFIGURATION ───────────────────────────────────────────
// REPLACE THIS WITH YOUR HUGGING FACE SPACE URL
const API_BASE = "https://YOUR-HUGGINGFACE-SPACE-URL.hf.space"; 

// ── Element refs ────────────────────────────────────────────
const summarizeBtn   = document.getElementById("summarizeBtn");
const youtubeUrl     = document.getElementById("youtubeUrl");
const summaryMode    = document.getElementById("summaryMode");
const loadingSection = document.getElementById("loadingSection");
const loadingHeading = document.getElementById("loadingHeading");
const summarySection = document.getElementById("summarySection");
const summaryContent = document.getElementById("summaryContent");
const videoInfoBar   = document.getElementById("videoInfoBar");
const videoThumb     = document.getElementById("videoThumb");
const videoTitleText = document.getElementById("videoTitleText");
const videoChannelTx = document.getElementById("videoChannelText");
const wordCountBadge = document.getElementById("wordCountBadge");
const copyBtn        = document.getElementById("copyBtn");
const themeToggle    = document.getElementById("themeToggle");
const chatSection    = document.getElementById("chatSection");
const chatMessages   = document.getElementById("chatMessages");
const chatInput      = document.getElementById("chatInput");
const askBtn         = document.getElementById("askBtn");

let currentSummary = "";
let currentVideoId = "";

// ── Helpers ──────────────────────────────────────────────────
function esc(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function md2html(text) {
  return esc(text)
    .replace(/^### (.+)$/gm,   "<h3>$1</h3>")
    .replace(/^## (.+)$/gm,    "<h2>$1</h2>")
    .replace(/^# (.+)$/gm,     "<h2>$1</h2>")
    .replace(/^\*\*(.+)\*\*$/gm,"<h3>$1</h3>")
    .replace(/^\* (.+)$/gm,    "<li>$1</li>")
    .replace(/^- (.+)$/gm,     "<li>$1</li>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n{2,}/g,        "</p><p>")
    .replace(/\n/g,            "<br>");
}

function addChat(role, msg) {
  const d      = document.createElement("div");
  d.className  = `chat-bubble ${role}`;
  d.innerHTML  = md2html(msg);
  chatMessages.appendChild(d);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Step animation ───────────────────────────────────────────
let _stepTimer  = null;
let _whisperTmr = null;

function startSteps() {
  const ids = ["step1","step2","step3"];
  let i = 0;
  ids.forEach((id,j) => document.getElementById(id)?.classList.toggle("active", j===0));
  _stepTimer = setInterval(() => {
    i = (i+1) % ids.length;
    ids.forEach((id,j) => document.getElementById(id)?.classList.toggle("active", j===i));
  }, 2500);

  // After 15 sec warn that AssemblyAI is running
  _whisperTmr = setTimeout(() => {
    if (loadingHeading) {
      loadingHeading.innerHTML =
        "Transcribing audio…<br>" +
        "<small style='font-size:13px;font-weight:400;opacity:.7'>" +
        "No captions found — downloading & transcribing audio.<br>" +
        "This takes 1-2 minutes for long videos ☕" +
        "</small>";
    }
  }, 15000);
}

function stopSteps() {
  clearInterval(_stepTimer);
  clearTimeout(_whisperTmr);
  ["step1","step2","step3"].forEach(id =>
    document.getElementById(id)?.classList.remove("active")
  );
  if (loadingHeading) loadingHeading.textContent = "Processing Video...";
}

// ═══════════════════════════════════════════════════════════
//  SIDEBAR NAV  (data-page attribute based — 100% reliable)
// ═══════════════════════════════════════════════════════════

function switchPage(pageId) {
  document.querySelectorAll(".page-view").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
  document.getElementById("page-" + pageId)?.classList.add("active");
  document.querySelector(`.nav-item[data-page="${pageId}"]`)?.classList.add("active");
  if (pageId === "history")   renderHistory();
  if (pageId === "analytics") renderAnalytics();
}

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    const p = btn.getAttribute("data-page");
    if (p) switchPage(p);
  });
});

// ═══════════════════════════════════════════════════════════
//  HISTORY  (localStorage with thumbnails)
// ═══════════════════════════════════════════════════════════

function getHistory() {
  return JSON.parse(localStorage.getItem("vdpro_history") || "[]");
}

function saveHistory(entry) {
  const hist = getHistory();
  const idx  = hist.findIndex(h => h.video_id === entry.video_id);
  if (idx !== -1) hist.splice(idx, 1);
  hist.unshift(entry);
  if (hist.length > 30) hist.pop();
  localStorage.setItem("vdpro_history", JSON.stringify(hist));
}

function renderHistory() {
  const list = document.getElementById("history-list");
  if (!list) return;
  const hist = getHistory();
  if (!hist.length) {
    list.innerHTML = `
      <div class="empty-state">
        <i class="fa-solid fa-clock-rotate-left"></i>
        <p>No summaries yet. Go to Dashboard and summarize a video!</p>
      </div>`;
    return;
  }
  list.innerHTML = hist.map(h => `
    <div class="hist-card" onclick="loadFromHistory('${esc(h.video_id)}')">
      <img class="hist-thumb" src="${esc(h.thumbnail||'')}" alt=""
           onerror="this.style.display='none'"/>
      <div class="hist-info">
        <div class="hist-title">${esc(h.title)}</div>
        <div class="hist-meta">${esc(h.channel||'')} &nbsp;·&nbsp; ${esc(h.mode)} &nbsp;·&nbsp; ${esc(h.date)}</div>
      </div>
      <i class="fa-solid fa-arrow-right hist-arrow"></i>
    </div>`).join("");
}

function loadFromHistory(video_id) {
  const entry = getHistory().find(h => h.video_id === video_id);
  if (!entry) return;
  switchPage("dashboard");
  currentSummary = entry.summary;
  currentVideoId = entry.video_id;

  // Show video meta bar
  showVideoMeta(entry);

  summarySection.classList.remove("hidden");
  summaryContent.innerHTML = `<p>${md2html(entry.summary)}</p>`;
  chatSection.classList.remove("hidden");
  chatMessages.innerHTML = "";
  addChat("ai", "📂 Loaded from history. Re-summarize this video to refresh the RAG index for Q&A.");
  summarySection.scrollIntoView({ behavior: "smooth" });
}

// ═══════════════════════════════════════════════════════════
//  ANALYTICS
// ═══════════════════════════════════════════════════════════

function renderAnalytics() {
  const hist  = getHistory();
  const total = hist.length;
  const words = hist.reduce((s,h) => s + (h.summary||"").split(/\s+/).length, 0);
  const freq  = {};
  hist.forEach(h => { freq[h.mode] = (freq[h.mode]||0) + 1; });
  const fav   = Object.entries(freq).sort((a,b) => b[1]-a[1])[0];
  const week  = hist.filter(h => {
    try { return (Date.now() - new Date(h.date_iso||h.date).getTime()) < 7*86400000; }
    catch { return false; }
  }).length;

  animateCount("stat-total", total);
  animateCount("stat-words", words);
  animateCount("stat-week",  week);
  document.getElementById("stat-mode").textContent = fav ? fav[0] : "—";
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  let v = 0;
  const step = Math.max(1, Math.ceil(target/30));
  const t    = setInterval(() => {
    v = Math.min(v+step, target);
    el.textContent = v.toLocaleString();
    if (v >= target) clearInterval(t);
  }, 28);
}

// ═══════════════════════════════════════════════════════════
//  VIDEO META BAR
// ═══════════════════════════════════════════════════════════

function showVideoMeta(data) {
  if (data.thumbnail) {
    videoThumb.src = data.thumbnail;
    videoThumb.style.display = "block";
  } else {
    videoThumb.style.display = "none";
  }
  videoTitleText.textContent   = data.title   || "YouTube Video";
  videoChannelTx.textContent   = data.channel || "";
  wordCountBadge.textContent   = data.word_count
    ? `${data.word_count.toLocaleString()} words transcribed`
    : "";
  videoInfoBar.classList.remove("hidden");
}

function hideVideoMeta() {
  videoInfoBar.classList.add("hidden");
}

// ═══════════════════════════════════════════════════════════
//  SUMMARIZE
// ═══════════════════════════════════════════════════════════

async function summarizeVideo() {
  const url  = youtubeUrl.value.trim();
  const mode = summaryMode.value;

  if (!url) {
    alert("Please paste a YouTube URL first."); youtubeUrl.focus(); return;
  }

  switchPage("dashboard");
  summarizeBtn.disabled  = true;
  summarizeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing…';
  loadingSection.classList.remove("hidden");
  summarySection.classList.add("hidden");
  chatSection.classList.add("hidden");
  hideVideoMeta();
  summaryContent.innerHTML = "";
  chatMessages.innerHTML   = "";
  startSteps();

  try {
    // UPDATED WITH API_BASE
    const res = await fetch(`${API_BASE}/summarize`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url, mode }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    if (!data.success) {
      summarySection.classList.remove("hidden");
      summaryContent.innerHTML = `<p class="error-message">${esc(data.error)}</p>`;
      return;
    }

    currentSummary = data.summary  || "";
    currentVideoId = data.video_id || "";

    // Show video metadata bar
    showVideoMeta(data);

    summarySection.classList.remove("hidden");
    summaryContent.innerHTML = `<p>${md2html(currentSummary)}</p>`;
    chatSection.classList.remove("hidden");
    addChat("ai", "✅ RAG index ready! Ask anything about this video — I'll answer only from the actual transcript.");

    // Save to history
    saveHistory({
      video_id:  currentVideoId,
      title:     data.title     || "YouTube Video",
      channel:   data.channel   || "",
      thumbnail: data.thumbnail || "",
      summary:   currentSummary,
      mode:      summaryMode.options[summaryMode.selectedIndex].text,
      word_count: data.word_count || 0,
      date:      new Date().toLocaleDateString("en-IN"),
      date_iso:  new Date().toISOString(),
    });

    summarySection.scrollIntoView({ behavior: "smooth" });

  } catch (err) {
    console.error(err);
    summarySection.classList.remove("hidden");
    summaryContent.innerHTML = `<p class="error-message">Something went wrong. Check the console for details.</p>`;
  } finally {
    stopSteps();
    loadingSection.classList.add("hidden");
    summarizeBtn.disabled  = false;
    summarizeBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Generate Summary';
  }
}

// ═══════════════════════════════════════════════════════════
//  RAG Q&A
// ═══════════════════════════════════════════════════════════

async function askQuestion() {
  const question = chatInput.value.trim();
  if (!currentVideoId) { addChat("ai","Please summarize a video first."); return; }
  if (!question)        { chatInput.focus(); return; }

  chatInput.value = "";
  addChat("user", question);
  askBtn.disabled  = true;
  askBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

  try {
    // UPDATED WITH API_BASE
    const res  = await fetch(`${API_BASE}/ask`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ video_id: currentVideoId, question }),
    });
    const data = await res.json();
    addChat("ai", data.success ? data.answer : (data.error || "Could not answer."));
  } catch {
    addChat("ai", "Something went wrong. Try again.");
  } finally {
    askBtn.disabled  = false;
    askBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i>';
  }
}

// ═══════════════════════════════════════════════════════════
//  COPY
// ═══════════════════════════════════════════════════════════

copyBtn.addEventListener("click", async () => {
  if (!currentSummary) return;
  await navigator.clipboard.writeText(currentSummary);
  copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
  setTimeout(() => { copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>'; }, 1400);
});

// ═══════════════════════════════════════════════════════════
//  THEME
// ═══════════════════════════════════════════════════════════

function applyTheme(light) {
  document.body.classList.toggle("light", light);
  themeToggle.innerHTML = light
    ? '<i class="fa-solid fa-sun"></i>'
    : '<i class="fa-solid fa-moon"></i>';
}

themeToggle.addEventListener("click", () => {
  const isLight = !document.body.classList.contains("light");
  applyTheme(isLight);
  localStorage.setItem("vdpro_theme", isLight ? "light" : "dark");
});

// ═══════════════════════════════════════════════════════════
//  EVENT LISTENERS
// ═══════════════════════════════════════════════════════════

summarizeBtn.addEventListener("click", summarizeVideo);
askBtn.addEventListener("click", askQuestion);
youtubeUrl.addEventListener("keydown",  e => { if (e.key==="Enter") summarizeVideo(); });
chatInput.addEventListener("keydown",   e => { if (e.key==="Enter") askQuestion(); });

// ═══════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════

applyTheme(localStorage.getItem("vdpro_theme") === "light");