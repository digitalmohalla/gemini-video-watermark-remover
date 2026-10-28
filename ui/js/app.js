/**
 * Gemini & Veo Video Watermark Remover - Frontend Controller
 * Connects to Flask backend, manages Windows file pickers,
 * listens to SSE real-time stream, and updates batch progress dynamically.
 */

// State
let queueItems = [];
let batchState = {};
let systemInfo = {};

// DOM Elements
const dropzone = document.getElementById("dropzone");
const dropzoneTitle = document.getElementById("dropzone-title");
const dropzoneSubtitle = document.getElementById("dropzone-subtitle");
const btnPickFiles = document.getElementById("btn-pick-files");
const btnPickFolder = document.getElementById("btn-pick-folder");
const btnStartBatch = document.getElementById("btn-start-batch");
const btnPauseBatch = document.getElementById("btn-pause-batch");
const btnCancelBatch = document.getElementById("btn-cancel-batch");
const btnClearQueue = document.getElementById("btn-clear-queue");
const btnDownloadZip = document.getElementById("btn-download-zip");
const btnRetryFailed = document.getElementById("btn-retry-failed");
const failedRetryCount = document.getElementById("failed-retry-count");
const queueList = document.getElementById("queue-list");
const queueCount = document.getElementById("queue-count");

// Selection Banner Elements
const selectionBanner = document.getElementById("selection-banner");
const selBadgeCount = document.getElementById("sel-badge-count");
const selTitleText = document.getElementById("sel-title-text");
const selSubtitleText = document.getElementById("sel-subtitle-text");
const btnStartBanner = document.getElementById("btn-start-banner");
const toastContainer = document.getElementById("toast-container");

// Stats Elements
const statTotal = document.getElementById("stat-total");
const statCompleted = document.getElementById("stat-completed");
const statPending = document.getElementById("stat-pending");
const statFailed = document.getElementById("stat-failed");
const batchProgressBar = document.getElementById("batch-progress-bar");
const batchProgressText = document.getElementById("batch-progress-text");

// Settings Elements
const destType = document.getElementById("dest-type");
const customDestGroup = document.getElementById("custom-dest-group");
const customDestInput = document.getElementById("custom-dest-input");
const btnBrowseDest = document.getElementById("btn-browse-dest");
const engineSelect = document.getElementById("engine-select");
const markSelect = document.getElementById("mark-select");
const denoiseSelect = document.getElementById("denoise-select");
const suffixInput = document.getElementById("suffix-input");
const subfolderCheck = document.getElementById("subfolder-check");
const autoZipCheck = document.getElementById("auto-zip-check");
const logsTerminal = document.getElementById("logs-terminal");
const gpuNameEl = document.getElementById("gpu-name");

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  fetchSystemInfo();
  setupEventListeners();
  connectSSE();
});

// Toast notifications
function showToast(text, type = "info") {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<span>${text}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 350);
  }, 4000);
}

// Fetch system details
async function fetchSystemInfo() {
  try {
    const res = await fetch("/api/system-info");
    systemInfo = await res.json();
    if (systemInfo.gpus && systemInfo.gpus.length > 0) {
      gpuNameEl.innerText = `${systemInfo.gpus[0]} (Vulkan Accelerated)`;
    } else {
      gpuNameEl.innerText = "CPU Mode";
    }
    addLog(`[System] Detected hardware: ${systemInfo.gpus.join(", ")} | ${systemInfo.cpu_count} CPU threads`);
  } catch (err) {
    console.error("Failed to load system info", err);
  }
}

// Connect Server-Sent Events (SSE)
function connectSSE() {
  const evtSource = new EventSource("/api/progress-stream");

  evtSource.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleSSEMessage(msg);
    } catch (err) {
      // Keepalive heartbeat
    }
  };

  evtSource.onerror = () => {
    console.warn("SSE connection interrupted, retrying in 3s...");
  };
}

function handleSSEMessage(msg) {
  const { type, data } = msg;

  if (type === "init") {
    batchState = data;
    queueItems = data.items || [];
    renderQueue();
    updateBatchStats();
  } else if (type === "queue_updated") {
    batchState = data;
    const newItems = data.items || [];
    if (newItems.length !== queueItems.length) {
      queueItems = newItems;
      renderQueue();
    } else {
      queueItems = newItems;
      for (const it of newItems) {
        updateCardItem(it);
      }
    }
    updateBatchStats();
  } else if (type === "item_progress") {
    updateCardProgress(data);
    updateBatchStats();
  } else if (type === "item_updated") {
    updateCardItem(data);
    updateBatchStats();
  } else if (type === "thumbnail_ready") {
    updateThumbnail(data.id, data.thumbnail);
  } else if (type === "log") {
    addLog(data);
  } else if (type === "batch_started") {
    batchState.is_running = true;
    btnStartBatch.style.display = "none";
    if (btnStartBanner) btnStartBanner.style.display = "none";
    btnPauseBatch.style.display = "inline-flex";
    btnCancelBatch.style.display = "inline-flex";
    updateBatchStats();
  } else if (type === "batch_paused") {
    batchState.is_running = false;
    btnStartBatch.style.display = "inline-flex";
    btnStartBatch.innerHTML = `<svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg> Resume`;
    if (btnStartBanner) btnStartBanner.style.display = "inline-flex";
    btnPauseBatch.style.display = "none";
    updateBatchStats();
  } else if (type === "batch_finished" || type === "batch_cancelled") {
    batchState.is_running = false;
    btnStartBatch.style.display = "inline-flex";
    btnStartBatch.innerHTML = `<svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg> Start Batch`;
    if (btnStartBanner) btnStartBanner.style.display = "inline-flex";
    btnPauseBatch.style.display = "none";
    btnCancelBatch.style.display = "none";
    updateBatchStats();
  } else if (type === "zip_created") {
    btnDownloadZip.style.display = "inline-flex";
    showToast("📦 ZIP Archive created successfully!", "success");
    addLog(`[Success] ZIP archive created. You can download or view it.`);
  }
}

// Event Listeners
function setupEventListeners() {
  // Windows file picker dialog
  btnPickFiles.addEventListener("click", async () => {
    addLog("[UI] Opening Windows file picker...");
    try {
      const res = await fetch("/api/pick-files", { method: "POST" });
      const data = await res.json();
      if (data.status === "success" && data.count > 0) {
        showToast(`✅ Added ${data.count} video(s) to queue!`, "success");
        addLog(`[UI] Added ${data.count} video(s) to queue.`);
      }
    } catch (err) {
      console.error(err);
    }
  });

  // Windows folder picker dialog
  btnPickFolder.addEventListener("click", async () => {
    addLog("[UI] Opening Windows folder picker...");
    try {
      const res = await fetch("/api/pick-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "input" }),
      });
      const data = await res.json();
      if (data.status === "success") {
        showToast(`📁 Found ${data.found_count} video(s) in folder!`, "success");
        addLog(`[UI] Scanned folder: ${data.folder} (${data.found_count} videos found)`);
      }
    } catch (err) {
      console.error(err);
    }
  });

  // Destination browse button
  btnBrowseDest.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/pick-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "output" }),
      });
      const data = await res.json();
      if (data.status === "success" && data.folder) {
        customDestInput.value = data.folder;
        saveSettings();
      }
    } catch (err) {
      console.error(err);
    }
  });

  // Drag and Drop
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const paths = [];
      for (let i = 0; i < files.length; i++) {
        if (files[i].path) {
          paths.push(files[i].path);
        }
      }
      if (paths.length > 0) {
        const res = await fetch("/api/add-files", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_paths: paths }),
        });
        const data = await res.json();
        showToast(`✅ Dropped ${data.added_count} video(s) into queue!`, "success");
      } else {
        btnPickFiles.click();
      }
    }
  });

  // Settings change listeners
  destType.addEventListener("change", () => {
    customDestGroup.style.display = destType.value === "custom" ? "flex" : "none";
    saveSettings();
  });
  engineSelect.addEventListener("change", saveSettings);
  markSelect.addEventListener("change", saveSettings);
  denoiseSelect.addEventListener("change", saveSettings);
  suffixInput.addEventListener("input", saveSettings);
  subfolderCheck.addEventListener("change", saveSettings);
  autoZipCheck.addEventListener("change", saveSettings);

  // Settings collapsible toggle
  document.getElementById("settings-toggle").addEventListener("click", () => {
    const body = document.getElementById("settings-body");
    const arrow = document.getElementById("settings-arrow");
    const isHidden = body.style.display === "none";
    body.style.display = isHidden ? "grid" : "none";
    arrow.innerText = isHidden ? "▼" : "▶";
  });

  // Logs collapsible toggle
  document.getElementById("logs-toggle").addEventListener("click", () => {
    const content = document.getElementById("logs-terminal");
    const arrow = document.getElementById("logs-arrow");
    const isHidden = content.style.display === "none";
    content.style.display = isHidden ? "block" : "none";
    arrow.innerText = isHidden ? "▼" : "▶";
  });

  // Batch action buttons
  btnStartBatch.addEventListener("click", async () => {
    await fetch("/api/start", { method: "POST" });
  });

  if (btnStartBanner) {
    btnStartBanner.addEventListener("click", async () => {
      await fetch("/api/start", { method: "POST" });
    });
  }

  btnPauseBatch.addEventListener("click", async () => {
    await fetch("/api/pause", { method: "POST" });
  });

  btnCancelBatch.addEventListener("click", async () => {
    await fetch("/api/cancel", { method: "POST" });
  });

  btnClearQueue.addEventListener("click", async () => {
    await fetch("/api/clear-queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ completed_only: false }),
    });
  });

  if (btnRetryFailed) {
    btnRetryFailed.addEventListener("click", async () => {
      showToast("🔄 Retrying all failed videos...", "info");
      await fetch("/api/retry-failed", { method: "POST" });
    });
  }

  btnDownloadZip.addEventListener("click", () => {
    window.location.href = "/api/download-zip";
  });
}

// Save Settings to Backend
async function saveSettings() {
  const payload = {
    destination_type: destType.value,
    custom_destination: customDestInput.value,
    engine: engineSelect.value,
    mark_type: markSelect.value,
    denoise_mode: denoiseSelect.value,
    output_suffix: suffixInput.value,
    use_subfolder: subfolderCheck.checked,
    auto_zip: autoZipCheck.checked,
  };
  try {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    console.error(err);
  }
}

// Render Queue Cards
function renderQueue() {
  queueCount.innerText = queueItems.length;

  if (queueItems.length === 0) {
    queueList.innerHTML = `
      <div class="empty-queue glass">
        <p>No videos added yet. Click <strong>"Select Videos"</strong> or drag video files above.</p>
      </div>`;
    return;
  }

  let html = "";
  for (const item of queueItems) {
    const thumbHtml = item.has_thumbnail
      ? `<img src="/api/thumbnail/${item.id}" alt="${item.filename}">`
      : `<span class="card-thumb-placeholder">Video</span>`;

    html += `
      <div class="queue-card glass" id="card-${item.id}">
        <div class="card-thumb" id="thumb-${item.id}">
          ${thumbHtml}
        </div>
        <div class="card-info">
          <div class="card-title-row">
            <span class="card-filename" title="${item.input_path}">${item.filename}</span>
            <span class="status-badge status-${item.status}" id="badge-${item.id}">${item.status}</span>
          </div>
          <div class="card-meta">
            <span>📐 ${item.resolution}</span>
            <span class="badge-tag">${item.aspect_ratio || '16:9'}</span>
            ${item.is_vertical ? `<span class="badge-tag">📱 9:16 Vertical</span>` : ''}
            ${item.is_square ? `<span class="badge-tag">⏹ 1:1 Square</span>` : ''}
            ${item.has_black_bars ? `<span class="badge-tag badge-bars">⬛ Letterbox (${item.content_aspect_ratio})</span>` : ''}
            ${item.duration === 'Image' || item.total_frames === 1 
              ? `<span>🖼 Image</span>` 
              : `<span>⏱ ${item.duration}</span><span>🎞 ${item.total_frames} frames</span>`}
            <span>💾 ${item.size_mb} MB</span>
          </div>
          <div class="card-wm-detected">
            ${item.watermark_found 
              ? `<span>🎯 <strong>${item.watermark_size}</strong> Diamond @ <code>${item.watermark_region}</code> (${Math.round(item.measured_alpha * 100)}% α) · <em>Extracted from media</em></span>`
              : `<span>🔍 Auto multi-scale snap search active</span>`}
          </div>
          <div class="card-progress-bar">
            <div class="card-progress-fill" id="pfill-${item.id}" style="width: ${item.progress_pct}%;"></div>
          </div>
          <div class="card-status-msg" id="msg-${item.id}">${item.status_msg}</div>
        </div>
        <div class="card-actions" id="actions-${item.id}">
          ${item.status === "error" ? `
            <button class="btn btn-secondary btn-sm" onclick="retryItem('${item.id}')" title="Retry processing this video">
              🔄 Retry
            </button>` : ""}
          ${item.output_path && item.status === "completed" ? `
            <button class="btn btn-secondary btn-sm" onclick="openExplorer('${escapePath(item.output_path)}')">
              📁 Open Folder
            </button>` : ""}
          <button class="btn btn-danger btn-sm" onclick="removeItem('${item.id}')">
            ✖
          </button>
        </div>
      </div>
    `;
  }
  queueList.innerHTML = html;
}

// Real-time Card Progress Update
function updateCardProgress(item) {
  // Update in-memory queue item
  const idx = queueItems.findIndex(i => i.id === item.id);
  if (idx !== -1) {
    queueItems[idx] = { ...queueItems[idx], ...item };
  }

  const pfill = document.getElementById(`pfill-${item.id}`);
  const msg = document.getElementById(`msg-${item.id}`);
  const badge = document.getElementById(`badge-${item.id}`);

  if (pfill) pfill.style.width = `${item.progress_pct}%`;
  if (msg) msg.innerText = `${item.status_msg} ${item.current_fps > 0 ? `· ${item.current_fps} fps` : ""}`;
  if (badge && badge.innerText.toLowerCase() !== "processing") {
    badge.className = "status-badge status-processing";
    badge.innerText = "processing";
  }
}

// Card status update
function updateCardItem(item) {
  // Update in-memory queue item
  const idx = queueItems.findIndex(i => i.id === item.id);
  if (idx !== -1) {
    queueItems[idx] = { ...queueItems[idx], ...item };
  }

  const card = document.getElementById(`card-${item.id}`);
  if (card) {
    const badge = document.getElementById(`badge-${item.id}`);
    const msg = document.getElementById(`msg-${item.id}`);
    const pfill = document.getElementById(`pfill-${item.id}`);
    const actions = document.getElementById(`actions-${item.id}`);

    if (badge) {
      badge.className = `status-badge status-${item.status}`;
      badge.innerText = item.status;
    }
    if (msg) msg.innerText = item.status_msg;
    if (pfill) pfill.style.width = `${item.progress_pct}%`;

    // Re-render actions for completed or error states
    if (actions) {
      let actionsHtml = "";
      if (item.status === "error") {
        actionsHtml += `
          <button class="btn btn-secondary btn-sm" onclick="retryItem('${item.id}')" title="Retry processing this video">
            🔄 Retry
          </button>`;
      }
      if (item.status === "completed" && item.output_path) {
        actionsHtml += `
          <button class="btn btn-secondary btn-sm" onclick="openExplorer('${escapePath(item.output_path)}')">
            📁 Open Folder
          </button>`;
      }
      actionsHtml += `
        <button class="btn btn-danger btn-sm" onclick="removeItem('${item.id}')">
          ✖
        </button>`;
      actions.innerHTML = actionsHtml;
    }
  }
}

function updateThumbnail(id, b64) {
  const thumb = document.getElementById(`thumb-${id}`);
  if (thumb) {
    thumb.innerHTML = `<img src="data:image/jpeg;base64,${b64}">`;
  }
}

// Update Batch Progress and Counts
function updateBatchStats(stateOverride) {
  if (stateOverride && typeof stateOverride === "object") {
    batchState = { ...batchState, ...stateOverride };
  }

  // Always compute directly from the live queueItems array!
  const total = queueItems.length;
  const completed = queueItems.filter(i => i.status === "completed").length;
  const processing = queueItems.filter(i => i.status === "processing").length;
  const failed = queueItems.filter(i => i.status === "error").length;
  const pending = queueItems.filter(i => i.status === "pending").length;

  let overall = 0;
  if (total > 0) {
    const completedSum = completed * 100.0;
    const activeSum = queueItems
      .filter(i => i.status === "processing")
      .reduce((acc, it) => acc + (it.progress_pct || 0), 0);
    overall = Math.min(100, Math.round(((completedSum + activeSum) / total) * 10) / 10);
  }

  const isRunning = batchState.is_running || processing > 0;

  statTotal.innerText = total;
  statCompleted.innerText = completed;
  statPending.innerText = pending;
  statFailed.innerText = failed;

  batchProgressBar.style.width = `${overall}%`;
  batchProgressText.innerText = `${overall}%`;

  // Calculate total size
  const totalMb = queueItems.reduce((acc, it) => acc + (it.size_mb || 0), 0);

  // Dynamic feedback when files are loaded
  if (total > 0) {
    dropzoneTitle.innerText = `📁 ${total} Videos & Images in Batch`;
    dropzoneSubtitle.innerText = `Click "Process All Videos Now" below, or drop more files to append to queue.`;

    if (selectionBanner) {
      selectionBanner.style.display = "flex";
      selBadgeCount.innerText = `${total} Items`;
      selTitleText.innerText = `${total} Items Loaded (${totalMb.toFixed(1)} MB)`;
      selSubtitleText.innerText = `${pending} pending, ${completed} completed, ${failed} failed.`;
    }

    if (pending > 0 && !isRunning) {
      btnStartBatch.classList.add("pulse-glow");
      if (btnStartBanner) btnStartBanner.classList.add("pulse-glow");
    } else {
      btnStartBatch.classList.remove("pulse-glow");
      if (btnStartBanner) btnStartBanner.classList.remove("pulse-glow");
    }
  } else {
    dropzoneTitle.innerText = "Drop Gemini / Veo Videos & Images Here (1 to 100s)";
    dropzoneSubtitle.innerText = "Supports MP4, MOV, WebM, MKV, PNG, JPG, WEBP · Drag & drop files or use the native Windows file picker";
    if (selectionBanner) selectionBanner.style.display = "none";
    btnStartBatch.classList.remove("pulse-glow");
  }

  // Show or hide Retry Failed button
  if (btnRetryFailed) {
    if (failed > 0) {
      btnRetryFailed.style.display = "inline-flex";
      if (failedRetryCount) failedRetryCount.innerText = failed;
    } else {
      btnRetryFailed.style.display = "none";
    }
  }
}

// Periodic synchronization fallback to ensure UI counters never lag or miss state
setInterval(async () => {
  const isRunning = batchState.is_running || queueItems.some(i => i.status === "processing" || i.status === "pending");
  if (isRunning) {
    try {
      const res = await fetch("/api/queue");
      if (res.ok) {
        const data = await res.json();
        batchState = data;
        if (data.items) {
          let changed = false;
          for (const remoteIt of data.items) {
            const localIt = queueItems.find(i => i.id === remoteIt.id);
            if (localIt) {
              if (localIt.status !== remoteIt.status || localIt.progress_pct !== remoteIt.progress_pct) {
                updateCardItem(remoteIt);
                changed = true;
              }
            }
          }
          if (changed) {
            updateBatchStats();
          }
        }
      }
    } catch (e) {
      // Ignore background fetch hiccup
    }
  }
}, 2000);

// Open Windows Explorer
async function openExplorer(path) {
  try {
    await fetch("/api/open-in-explorer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
  } catch (err) {
    console.error(err);
  }
}

// Retry single item
async function retryItem(itemId) {
  try {
    showToast("🔄 Retrying video...", "info");
    await fetch("/api/retry-item", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: itemId }),
    });
  } catch (err) {
    console.error(err);
  }
}

// Remove item
async function removeItem(itemId) {
  try {
    await fetch("/api/remove-item", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: itemId }),
    });
  } catch (err) {
    console.error(err);
  }
}

// Add line to terminal log
function addLog(text) {
  if (!logsTerminal) return;
  const p = document.createElement("p");
  p.innerText = text;
  logsTerminal.appendChild(p);
  logsTerminal.scrollTop = logsTerminal.scrollHeight;
}

function escapePath(path) {
  return (path || "").replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}
