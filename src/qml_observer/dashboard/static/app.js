// QML Observer dashboard frontend.
//
// Polls the app's JSON endpoints (dashboard/app.py) on a fixed interval
// and re-renders. No build step, no framework: this is deliberately a
// small amount of vanilla JS, matching the project's stated "no heavy
// frontend in the MVP" stance (plan.md §16) extended to the dashboard.
(function () {
  "use strict";

  const POLL_MS = 1500;
  const lossCanvas = document.getElementById("loss-chart");
  const gradientCanvas = document.getElementById("gradient-chart");
  const gradientUnavailable = document.getElementById("gradient-unavailable");
  const statusDot = document.getElementById("status-dot");

  async function getJSON(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    return res.json();
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function renderStatus(status) {
    setText("run-id", `run: ${status.run_id || "-"}`);
    statusDot.className = "dot dot-ok";
  }

  function renderLoss(loss) {
    const points = loss.steps.map((s, i) => [s, loss.loss[i]]);
    drawLineChart(lossCanvas, {
      series: [{ label: "loss", points, color: "#4da3ff" }],
    });
  }

  function renderGradient(gradient) {
    if (!gradient.available) {
      gradientUnavailable.classList.remove("hidden");
      gradientCanvas.classList.add("hidden");
      return;
    }
    gradientUnavailable.classList.add("hidden");
    gradientCanvas.classList.remove("hidden");
    const normPoints = gradient.steps.map((s, i) => [s, gradient.norm_l2[i]]);
    drawLineChart(gradientCanvas, {
      series: [{ label: "gradient norm", points: normPoints, color: "#ffb84d" }],
    });
  }

  function renderDiagnosis(diag) {
    const banner = document.getElementById("degraded-banner");
    const issueBadge = document.getElementById("issue");
    const severityBadge = document.getElementById("severity");
    const confFill = document.getElementById("confidence-fill");
    const confValue = document.getElementById("confidence-value");
    const evidenceList = document.getElementById("evidence-list");
    const recsList = document.getElementById("recommendations-list");

    if (!diag || Object.keys(diag).length === 0) {
      banner.classList.add("hidden");
      issueBadge.textContent = "no diagnosis yet";
      issueBadge.className = "issue-badge issue-unknown";
      severityBadge.textContent = "";
      confFill.style.width = "0%";
      confValue.textContent = "n/a";
      evidenceList.innerHTML = '<li class="muted">none</li>';
      recsList.innerHTML = '<li class="muted">none</li>';
      return;
    }

    banner.classList.toggle("hidden", !diag.degraded);
    issueBadge.textContent = String(diag.issue || "unknown").toUpperCase().replace(/_/g, " ");
    issueBadge.className = `issue-badge issue-${diag.issue || "unknown"}`;
    severityBadge.textContent = diag.severity || "";
    severityBadge.className = `severity-badge severity-${diag.severity || ""}`;

    const pct = diag.confidence != null ? Math.round(diag.confidence * 100) : 0;
    confFill.style.width = `${pct}%`;
    confValue.textContent = diag.confidence != null ? `${pct}%` : "n/a";

    fillList(evidenceList, diag.evidence);
    fillList(recsList, diag.recommendations);
  }

  function fillList(el, items) {
    if (!items || items.length === 0) {
      el.innerHTML = '<li class="muted">none</li>';
      return;
    }
    el.innerHTML = "";
    for (const item of items) {
      const li = document.createElement("li");
      li.textContent = item;
      el.appendChild(li);
    }
  }

  function renderCompute(compute) {
    setText("actual-steps", compute.actual_steps.toLocaleString());
    setText("planned-steps", compute.planned_steps != null ? compute.planned_steps.toLocaleString() : "-");
    setText(
      "mean-wall-time",
      compute.mean_wall_time_per_step != null ? compute.mean_wall_time_per_step.toFixed(4) : "-"
    );
    setText("compute-saved", compute.formatted);
  }

  function renderHistory(history) {
    const unconfigured = document.getElementById("history-unconfigured");
    const empty = document.getElementById("history-empty");
    const table = document.getElementById("history-table");
    const tbody = document.getElementById("history-tbody");

    if (!history.directory) {
      unconfigured.classList.remove("hidden");
      empty.classList.add("hidden");
      table.classList.add("hidden");
      return;
    }
    unconfigured.classList.add("hidden");

    if (!history.entries || history.entries.length === 0) {
      empty.classList.remove("hidden");
      table.classList.add("hidden");
      return;
    }
    empty.classList.add("hidden");
    table.classList.remove("hidden");

    tbody.innerHTML = "";
    for (const entry of history.entries) {
      const tr = document.createElement("tr");
      const confidence = entry.confidence != null ? `${Math.round(entry.confidence * 100)}%` : "-";
      const cells = [
        entry.run_id || "-",
        entry.framework || "-",
        entry.steps,
        entry.issue || "-",
        confidence,
        entry.formatted_compute_saved,
        entry.modified_at ? entry.modified_at.replace("T", " ").slice(0, 19) : "-",
      ];
      for (const value of cells) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  async function refresh() {
    try {
      const [status, loss, gradient, diagnosis, compute, history] = await Promise.all([
        getJSON("/api/status"),
        getJSON("/api/loss"),
        getJSON("/api/gradient"),
        getJSON("/api/diagnosis"),
        getJSON("/api/compute"),
        getJSON("/api/history"),
      ]);
      renderStatus(status);
      renderLoss(loss);
      renderGradient(gradient);
      renderDiagnosis(diagnosis);
      renderCompute(compute);
      renderHistory(history);
    } catch (err) {
      statusDot.className = "dot dot-error";
      console.error("dashboard refresh failed:", err);
    }
  }

  refresh();
  setInterval(refresh, POLL_MS);
})();
