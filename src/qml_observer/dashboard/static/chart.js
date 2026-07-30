// Minimal, dependency-free canvas line-chart drawer for the QML Observer
// dashboard. Deliberately not Chart.js/D3/etc: the dashboard should render
// with zero internet access (matching the project's "nothing leaves your
// machine by default" posture), and the charting need here -- one or more
// line series against a shared x-axis -- doesn't need a general-purpose
// charting library.
//
// Usage:
//   drawLineChart(canvas, {
//     series: [{ label: "loss", points: [[step, value], ...], color: "#..." }],
//     yLabel: "loss",
//   });
(function (global) {
  "use strict";

  function niceRange(min, max) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    if (min === max) {
      const pad = Math.abs(min) * 0.1 || 1;
      return [min - pad, max + pad];
    }
    const pad = (max - min) * 0.08;
    return [min - pad, max + pad];
  }

  function drawLineChart(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const series = (opts.series || []).filter((s) => s.points && s.points.length > 0);
    const marginLeft = 56;
    const marginRight = 12;
    const marginTop = 12;
    const marginBottom = 28;
    const plotW = w - marginLeft - marginRight;
    const plotH = h - marginTop - marginBottom;

    ctx.fillStyle = "#0f1420";
    ctx.fillRect(0, 0, w, h);

    if (series.length === 0) {
      ctx.fillStyle = "#5b6478";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText("no data yet", marginLeft, h / 2);
      return;
    }

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const s of series) {
      for (const [x, y] of s.points) {
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y !== null && y !== undefined && isFinite(y)) {
          if (y < yMin) yMin = y;
          if (y > yMax) yMax = y;
        }
      }
    }
    if (xMin === xMax) xMax = xMin + 1;
    const [yLo, yHi] = niceRange(yMin, yMax);

    const px = (x) => marginLeft + ((x - xMin) / (xMax - xMin)) * plotW;
    const py = (y) => marginTop + plotH - ((y - yLo) / (yHi - yLo)) * plotH;

    // gridlines + y labels
    ctx.strokeStyle = "#1e2436";
    ctx.fillStyle = "#5b6478";
    ctx.font = "11px system-ui, sans-serif";
    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
      const y = yLo + ((yHi - yLo) * i) / ySteps;
      const yy = py(y);
      ctx.beginPath();
      ctx.moveTo(marginLeft, yy);
      ctx.lineTo(w - marginRight, yy);
      ctx.stroke();
      ctx.fillText(formatNumber(y), 4, yy + 4);
    }

    // x labels (first/last step)
    ctx.fillText(String(Math.round(xMin)), marginLeft, h - 8);
    const lastLabel = String(Math.round(xMax));
    ctx.fillText(lastLabel, w - marginRight - ctx.measureText(lastLabel).width, h - 8);

    for (const s of series) {
      ctx.strokeStyle = s.color || "#4da3ff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (const [x, y] of s.points) {
        if (y === null || y === undefined || !isFinite(y)) {
          started = false;
          continue;
        }
        const xx = px(x);
        const yy = py(y);
        if (!started) {
          ctx.moveTo(xx, yy);
          started = true;
        } else {
          ctx.lineTo(xx, yy);
        }
      }
      ctx.stroke();
    }
  }

  function formatNumber(n) {
    if (Math.abs(n) >= 1000 || (Math.abs(n) < 0.001 && n !== 0)) {
      return n.toExponential(1);
    }
    return n.toFixed(3);
  }

  global.drawLineChart = drawLineChart;
})(window);
