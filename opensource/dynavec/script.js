/* ================================================
   dynavec landing page — script.js
   1. Animated dot-node network background
   2. Benchmark charts (Chart.js)
   3. Cost calculator
   ================================================ */

/* ---- 1. Background node network canvas ---- */
(function () {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const NODES = 70;
  const MAX_DIST = 160;
  const NODE_RADIUS = 2.2;
  const SPEED = 0.35;
  const CORAL = "#e05a3a";
  const DOT_COLOR = "rgba(180,180,200,0.55)";
  const LINE_COLOR = "rgba(180,180,200,";

  let W, H, nodes, dpr;

  function resize() {
    dpr = window.devicePixelRatio || 1;
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function makeNode() {
    const angle = Math.random() * Math.PI * 2;
    const speed = SPEED * (0.4 + Math.random() * 0.6);
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      accent: Math.random() < 0.08,
    };
  }

  function init() {
    resize();
    nodes = Array.from({ length: NODES }, makeNode);
  }

  function tick() {
    ctx.clearRect(0, 0, W, H);

    nodes.forEach((n) => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });

    // draw edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DIST) {
          const alpha = (1 - dist / MAX_DIST) * 0.45;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = LINE_COLOR + alpha + ")";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    // draw nodes
    nodes.forEach((n) => {
      ctx.beginPath();
      ctx.arc(
        n.x,
        n.y,
        n.accent ? NODE_RADIUS * 1.6 : NODE_RADIUS,
        0,
        Math.PI * 2
      );
      ctx.fillStyle = n.accent ? CORAL : DOT_COLOR;
      ctx.fill();
    });

    requestAnimationFrame(tick);
  }

  window.addEventListener("resize", () => {
    resize(); // re-applies dpr scale transform
    nodes.forEach((n) => {
      n.x = Math.min(n.x, W);
      n.y = Math.min(n.y, H);
    });
  });

  init();
  tick();
})();

/* ---- 2. Benchmark charts ---- */
window.addEventListener("DOMContentLoaded", function () {
  if (typeof Chart === "undefined") return;

  const CORAL = "#e05a3a";
  const GRAY1 = "#9ca3af";
  const GRAY2 = "#c4c9d4";
  const GRAY3 = "#d1d5db";
  const GRAY4 = "#e5e7eb";

  const font = { family: "'Inter', sans-serif", size: 12 };
  Chart.defaults.font = font;
  Chart.defaults.color = "#6a6a6a";

  // --- bench-matrix: grouped bar — cost by scale at 1536-dim ---
  const matrixCtx = document.getElementById("bench-matrix");
  if (matrixCtx) {
    const scales = ["100K", "1M", "10M", "100M", "1B"];
    new Chart(matrixCtx, {
      type: "bar",
      data: {
        labels: scales,
        datasets: [
          {
            label: "dynavec",
            data: [3, 3, 8, 50, 469],
            backgroundColor: CORAL,
          },
          {
            label: "Pinecone",
            data: [9, 10, 27, 197, 1897],
            backgroundColor: GRAY1,
          },
          {
            label: "Qdrant",
            data: [160, 160, 960, 8640, 85920],
            backgroundColor: GRAY2,
          },
          {
            label: "Weaviate",
            data: [175, 175, 1050, 9450, 93975],
            backgroundColor: GRAY3,
          },
          {
            label: "OpenSearch",
            data: [701, 701, 877, 8423, 83708],
            backgroundColor: GRAY4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          title: {
            display: true,
            text: "Monthly cost ($/mo) — 1536-dim, 1M queries/mo",
            font: { size: 13 },
          },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ` ${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString()}`,
            },
          },
        },
        scales: {
          y: {
            type: "logarithmic",
            title: { display: true, text: "$/month (log scale)" },
            ticks: { callback: (v) => "$" + v.toLocaleString() },
          },
          x: { title: { display: true, text: "Vector count" } },
        },
      },
    });
  }

  // --- bench-scale: line — cost by scale at 768-dim ---
  const scaleCtx = document.getElementById("bench-scale");
  if (scaleCtx) {
    const pts = ["100K", "1M", "10M", "100M", "1B"];
    new Chart(scaleCtx, {
      type: "line",
      data: {
        labels: pts,
        datasets: [
          {
            label: "dynavec",
            data: [2, 2, 5, 32, 295],
            borderColor: CORAL,
            backgroundColor: CORAL + "22",
            tension: 0.35,
            fill: true,
          },
          {
            label: "Pinecone",
            data: [9, 10, 27, 197, 1897],
            borderColor: GRAY1,
            backgroundColor: "transparent",
            tension: 0.35,
          },
          {
            label: "Qdrant",
            data: [80, 80, 480, 4320, 42960],
            borderColor: GRAY2,
            backgroundColor: "transparent",
            tension: 0.35,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          title: {
            display: true,
            text: "Cost by scale (768-d)",
            font: { size: 12 },
          },
        },
        scales: {
          y: {
            type: "logarithmic",
            ticks: { callback: (v) => "$" + v.toLocaleString() },
          },
        },
      },
    });
  }

  // --- bench-storage: bar — raw storage footprint ---
  const storageCtx = document.getElementById("bench-storage");
  if (storageCtx) {
    new Chart(storageCtx, {
      type: "bar",
      data: {
        labels: ["100K", "1M", "10M", "100M", "1B"],
        datasets: [
          {
            label: "Storage (GiB)",
            data: [0.57, 5.7, 57, 573, 5730],
            backgroundColor: CORAL + "cc",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: "Float32 storage footprint (1536-d)",
            font: { size: 12 },
          },
        },
        scales: {
          y: {
            type: "logarithmic",
            title: { display: true, text: "GiB (log)" },
            ticks: { callback: (v) => v + " GiB" },
          },
        },
      },
    });
  }

  // --- quality-chart: doughnut — recall/latency score ---
  const qualityCtx = document.getElementById("quality-chart");
  if (qualityCtx) {
    new Chart(qualityCtx, {
      type: "doughnut",
      data: {
        labels: ["dynavec", "Pinecone", "Qdrant", "Weaviate", "OpenSearch"],
        datasets: [
          {
            data: [92, 88, 85, 82, 74],
            backgroundColor: [CORAL, GRAY1, GRAY2, GRAY3, GRAY4],
            borderWidth: 2,
            borderColor: "#fff",
          },
        ],
      },
      options: {
        responsive: true,
        cutout: "62%",
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 12 } },
          title: {
            display: true,
            text: "Recall ÷ Latency score",
            font: { size: 12 },
          },
          tooltip: {
            callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.parsed}` },
          },
        },
      },
    });
  }

  /* ---- 3. Cost calculator ---- */
  const calcVectors = document.getElementById("calc-vectors");
  const calcDim = document.getElementById("calc-dim");
  const calcQpm = document.getElementById("calc-qpm");
  const calcWpm = document.getElementById("calc-wpm");
  const calcTotal = document.getElementById("calc-total");
  const calcBreakdown = document.getElementById("calc-breakdown");

  function computeCost() {
    if (!calcVectors) return;
    const vectors = parseFloat(calcVectors.value) || 0;
    const dim = parseFloat(calcDim.value) || 768;
    const qpm = parseFloat(calcQpm.value) || 0;
    const wpm = parseFloat(calcWpm.value) || 0;

    // S3 Vectors: $0.04 / GB-month stored + $0.04 / 1M reads
    const bytesPerVector = dim * 4; // float32
    const storedGB = (vectors * bytesPerVector) / 1e9;
    const storageCost = storedGB * 0.04;
    const readCost = (qpm / 1e6) * 0.04;

    // DynamoDB: $0.00013 / read RCU + $0.00065 / write WCU (on-demand, us-east-1)
    const dynRead = qpm * 0.00000013;
    const dynWrite = wpm * 0.00000065;

    const total = storageCost + readCost + dynRead + dynWrite;

    if (calcTotal) calcTotal.textContent = `$${total.toFixed(2)} / mo`;
    if (calcBreakdown) {
      calcBreakdown.innerHTML =
        `<span>S3 storage ${storedGB.toFixed(2)} GB: $${storageCost.toFixed(
          2
        )}</span>` +
        `<span>S3 reads: $${readCost.toFixed(2)}</span>` +
        `<span>DynamoDB reads: $${dynRead.toFixed(2)}</span>` +
        `<span>DynamoDB writes: $${dynWrite.toFixed(2)}</span>`;
    }
  }

  [calcVectors, calcDim, calcQpm, calcWpm].forEach((el) => {
    if (el) el.addEventListener("input", computeCost);
  });
  computeCost();
});

/* ---- 4. Sticky nav ---- */
(function () {
  const nav = document.getElementById("nav");
  if (!nav) return;
  const onScroll = () => nav.classList.toggle("is-stuck", window.scrollY > 10);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();

/* ---- 5. Scroll reveal (IntersectionObserver) ---- */
(function () {
  const targets = document.querySelectorAll(
    "[data-reveal], .section__title, .hero__stats"
  );
  if (!targets.length || !("IntersectionObserver" in window)) {
    targets.forEach((el) => el.classList.add("is-visible"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("is-visible");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  targets.forEach((el) => io.observe(el));
})();

/* ---- 6. Tab switching ---- */
(function () {
  const tabsEl = document.querySelector("[data-tabs]");
  if (!tabsEl) return;
  const tabs = tabsEl.querySelectorAll(".tab");
  const panels = tabsEl.querySelectorAll('[role="tabpanel"]');

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.classList.remove("is-active");
        t.setAttribute("aria-selected", "false");
      });
      panels.forEach((p) => p.classList.add("is-hidden"));
      tab.classList.add("is-active");
      tab.setAttribute("aria-selected", "true");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.remove("is-hidden");
    });
  });
})();

/* ---- 7. Copy buttons ---- */
(function () {
  document.querySelectorAll(".copy[data-copy]").forEach((btn) => {
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(btn.dataset.copy).then(() => {
        btn.textContent = "Copied!";
        btn.classList.add("is-done");
        setTimeout(() => {
          btn.textContent = "copy";
          btn.classList.remove("is-done");
        }, 1800);
      });
    });
  });
})();

/* ---- 8. YouTube facade ---- */
(function () {
  const facade = document.getElementById("yt-facade");
  if (!facade) return;
  facade.addEventListener("click", () => {
    const iframe = document.createElement("iframe");
    iframe.src = "https://www.youtube.com/embed/UJ9MBALD380?autoplay=1&rel=0";
    iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
    iframe.setAttribute("allowfullscreen", "");
    iframe.style.cssText = "position:absolute;inset:0;width:100%;height:100%;border:0;";
    // Clear thumbnail/play button and drop in the iframe
    facade.innerHTML = "";
    facade.style.cursor = "default";
    facade.appendChild(iframe);
  });
  facade.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") facade.click();
  });
})();

/* ---- 9. Contributors carousel ---- */
(function () {
  const section = document.querySelector(".contributors-section");
  const wrap = section && section.querySelector(".contrib-track-wrap");
  const track = document.getElementById("contrib-track");
  if (!section || !wrap || !track) return;

  // All photos live in images/contributors/<filename>
  // Add a file here and reference it below — no GitHub CDN used.
  const allData = [
    {
      name: "Abhishek Gupta",
      photo: "images/contributors/abhishek-gupta.png",
      url: "https://github.com/shivamm-gupta",
    },
    {
      name: "Sanket Tikhande",
      photo: "images/contributors/sanket-tikhande.jpg",
      url: "https://github.com/tikhandesanket",
    },
    { name: "Kaap10", photo: null, url: "https://github.com/Kaap10" },
    {
      name: "shivamm-gupta",
      photo: null,
      url: "https://github.com/shivamm-gupta",
    },
    { name: "Ashishds", photo: null, url: "https://github.com/Ashishds" },
    { name: "Isha-Zaka", photo: null, url: "https://github.com/Isha-Zaka" },
    { name: "vaishnavk09", photo: null, url: "https://github.com/vaishnavk09" },
    { name: "be-student", photo: null, url: "https://github.com/be-student" },
    { name: "redcode333", photo: null, url: "https://github.com/redcode333" },
    { name: "R3108", photo: null, url: "https://github.com/R3108" },
    {
      name: "GenZ-CODER-X",
      photo: null,
      url: "https://github.com/GenZ-CODER-X",
    },
    {
      name: "AyushhVatsal",
      photo: null,
      url: "https://github.com/AyushhVatsal",
    },
    { name: "arshsk16", photo: null, url: "https://github.com/arshsk16" },
    { name: "Uzmaa7", photo: null, url: "https://github.com/Uzmaa7" },
    {
      name: "ramashishmaurya",
      photo: null,
      url: "https://github.com/ramashishmaurya",
    },
    { name: "Henilll", photo: null, url: "https://github.com/Henilll" },
    { name: "Lawliet2004", photo: null, url: "https://github.com/Lawliet2004" },
    { name: "wang1408", photo: null, url: "https://github.com/wang1408" },
    { name: "theman6660", photo: null, url: "https://github.com/theman6660" },
    {
      name: "osamashabih6960",
      photo: null,
      url: "https://github.com/osamashabih6960",
    },
  ];

  // Palette for initials avatars (cycles through)
  const PALETTE = [
    "#e8623b",
    "#4f8ef7",
    "#2db87c",
    "#9b67e0",
    "#e8a43b",
    "#e84f7a",
    "#3bbde8",
  ];

  function initialsAvatar(name, idx) {
    const words = name.trim().split(/[\s_\-]+/);
    const initials =
      words.length >= 2
        ? (words[0][0] + words[1][0]).toUpperCase()
        : name.slice(0, 2).toUpperCase();
    const color = PALETTE[idx % PALETTE.length];
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <circle cx="40" cy="40" r="40" fill="${color}22"/>
      <circle cx="40" cy="40" r="39" fill="none" stroke="${color}" stroke-width="2"/>
      <text x="40" y="46" font-family="Inter,sans-serif" font-size="26" font-weight="700"
            fill="${color}" text-anchor="middle" dominant-baseline="middle">${initials}</text>
    </svg>`;
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }

  function makeCard(c, idx) {
    const hasPhoto = !!c.photo;
    const card = document.createElement("a");
    card.href = c.url;
    card.target = "_blank";
    card.rel = "noopener noreferrer";
    card.className = "contrib-card" + (hasPhoto ? " contrib-card--team" : "");

    const img = document.createElement("img");
    img.src = hasPhoto ? c.photo : initialsAvatar(c.name, idx);
    img.alt = c.name;
    img.className = "contrib-card__avatar";
    img.loading = "lazy";
    img.width = 80;
    img.height = 80;

    const name = document.createElement("div");
    name.className = "contrib-card__name";
    name.textContent = c.name;

    card.appendChild(img);
    card.appendChild(name);
    return card;
  }

  allData.forEach((c, i) => track.appendChild(makeCard(c, i)));

  const cards = Array.from(track.querySelectorAll(".contrib-card"));
  const total = cards.length;
  const btnPrev = document.getElementById("contrib-prev");
  const btnNext = document.getElementById("contrib-next");

  // Only enable interactive scroll if more than 5 contributors
  if (total <= 5) {
    if (btnPrev) btnPrev.style.display = "none";
    if (btnNext) btnNext.style.display = "none";
    return;
  }

  /* ---- Autoplay ---- */
  const DWELL_TOP5 = 3000; // ms — first 5 stay longer
  const DWELL_REST = 1500;
  const MAX_SPEED = 6; // px per frame during hover
  const DEAD_ZONE = 0.25; // centre fraction with no scroll

  let currentIdx = 0;
  let autoTimer = null;
  let rafId = null;
  let isHovering = false;
  let velocity = 0; // px/frame, set by pointer position

  const CARD_STEP = 130 + 32; // card-w + card-gap

  function dwellFor(idx) {
    return idx < 5 ? DWELL_TOP5 : DWELL_REST;
  }

  function snapToCard(idx, smooth) {
    const card = cards[idx];
    if (!card) return;
    const target = card.offsetLeft - 40; // account for track padding
    if (smooth) {
      smoothScrollTo(target, 500);
    } else {
      wrap.scrollLeft = target;
    }
  }

  function smoothScrollTo(targetLeft, duration) {
    const start = wrap.scrollLeft;
    const distance = targetLeft - start;
    const startTime = performance.now();
    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      wrap.scrollLeft = start + distance * ease;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function updateButtons() {
    if (btnPrev) btnPrev.disabled = wrap.scrollLeft <= 0;
    if (btnNext) btnNext.disabled = wrap.scrollLeft >= wrap.scrollWidth - wrap.clientWidth - 1;
  }

  function scheduleNext() {
    clearTimeout(autoTimer);
    if (isHovering) return;
    // Stop autoplay at the last card — no wrap-around
    if (currentIdx >= total - 1) return;
    autoTimer = setTimeout(() => {
      if (isHovering) return;
      currentIdx = currentIdx + 1;
      snapToCard(currentIdx, true);
      scheduleNext();
    }, dwellFor(currentIdx));
  }

  /* ---- Pointer-based hover scroll (rAF loop) ---- */
  function pointerLoop() {
    if (!isHovering) return;
    if (velocity !== 0) {
      const next = wrap.scrollLeft + velocity;
      // Clamp at edges — no wrap-around
      wrap.scrollLeft = Math.max(0, Math.min(next, wrap.scrollWidth - wrap.clientWidth));
    }
    rafId = requestAnimationFrame(pointerLoop);
  }

  section.addEventListener("mousemove", (e) => {
    const rect = section.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const w = rect.width;
    const norm = (x / w) * 2 - 1;
    if (Math.abs(norm) < DEAD_ZONE) {
      velocity = 0;
    } else {
      const sign = norm > 0 ? 1 : -1;
      const ratio = (Math.abs(norm) - DEAD_ZONE) / (1 - DEAD_ZONE);
      velocity = sign * ratio * MAX_SPEED;
    }
  });

  section.addEventListener("mouseenter", () => {
    isHovering = true;
    clearTimeout(autoTimer);
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(pointerLoop);
  });

  section.addEventListener("mouseleave", () => {
    isHovering = false;
    velocity = 0;
    cancelAnimationFrame(rafId);
    scheduleNext();
  });

  /* ---- Button click scroll ---- */
  function smoothScrollBy(delta) {
    const start    = wrap.scrollLeft;
    const target   = Math.max(0, Math.min(start + delta, wrap.scrollWidth - wrap.clientWidth));
    const duration = 120;
    const t0       = performance.now();
    function step(now) {
      const t    = Math.min((now - t0) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 2);
      wrap.scrollLeft = start + (target - start) * ease;
      if (t < 1) requestAnimationFrame(step);
      else updateButtons();
    }
    requestAnimationFrame(step);
  }

  if (btnPrev) btnPrev.addEventListener("click", () => smoothScrollBy(-CARD_STEP * 3));
  if (btnNext) btnNext.addEventListener("click", () => smoothScrollBy(CARD_STEP * 3));
  wrap.addEventListener("scroll", updateButtons, { passive: true });
  updateButtons();

  // Kick off autoplay
  scheduleNext();
})();
