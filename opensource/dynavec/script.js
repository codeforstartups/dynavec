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
});

/* ---- 3. Cost calculator v2 ---- */
(function () {
  function init() {
    const slVectors = document.getElementById("sl-vectors");
    const slDim = document.getElementById("sl-dim");
    const slQpm = document.getElementById("sl-qpm");
    const slWpm = document.getElementById("sl-wpm");
    if (!slVectors) return;

    const DIM_OPTS = [384, 768, 1536, 3072];

    function getVectors() {
      return Math.round(Math.pow(10, parseFloat(slVectors.value)));
    }
    function getDim() {
      return DIM_OPTS[parseInt(slDim.value)];
    }
    function getQpm() {
      return Math.round(Math.pow(10, parseFloat(slQpm.value)));
    }
    function getWpm() {
      return Math.round(Math.pow(10, parseFloat(slWpm.value)));
    }

    function fmtN(n) {
      if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, "") + " B";
      if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + " M";
      if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + " K";
      return String(n);
    }
    function fmtUSD(n) {
      if (n >= 1000)
        return "$" + n.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
      return "$" + n.toFixed(2);
    }

    const PRESETS = {
      starter: { vectors: 5, dim: 1, qpm: 5, wpm: 4 },
      growth: { vectors: 6, dim: 2, qpm: 6.7, wpm: 5.7 },
      scale: { vectors: 7, dim: 2, qpm: 7.7, wpm: 6.7 },
    };

    const presetBtns = document.querySelectorAll(".calc2__preset");

    function updateActivePreset() {
      const v = parseFloat(slVectors.value);
      const d = parseInt(slDim.value);
      const q = parseFloat(slQpm.value);
      const w = parseFloat(slWpm.value);

      let matchedPreset = null;
      for (const [key, p] of Object.entries(PRESETS)) {
        if (
          Math.abs(v - p.vectors) < 0.05 &&
          d === p.dim &&
          Math.abs(q - p.qpm) < 0.05 &&
          Math.abs(w - p.wpm) < 0.05
        ) {
          matchedPreset = key;
          break;
        }
      }

      presetBtns.forEach((btn) => {
        btn.classList.toggle("is-active", btn.dataset.preset === matchedPreset);
      });
    }

    presetBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = PRESETS[btn.dataset.preset];
        if (!p) return;
        slVectors.value = p.vectors;
        slDim.value = p.dim;
        slQpm.value = p.qpm;
        slWpm.value = p.wpm;
        presetBtns.forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        compute();
      });
    });

    function setBar(id, amtId, cost, max) {
      const fill = document.getElementById(id);
      const amt = document.getElementById(amtId);
      if (fill)
        fill.style.width = (max > 0 ? Math.min(cost / max, 1) * 100 : 0) + "%";
      if (amt) amt.textContent = fmtUSD(cost);
    }

    function compute() {
      const vectors = getVectors();
      const dim = getDim();
      const qpm = getQpm();
      const wpm = getWpm();

      const lblV = document.getElementById("lbl-vectors");
      const lblD = document.getElementById("lbl-dim");
      const lblQ = document.getElementById("lbl-qpm");
      const lblW = document.getElementById("lbl-wpm");
      if (lblV) lblV.textContent = fmtN(vectors);
      if (lblD) lblD.textContent = dim;
      if (lblQ) lblQ.textContent = fmtN(qpm);
      if (lblW) lblW.textContent = fmtN(wpm);

      const storedGB = (vectors * dim * 4) / 1e9;
      const storageCost = storedGB * 0.04;
      const readCost = (qpm / 1e6) * 0.04;
      const dynRead = qpm * 0.00000013;
      const dynWrite = wpm * 0.00000065;
      const dvTotal = storageCost + readCost + dynRead + dynWrite;

      const pnTotal =
        storedGB * 0.096 + (qpm / 1e6) * 0.1 + (wpm / 1e6) * 0.5 + 20;
      const saving =
        pnTotal > 0 ? Math.round((1 - dvTotal / pnTotal) * 100) : 0;

      const dvEl = document.getElementById("calc-dv-price");
      const pnEl = document.getElementById("calc-pn-price");
      const svEl = document.getElementById("calc-savings");
      if (dvEl) dvEl.textContent = fmtUSD(dvTotal) + " /mo";
      if (pnEl) pnEl.textContent = "~" + fmtUSD(pnTotal) + " /mo";
      if (svEl)
        svEl.textContent =
          saving > 0
            ? `You save ~${saving}% vs Pinecone`
            : "Similar cost to Pinecone";

      const maxCost = Math.max(storageCost, readCost, dynRead, dynWrite, 0.01);
      setBar("bar-storage", "amt-storage", storageCost, maxCost);
      setBar("bar-reads", "amt-reads", readCost, maxCost);
      setBar("bar-ddb-r", "amt-ddb-r", dynRead, maxCost);
      setBar("bar-ddb-w", "amt-ddb-w", dynWrite, maxCost);
      updateActivePreset();
    }

    [slVectors, slDim, slQpm, slWpm].forEach((el) =>
      el.addEventListener("input", compute)
    );
    compute();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

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

  function activateVideo(e) {
    if (e && (e.ctrlKey || e.metaKey || e.shiftKey)) return;
    if (e) e.preventDefault();

    if (facade.querySelector("iframe")) return;

    const iframe = document.createElement("iframe");
    iframe.src = "https://www.youtube.com/embed/UJ9MBALD380?autoplay=1&rel=0";
    iframe.title = "dynavec demo video";
    iframe.allow =
      "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
    iframe.setAttribute("allowfullscreen", "");
    iframe.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;border:0;border-radius:inherit;";

    facade.removeAttribute("href");
    facade.removeAttribute("target");
    facade.removeAttribute("rel");
    facade.style.cursor = "default";
    facade.innerHTML = "";
    facade.appendChild(iframe);
  }

  facade.addEventListener("click", activateVideo);
  facade.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      activateVideo();
    }
  });
})();

/* ---- 9. Contributors grid ---- */
(function () {
  const coreEl = document.getElementById("contrib-core");
  const allEl = document.getElementById("contrib-all");
  if (!coreEl || !allEl) return;

  const CORE_NAMES = [
    "Abhishek Gupta",
    "Sanket Tikhande",
    "Vardhman Gupta",
    "Shivam Gupta",
    "Isha Zaka",
  ];

  // To add a contributor: append their name here.
  // Drop a photo as images/contributors/<name-lowercase-hyphenated>.png — auto-detected.
  const ALL_NAMES = [
    "Abhishek Gupta",
    "Sanket Tikhande",
    "Vardhman Gupta",
    "Shivam Gupta",
    "Isha Zaka",
    "Uzma Khan",
    "Ashish Kumar",
    "Madhav Sharma",
    "Tanmay Kumar",
    "Henil Bhavsar",
    "Mohammad Arshad Ali",
    "vaishnavk09",
    "be-student",
    "redcode333",
    "R3108",
    "AyushhVatsal",
    "ramashishmaurya",
    "Lawliet2004",
    "wang1408",
    "theman6660",
    "osamashabih6960",
  ];

  function nameToPhotoPath(name) {
    return (
      "images/contributors/" +
      name.trim().toLowerCase().replace(/\s+/g, "-") +
      ".png"
    );
  }

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
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
      <rect width="200" height="200" fill="${color}18"/>
      <text x="100" y="115" font-family="Inter,sans-serif" font-size="72" font-weight="700"
            fill="${color}" text-anchor="middle">${initials}</text>
    </svg>`;
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  }

  function makeCard(name, idx, isCore) {
    const photo = nameToPhotoPath(name);

    const card = document.createElement("div");
    card.className = "cg-card" + (isCore ? " cg-card--core" : "");

    const photoWrap = document.createElement("div");
    photoWrap.className = "cg-card__photo";

    const img = document.createElement("img");
    img.src = photo;
    img.alt = name;
    img.loading = "lazy";
    img.onerror = () => {
      img.onerror = null;
      img.src = initialsAvatar(name, idx);
    };

    photoWrap.appendChild(img);

    const info = document.createElement("div");
    info.className = "cg-card__info";

    const nameEl = document.createElement("div");
    nameEl.className = "cg-card__name";
    nameEl.textContent = name;

    info.appendChild(nameEl);

    card.appendChild(photoWrap);
    card.appendChild(info);
    return card;
  }

  const coreSet = new Set(CORE_NAMES);
  CORE_NAMES.forEach((name, i) => coreEl.appendChild(makeCard(name, i, true)));

  const restNames = ALL_NAMES.filter((n) => !coreSet.has(n));
  const INITIAL_SHOW = 5;

  // Render first 5 immediately
  restNames
    .slice(0, INITIAL_SHOW)
    .forEach((name, i) => allEl.appendChild(makeCard(name, i, false)));

  // If there are more, add a "Show more" button
  if (restNames.length > INITIAL_SHOW) {
    const showMoreBtn = document.createElement("button");
    showMoreBtn.className = "contrib-show-more";
    showMoreBtn.textContent = `Show ${
      restNames.length - INITIAL_SHOW
    } more contributors`;
    allEl.after(showMoreBtn);

    showMoreBtn.addEventListener("click", () => {
      restNames
        .slice(INITIAL_SHOW)
        .forEach((name, i) =>
          allEl.appendChild(makeCard(name, INITIAL_SHOW + i, false))
        );
      showMoreBtn.remove();
    });
  }
})();

/* ---- 10. GitHub star count ---- */
(function () {
  fetch("https://api.github.com/repos/codeforstartups/dynavec")
    .then((r) => r.json())
    .then((data) => {
      const count = data.stargazers_count;
      if (!count) return;
      const fmt =
        count >= 1000
          ? (count / 1000).toFixed(1).replace(/\.0$/, "") + "k"
          : String(count);
      document.querySelectorAll("[data-stars]").forEach((el) => {
        el.textContent = fmt;
      });
    })
    .catch(() => {});
})();
