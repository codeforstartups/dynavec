/* dynavec landing — no external dependencies, no tracking. */
(function () {
  "use strict";

  var REPO = "codeforstartups/dynavec";

  /* ================================================================
     STICKY NAV
     ================================================================ */
  var nav = document.getElementById("nav");
  function onScroll() { if (nav) nav.classList.toggle("is-stuck", window.scrollY > 8); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ================================================================
     COPY BUTTONS
     ================================================================ */
  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy") || "";
      navigator.clipboard.writeText(text).then(function () {
        var old = btn.textContent;
        btn.textContent = "copied";
        btn.classList.add("is-done");
        setTimeout(function () { btn.textContent = old; btn.classList.remove("is-done"); }, 1400);
      });
    });
  });

  /* ================================================================
     TABS
     ================================================================ */
  var tabsRoot = document.querySelector("[data-tabs]");
  if (tabsRoot) {
    var tabs = tabsRoot.querySelectorAll(".tab");
    var panels = tabsRoot.querySelectorAll(".tabs__panels > .code");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (t) { t.classList.remove("is-active"); t.setAttribute("aria-selected", "false"); });
        panels.forEach(function (p) { p.classList.add("is-hidden"); });
        tab.classList.add("is-active");
        tab.setAttribute("aria-selected", "true");
        var target = document.getElementById(tab.getAttribute("data-tab"));
        if (target) target.classList.remove("is-hidden");
      });
    });
  }

  /* ================================================================
     GITHUB STAR COUNT
     ================================================================ */
  function formatStars(n) {
    if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k";
    return String(n);
  }
  fetch("https://api.github.com/repos/" + REPO)
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (data) {
      if (!data || data.stargazers_count == null) return;
      var label = formatStars(data.stargazers_count);
      document.querySelectorAll("[data-stars]").forEach(function (el) { el.textContent = label; });
    })
    .catch(function () {});

  /* ================================================================
     PYTHON SYNTAX HIGHLIGHTER
     ================================================================ */
  var KW = /\b(from|import|for|in|as|def|return|if|else|elif|not|and|or|None|True|False|with|class|lambda|print|is)\b/g;
  function highlight(text) {
    return text.split("\n").map(function (line) {
      var inStr = null, ci = -1;
      for (var i = 0; i < line.length; i++) {
        var ch = line[i];
        if (inStr) { if (ch === inStr) inStr = null; }
        else if (ch === "'" || ch === '"') inStr = ch;
        else if (ch === "#") { ci = i; break; }
      }
      var code = line, comment = "";
      if (ci >= 0) { code = line.slice(0, ci); comment = line.slice(ci); }
      var strs = [];
      code = code.replace(/(['"])(?:\\.|(?!\1).)*\1/g, function (m) { strs.push(m); return " " + (strs.length - 1) + " "; });
      code = code.replace(KW, '<span class="c-kw">$1</span>');
      code = code.replace(/ (\d+) /g, function (m, idx) { return '<span class="c-str">' + strs[idx] + "</span>"; });
      if (comment) comment = '<span class="c-comment">' + comment + "</span>";
      return code + comment;
    }).join("\n");
  }
  document.querySelectorAll(".tabs__panels .code code, .code--sm code").forEach(function (el) {
    if (el.textContent.indexOf("<") === -1) el.innerHTML = highlight(el.textContent);
  });

  /* ================================================================
     FULL-PAGE BACKGROUND NODE NETWORK
     ================================================================ */
  (function () {
    var canvas = document.getElementById("bg-canvas");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var W, H, nodes;
    var NODE_COUNT = 55;
    var MAX_DIST = 180;

    function resize() {
      var dpr = window.devicePixelRatio || 1;
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = W + "px";
      canvas.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function makeNodes() {
      nodes = [];
      for (var i = 0; i < NODE_COUNT; i++) {
        nodes.push({
          x: Math.random() * W,
          y: Math.random() * H,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          r: 2 + Math.random() * 2.5,
        });
      }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      for (var i = 0; i < nodes.length; i++) {
        var a = nodes[i];
        a.x += a.vx; a.y += a.vy;
        if (a.x < 0 || a.x > W) a.vx *= -1;
        if (a.y < 0 || a.y > H) a.vy *= -1;
        for (var j = i + 1; j < nodes.length; j++) {
          var b = nodes[j];
          var dx = a.x - b.x, dy = a.y - b.y;
          var dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MAX_DIST) {
            ctx.strokeStyle = "rgba(180,60,20," + (0.45 * (1 - dist / MAX_DIST)) + ")";
            ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
        ctx.fillStyle = "rgba(200,70,30,0.85)";
        ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2); ctx.fill();
      }
      requestAnimationFrame(draw);
    }

    resize(); makeNodes(); draw();
    window.addEventListener("resize", function () { resize(); makeNodes(); });
  })();

  /* ================================================================
     3D VECTOR SPACE CANVAS
     ================================================================ */
  function initCanvas() {
    var canvas = document.getElementById("hero-canvas");
    if (!canvas) return;

    var ctx = canvas.getContext("2d");
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = 0, H = 0;
    var t = 0;
    var mouseX = 0, mouseY = 0;
    var targetMX = 0, targetMY = 0;
    var raf;

    /* -- Node config -- */
    var N = 60;
    var CONNECT_DIST_SQ = 115 * 115;
    var ACCENT = [232, 98, 59];   /* coral */
    var DARK   = [13, 13, 13];

    /* -- Create nodes -- */
    var nodes = [];
    for (var i = 0; i < N; i++) {
      var theta = Math.random() * Math.PI * 2;
      var phi   = Math.acos(2 * Math.random() - 1);
      var r     = 70 + Math.random() * 65;
      nodes.push({
        x:     r * Math.sin(phi) * Math.cos(theta),
        y:     r * Math.sin(phi) * Math.sin(theta),
        z:     r * Math.cos(phi),
        vx:    (Math.random() - 0.5) * 0.25,
        vy:    (Math.random() - 0.5) * 0.25,
        vz:    (Math.random() - 0.5) * 0.25,
        hot:   Math.random() < 0.22,
        phase: Math.random() * Math.PI * 2,
        size:  1.5 + Math.random() * 1.8
      });
    }

    /* -- Resize -- */
    function resize() {
      var rect = canvas.parentElement.getBoundingClientRect();
      W = rect.width;
      H = rect.height;
      canvas.width  = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.width  = W + "px";
      canvas.style.height = H + "px";
    }
    resize();
    window.addEventListener("resize", resize, { passive: true });

    /* -- Mouse parallax -- */
    var canvasWrap = document.querySelector(".hero__canvas-wrap");
    if (canvasWrap) {
      canvasWrap.addEventListener("mousemove", function (e) {
        var rect = canvasWrap.getBoundingClientRect();
        targetMX = ((e.clientX - rect.left) / rect.width  - 0.5) * 2;
        targetMY = ((e.clientY - rect.top)  / rect.height - 0.5) * 2;
      }, { passive: true });
      canvasWrap.addEventListener("mouseleave", function () { targetMX = 0; targetMY = 0; }, { passive: true });
    }

    /* -- 3D helpers -- */
    function rotateY(x, y, z, a) {
      var c = Math.cos(a), s = Math.sin(a);
      return { x: x * c + z * s, y: y, z: -x * s + z * c };
    }
    function rotateX(x, y, z, a) {
      var c = Math.cos(a), s = Math.sin(a);
      return { x: x, y: y * c - z * s, z: y * s + z * c };
    }
    function project(x, y, z) {
      var fov = 340;
      var d = fov / (fov + z + 120);
      return { sx: x * d * dpr + (W * dpr) / 2, sy: y * d * dpr + (H * dpr) / 2, scale: d };
    }

    /* -- Draw loop -- */
    function draw() {
      raf = requestAnimationFrame(draw);
      t += 0.007;

      /* Smooth mouse follow */
      mouseX += (targetMX - mouseX) * 0.06;
      mouseY += (targetMY - mouseY) * 0.06;

      var ry = t * 0.35 + mouseX * 0.55;
      var rx = Math.sin(t * 0.25) * 0.25 + mouseY * 0.4;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      /* Transform all nodes */
      var pts = nodes.map(function (n) {
        /* Gentle drift */
        n.x += n.vx * 0.12;
        n.y += n.vy * 0.12;
        n.z += n.vz * 0.12;
        /* Soft boundary — pull back if too far */
        var d2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (d2 > 155 * 155) {
          n.vx *= -0.6; n.vy *= -0.6; n.vz *= -0.6;
        }
        var p1 = rotateY(n.x, n.y, n.z, ry);
        var p2 = rotateX(p1.x, p1.y, p1.z, rx);
        var pr = project(p2.x, p2.y, p2.z);
        return { sx: pr.sx, sy: pr.sy, scale: pr.scale, rz: p2.z, node: n };
      });

      /* Sort back → front */
      pts.sort(function (a, b) { return a.rz - b.rz; });

      /* Draw edges */
      for (var i = 0; i < pts.length; i++) {
        for (var j = i + 1; j < pts.length; j++) {
          var a = pts[i], b = pts[j];
          var dx = a.sx - b.sx, dy = a.sy - b.sy;
          var dist2 = dx * dx + dy * dy;
          /* Scale the connect threshold by dpr */
          if (dist2 > CONNECT_DIST_SQ * dpr * dpr) continue;
          var dist  = Math.sqrt(dist2) / dpr;
          var alpha = (1 - dist / 115) * 0.4 * Math.min(a.scale, b.scale) * 2.5;
          var isHot = a.node.hot || b.node.hot;
          ctx.beginPath();
          ctx.moveTo(a.sx, a.sy);
          ctx.lineTo(b.sx, b.sy);
          if (isHot) {
            ctx.strokeStyle = "rgba(" + ACCENT[0] + "," + ACCENT[1] + "," + ACCENT[2] + "," + (alpha * 0.75) + ")";
            ctx.lineWidth = 1.2 * dpr;
          } else {
            ctx.strokeStyle = "rgba(180,180,180," + alpha + ")";
            ctx.lineWidth = 0.8 * dpr;
          }
          ctx.stroke();
        }
      }

      /* Draw nodes */
      pts.forEach(function (item) {
        var n = item.node;
        var pulse = (Math.sin(t * 2.2 + n.phase) * 0.5 + 0.5);
        var r = n.size * item.scale * 2.8 * dpr;

        if (n.hot) {
          /* Glow halo */
          var glowR = r * (3 + pulse * 1.5);
          var g = ctx.createRadialGradient(item.sx, item.sy, 0, item.sx, item.sy, glowR);
          g.addColorStop(0, "rgba(" + ACCENT[0] + "," + ACCENT[1] + "," + ACCENT[2] + "," + (0.28 * pulse * item.scale + 0.05) + ")");
          g.addColorStop(1, "rgba(" + ACCENT[0] + "," + ACCENT[1] + "," + ACCENT[2] + ",0)");
          ctx.beginPath();
          ctx.arc(item.sx, item.sy, glowR, 0, Math.PI * 2);
          ctx.fillStyle = g;
          ctx.fill();
          /* Core */
          ctx.beginPath();
          ctx.arc(item.sx, item.sy, r * (1 + pulse * 0.25), 0, Math.PI * 2);
          ctx.fillStyle = "rgba(" + ACCENT[0] + "," + ACCENT[1] + "," + ACCENT[2] + "," + Math.min(1, 0.75 * item.scale * 1.8 + 0.25) + ")";
          ctx.fill();
        } else {
          ctx.beginPath();
          ctx.arc(item.sx, item.sy, r, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(" + DARK[0] + "," + DARK[1] + "," + DARK[2] + "," + (0.35 * item.scale + 0.1) + ")";
          ctx.fill();
        }
      });
    }

    draw();

    /* Pause when tab hidden */
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) cancelAnimationFrame(raf);
      else draw();
    });
  }

  /* ================================================================
     INSTALL BLOCK TYPEWRITER
     ================================================================ */
  function initInstallTyper() {
    var rows = document.querySelectorAll(".install__row code");
    if (!rows.length) return;

    /* Collect texts, clear elements, add cursor spans */
    var queue = [];
    rows.forEach(function (code) {
      var text = code.textContent;
      code.textContent = "";
      var cursor = document.createElement("span");
      cursor.className = "install__cursor";
      cursor.setAttribute("aria-hidden", "true");
      code.appendChild(cursor);
      queue.push({ code: code, cursor: cursor, text: text });
    });

    /* Type each row sequentially */
    var CHAR_MS  = 48;   /* ms per character */
    var ROW_PAUSE = 520; /* pause between rows */
    var START    = 1200; /* wait for fadeUp to finish */

    function typeRow(idx, afterDelay) {
      if (idx >= queue.length) return;
      var q = queue[idx];

      setTimeout(function () {
        var i = 0;
        var iv = setInterval(function () {
          /* insert character before cursor */
          q.code.insertBefore(document.createTextNode(q.text[i]), q.cursor);
          i++;
          if (i >= q.text.length) {
            clearInterval(iv);
            typeRow(idx + 1, ROW_PAUSE);
          }
        }, CHAR_MS);
      }, afterDelay);
    }

    typeRow(0, START);
  }

  initInstallTyper();

  /* YouTube facade — load iframe on click */
  (function () {
    var facade = document.getElementById("yt-facade");
    if (!facade) return;
    function activate() {
      var iframe = document.createElement("iframe");
      iframe.src = "https://www.youtube.com/embed/UJ9MBALD380?autoplay=1&rel=0";
      iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
      iframe.allowFullscreen = true;
      var thumb = facade.querySelector(".yt-facade__thumb");
      var playBtn = facade.querySelector(".yt-facade__play");
      if (thumb) thumb.style.display = "none";
      if (playBtn) playBtn.style.display = "none";
      facade.appendChild(iframe);
      facade.removeEventListener("click", activate);
      facade.removeEventListener("keydown", onKey);
    }
    function onKey(e) { if (e.key === "Enter" || e.key === " ") activate(); }
    facade.addEventListener("click", activate);
    facade.addEventListener("keydown", onKey);
  })();

  initCanvas();

  /* ================================================================
     RECALL / LATENCY DONUT CHART
     ================================================================ */
  function initQualityChart() {
    var canvas = document.getElementById("quality-chart");
    if (!canvas) return;

    var dpr = Math.min(window.devicePixelRatio || 1, 2);

    var DATA = [
      { label: "dynavec",    recall: 90, lat: 10, color: "#e8623b" },
      { label: "Qdrant",     recall: 95, lat:  8, color: "#2a2a2a" },
      { label: "Milvus",     recall: 91, lat: 11, color: "#505050" },
      { label: "Weaviate",   recall: 93, lat: 17, color: "#737373" },
      { label: "Pinecone",   recall: 92, lat: 20, color: "#979797" },
      { label: "OpenSearch", recall: 85, lat: 50, color: "#c0c0c0" }
    ];

    var total = DATA.reduce(function(s, d) { return s + d.recall / d.lat; }, 0);
    DATA.forEach(function(d) { d.pct = (d.recall / d.lat) / total; });

    function draw() {
      var W = canvas.offsetWidth;
      var H = canvas.offsetHeight || 260;
      canvas.width  = W * dpr;
      canvas.height = H * dpr;

      var ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, W, H);

      var cx  = Math.round(W * 0.28);
      var cy  = Math.round(H / 2);
      var R   = Math.min(cx - 8, cy - 8, 105);
      var r   = Math.round(R * 0.55);

      /* -- Segments -- */
      var angle = -Math.PI / 2;
      DATA.forEach(function(d) {
        var span = d.pct * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, R, angle, angle + span);
        ctx.closePath();
        ctx.fillStyle = d.color;
        ctx.fill();
        d._a = angle;
        angle += span;
      });

      /* -- White dividers -- */
      DATA.forEach(function(d) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + R * Math.cos(d._a), cy + R * Math.sin(d._a));
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth   = 2.5;
        ctx.stroke();
        ctx.restore();
      });

      /* -- Donut hole -- */
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      /* -- Center text -- */
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle    = "#0d0d0d";
      ctx.font         = "700 12px Inter, system-ui, sans-serif";
      ctx.fillText("Recall", cx, cy - 14);
      ctx.fillText("÷ Latency", cx, cy);
      ctx.fillStyle = "#9a9a9a";
      ctx.font      = "9px JetBrains Mono, monospace, sans-serif";
      ctx.fillText("score index", cx, cy + 16);

      /* -- Legend -- */
      var lx  = cx + R + 22;
      var gap = (H - 24) / DATA.length;
      var ly  = 12 + gap * 0.18;

      DATA.forEach(function(d, i) {
        var y = ly + i * gap;

        /* swatch */
        ctx.fillStyle = d.color;
        ctx.beginPath();
        ctx.rect(lx, y + 1, 10, 10);
        ctx.fill();

        /* name */
        ctx.fillStyle    = d.label === "dynavec" ? "#0d0d0d" : "#3a3a3a";
        ctx.font         = (d.label === "dynavec" ? "700" : "500") + " 12px Inter, system-ui, sans-serif";
        ctx.textAlign    = "left";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(d.label, lx + 15, y + 10);

        /* stats */
        ctx.fillStyle = "#9a9a9a";
        ctx.font      = "9px JetBrains Mono, monospace, sans-serif";
        ctx.fillText(d.recall + "% recall · " + d.lat + "ms", lx + 15, y + 22);

        /* share */
        ctx.fillStyle = d.label === "dynavec" ? "#e8623b" : "#b0b0b0";
        ctx.font      = "700 10px JetBrains Mono, monospace, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(Math.round(d.pct * 100) + "%", W - 8, y + 10);
      });
    }

    draw();
    window.addEventListener("resize", draw, { passive: true });
  }

  initQualityChart();

  /* ================================================================
     SHARED LIGHT BACKGROUND HELPER
     ================================================================ */
  function drawLightBg(ctx, W, H) {
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "rgba(0,0,0,0.025)";
    for (var gx = 12; gx < W; gx += 24) {
      for (var gy = 12; gy < H; gy += 24) {
        ctx.beginPath();
        ctx.arc(gx, gy, 0.7, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  /* ================================================================
     BENCHMARK HEAT-MAP MATRIX
     ================================================================ */
  function initBenchMatrix() {
    var el = document.getElementById("bench-matrix");
    if (!el) return;

    var SERIES = [
      { name: "dynavec",    costs: [3, 3, 8, 50, 469],              accent: true, color: "#e8623b" },
      { name: "Pinecone",   costs: [9, 10, 27, 197, 1897],          color: "#505050" },
      { name: "Milvus",     costs: [150, 150, 900, 8100, 80550],    color: "#686868" },
      { name: "OpenSearch", costs: [701, 701, 877, 8423, 83708],    color: "#7e7e7e" },
      { name: "Qdrant",     costs: [160, 160, 960, 8640, 85920],    color: "#929292" },
      { name: "Weaviate",   costs: [175, 175, 1050, 9450, 93975],   color: "#a6a6a6" }
    ];
    var SCALE_LABELS = ["100K", "1M", "10M", "100M", "1B"];
    var minLog = Math.log10(3);
    var maxLog = Math.log10(93975);

    function fmtCost(v) {
      if (v >= 1000) return "$" + (v / 1000).toFixed(1) + "K";
      return "$" + v;
    }

    function draw() {
      var W   = el.offsetWidth;
      var H   = el.offsetHeight || 320;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      el.width  = W * dpr;
      el.height = H * dpr;
      var ctx = el.getContext("2d");
      ctx.scale(dpr, dpr);

      drawLightBg(ctx, W, H);

      var padL = 96, padR = 12, padT = 52, padB = 12;
      var cols = 5, rows = 6, gap = 3;
      var gridW = W - padL - padR;
      var gridH = H - padT - padB;
      var cellW = (gridW - gap * (cols - 1)) / cols;
      var cellH = (gridH - gap * (rows - 1)) / rows;

      /* column headers */
      ctx.fillStyle    = "#9a9a9a";
      ctx.font         = "700 10px JetBrains Mono, monospace, sans-serif";
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";
      for (var c = 0; c < cols; c++) {
        var cx = padL + c * (cellW + gap) + cellW / 2;
        ctx.fillText(SCALE_LABELS[c], cx, padT - 12);
      }

      /* cells */
      for (var row = 0; row < rows; row++) {
        var series = SERIES[row];
        var ry = padT + row * (cellH + gap);

        /* row label */
        ctx.fillStyle    = series.accent ? "#e8623b" : "#3a3a3a";
        ctx.font         = series.accent
          ? "700 11px Inter, system-ui, sans-serif"
          : "11px Inter, system-ui, sans-serif";
        ctx.textAlign    = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(series.name, padL - 8, ry + cellH / 2);

        for (var col = 0; col < cols; col++) {
          var cost = series.costs[col];
          var t    = Math.max(0, Math.min(1, (Math.log10(cost) - minLog) / (maxLog - minLog)));
          var cx2  = padL + col * (cellW + gap);

          if (series.accent) {
            /* coral tint: light at low cost, vivid at high */
            ctx.fillStyle = "rgba(232,98,59," + (0.12 + t * 0.72) + ")";
          } else {
            /* neutral gray → warm red as cost rises */
            var rv = Math.round(220 - 80 * t);
            var gv = Math.round(220 - 170 * t);
            var bv = Math.round(220 - 185 * t);
            ctx.fillStyle = "rgba(" + rv + "," + gv + "," + bv + ",0.85)";
          }

          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(cx2, ry, cellW, cellH, 4);
          } else {
            ctx.rect(cx2, ry, cellW, cellH);
          }
          ctx.fill();

          /* cell text: dark on light cells, white on dark/red cells */
          var textDark = series.accent ? t < 0.55 : t < 0.6;
          ctx.fillStyle    = textDark ? "rgba(0,0,0,0.72)" : "rgba(255,255,255,0.92)";
          ctx.font         = series.accent
            ? "700 9px JetBrains Mono, monospace, sans-serif"
            : "9px JetBrains Mono, monospace, sans-serif";
          ctx.textAlign    = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(fmtCost(cost), cx2 + cellW / 2, ry + cellH / 2);
        }
      }

      /* title */
      ctx.fillStyle    = "#0d0d0d";
      ctx.font         = "700 13px Inter, system-ui, sans-serif";
      ctx.textAlign    = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(
        "Monthly cost by scale · 1536-dim · 1M queries/mo",
        padL, 18
      );

      /* subtitle */
      ctx.fillStyle = "#9a9a9a";
      ctx.font      = "9px JetBrains Mono, monospace, sans-serif";
      ctx.fillText(
        "log-scale color · coral = dynavec · red = expensive",
        padL, 32
      );
    }

    draw();
    window.addEventListener("resize", draw, { passive: true });
  }

  /* ================================================================
     BENCHMARK LOG-LOG LINE CHART
     ================================================================ */
  function initBenchScale() {
    var el = document.getElementById("bench-scale");
    if (!el) return;

    var SERIES = [
      { name: "dynavec",    costs: [3, 3, 8, 50, 469],              accent: true, color: "#e8623b" },
      { name: "Pinecone",   costs: [9, 10, 27, 197, 1897],          color: "#333333" },
      { name: "Milvus",     costs: [150, 150, 900, 8100, 80550],    color: "#4f4f4f" },
      { name: "OpenSearch", costs: [701, 701, 877, 8423, 83708],    color: "#686868" },
      { name: "Qdrant",     costs: [160, 160, 960, 8640, 85920],    color: "#818181" },
      { name: "Weaviate",   costs: [175, 175, 1050, 9450, 93975],   color: "#999999" }
    ];
    var SCALE_LABELS = ["100K", "1M", "10M", "100M", "1B"];
    var Y_TICKS  = [1, 10, 100, 1000, 10000, 100000];
    var Y_LABELS = ["$1", "$10", "$100", "$1K", "$10K", "$100K"];
    var logYMin  = 0;
    var logYMax  = 5.2;

    function draw() {
      var W   = el.offsetWidth;
      var H   = el.offsetHeight || 260;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      el.width  = W * dpr;
      el.height = H * dpr;
      var ctx = el.getContext("2d");
      ctx.scale(dpr, dpr);

      drawLightBg(ctx, W, H);

      var padL = 52, padR = 12, padT = 44, padB = 30;
      var plotW = W - padL - padR;
      var plotH = H - padT - padB;

      function xPos(i) {
        return padL + (i / (SCALE_LABELS.length - 1)) * plotW;
      }
      function yPos(val) {
        var lv = Math.log10(Math.max(0.1, val));
        return padT + (1 - (lv - logYMin) / (logYMax - logYMin)) * plotH;
      }

      /* Y grid lines */
      for (var yi = 0; yi < Y_TICKS.length; yi++) {
        var yy = yPos(Y_TICKS[yi]);
        if (yy < padT || yy > padT + plotH) continue;

        ctx.strokeStyle = "rgba(0,0,0,0.06)";
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.moveTo(padL, yy);
        ctx.lineTo(padL + plotW, yy);
        ctx.stroke();

        ctx.fillStyle    = "#9a9a9a";
        ctx.font         = "9px JetBrains Mono, monospace, sans-serif";
        ctx.textAlign    = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(Y_LABELS[yi], padL - 5, yy);
      }

      /* X axis labels */
      ctx.fillStyle    = "#9a9a9a";
      ctx.font         = "9px JetBrains Mono, monospace, sans-serif";
      ctx.textAlign    = "center";
      ctx.textBaseline = "top";
      for (var xi = 0; xi < SCALE_LABELS.length; xi++) {
        ctx.fillText(SCALE_LABELS[xi], xPos(xi), padT + plotH + 4);
      }

      /* Non-accent lines first (so dynavec renders on top) */
      for (var si = 0; si < SERIES.length; si++) {
        var s = SERIES[si];
        if (s.accent) continue;
        ctx.strokeStyle = s.color;
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        for (var pi = 0; pi < s.costs.length; pi++) {
          var px = xPos(pi), py = yPos(s.costs[pi]);
          if (pi === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }

      /* dynavec on top */
      for (var si2 = 0; si2 < SERIES.length; si2++) {
        var s2 = SERIES[si2];
        if (!s2.accent) continue;

        ctx.save();
        ctx.shadowColor  = "rgba(232,98,59,0.35)";
        ctx.shadowBlur   = 6;
        ctx.strokeStyle  = s2.color;
        ctx.lineWidth    = 2.5;
        ctx.beginPath();
        for (var pi2 = 0; pi2 < s2.costs.length; pi2++) {
          var px2 = xPos(pi2), py2 = yPos(s2.costs[pi2]);
          if (pi2 === 0) ctx.moveTo(px2, py2); else ctx.lineTo(px2, py2);
        }
        ctx.stroke();
        ctx.restore();

        /* dots */
        for (var pi3 = 0; pi3 < s2.costs.length; pi3++) {
          var px3 = xPos(pi3), py3 = yPos(s2.costs[pi3]);
          ctx.beginPath();
          ctx.arc(px3, py3, 4, 0, Math.PI * 2);
          ctx.fillStyle   = s2.color;
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth   = 1.5;
          ctx.stroke();
        }
      }

      /* Compact legend — top-right */
      var legX = W - padR - 80;
      var legY = padT;
      for (var li = 0; li < SERIES.length; li++) {
        var ls   = SERIES[li];
        var lrow = legY + li * 17;

        ctx.save();
        ctx.globalAlpha = ls.accent ? 1 : 0.7;
        ctx.strokeStyle = ls.color;
        ctx.lineWidth   = ls.accent ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.moveTo(legX, lrow + 5);
        ctx.lineTo(legX + 12, lrow + 5);
        ctx.stroke();

        ctx.fillStyle    = ls.accent ? "#e8623b" : "#3a3a3a";
        ctx.font         = (ls.accent ? "700" : "400") +
          " 9px Inter, system-ui, sans-serif";
        ctx.textAlign    = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(ls.name, legX + 16, lrow + 5);
        ctx.restore();
      }

      /* Title */
      ctx.fillStyle    = "#0d0d0d";
      ctx.font         = "700 12px Inter, system-ui, sans-serif";
      ctx.textAlign    = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("Cost by scale · 768-dim · 1M queries/mo", padL, 16);

      ctx.fillStyle = "#9a9a9a";
      ctx.font      = "9px JetBrains Mono, monospace, sans-serif";
      ctx.fillText("log-log scale", padL, 30);
    }

    draw();
    window.addEventListener("resize", draw, { passive: true });
  }

  /* ================================================================
     BENCHMARK STORAGE BAR CHART
     ================================================================ */
  function initBenchStorage() {
    var el = document.getElementById("bench-storage");
    if (!el) return;

    var BAR_DATA = [
      { label: "100K", gib: (1e5 * 1536 * 4) / (1024 * 1024 * 1024) },
      { label: "1M",   gib: (1e6 * 1536 * 4) / (1024 * 1024 * 1024) },
      { label: "10M",  gib: (1e7 * 1536 * 4) / (1024 * 1024 * 1024) },
      { label: "100M", gib: (1e8 * 1536 * 4) / (1024 * 1024 * 1024) },
      { label: "1B",   gib: (1e9 * 1536 * 4) / (1024 * 1024 * 1024) }
    ];
    var Y_TICKS  = [0.1, 1, 10, 100, 1000];
    var Y_LABELS = ["100M", "1G", "10G", "100G", "1T"];
    var logYMin  = Math.log10(0.05);
    var logYMax  = Math.log10(10000);

    function fmtGib(v) {
      if (v >= 1024) return (v / 1024).toFixed(2) + " TiB";
      if (v >= 1)    return v.toFixed(2) + " GiB";
      return (v * 1024).toFixed(0) + " MiB";
    }

    function draw() {
      var W   = el.offsetWidth;
      var H   = el.offsetHeight || 260;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      el.width  = W * dpr;
      el.height = H * dpr;
      var ctx = el.getContext("2d");
      ctx.scale(dpr, dpr);

      drawLightBg(ctx, W, H);

      var padL = 54, padR = 12, padT = 44, padB = 30;
      var plotW = W - padL - padR;
      var plotH = H - padT - padB;

      function yPos(val) {
        var lv = Math.log10(Math.max(0.001, val));
        return padT + (1 - (lv - logYMin) / (logYMax - logYMin)) * plotH;
      }

      /* Y grid */
      for (var yi = 0; yi < Y_TICKS.length; yi++) {
        var yy = yPos(Y_TICKS[yi]);
        if (yy < padT - 2 || yy > padT + plotH + 2) continue;

        ctx.strokeStyle = "rgba(0,0,0,0.06)";
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.moveTo(padL, yy);
        ctx.lineTo(padL + plotW, yy);
        ctx.stroke();

        ctx.fillStyle    = "#9a9a9a";
        ctx.font         = "9px JetBrains Mono, monospace, sans-serif";
        ctx.textAlign    = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(Y_LABELS[yi], padL - 5, yy);
      }

      /* Bars */
      var groupW = plotW / BAR_DATA.length;
      var barW   = groupW * 0.55;

      for (var bi = 0; bi < BAR_DATA.length; bi++) {
        var d      = BAR_DATA[bi];
        var isLast = bi === BAR_DATA.length - 1;
        var bx     = padL + bi * groupW + (groupW - barW) / 2;
        var by     = yPos(d.gib);
        var bh     = padT + plotH - by;

        var barGrad = ctx.createLinearGradient(bx, by, bx, by + bh);
        if (isLast) {
          barGrad.addColorStop(0, "rgba(240,107,64,0.9)");
          barGrad.addColorStop(1, "rgba(180,60,20,0.7)");
        } else {
          var a0 = 0.3 + bi * 0.12;
          barGrad.addColorStop(0, "rgba(232,98,59," + a0 + ")");
          barGrad.addColorStop(1, "rgba(180,60,20," + Math.max(0, a0 - 0.1) + ")");
        }

        ctx.save();
        if (isLast) {
          ctx.shadowColor = "#e8623b";
          ctx.shadowBlur  = 12;
        }

        /* rounded top corners, flat bottom */
        var radius = 5;
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(bx, by, barW, bh, [radius, radius, 0, 0]);
        } else {
          ctx.moveTo(bx + radius, by);
          ctx.lineTo(bx + barW - radius, by);
          ctx.quadraticCurveTo(bx + barW, by, bx + barW, by + radius);
          ctx.lineTo(bx + barW, by + bh);
          ctx.lineTo(bx, by + bh);
          ctx.lineTo(bx, by + radius);
          ctx.quadraticCurveTo(bx, by, bx + radius, by);
          ctx.closePath();
        }
        ctx.fillStyle = barGrad;
        ctx.fill();
        ctx.restore();

        /* value label above bar */
        ctx.fillStyle    = isLast ? "#e8623b" : "#6a6a6a";
        ctx.font         = (isLast ? "700 " : "") + "9px JetBrains Mono, monospace, sans-serif";
        ctx.textAlign    = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(fmtGib(d.gib), bx + barW / 2, by - 3);

        /* x label */
        ctx.fillStyle    = "#9a9a9a";
        ctx.font         = "9px JetBrains Mono, monospace, sans-serif";
        ctx.textAlign    = "center";
        ctx.textBaseline = "top";
        ctx.fillText(d.label, bx + barW / 2, padT + plotH + 4);
      }

      /* footer note */
      ctx.fillStyle    = "rgba(232,98,59,0.75)";
      ctx.font         = "8px JetBrains Mono, monospace, sans-serif";
      ctx.textAlign    = "right";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(
        "dynavec stores this in S3 at ~$0.023/GB — not in RAM",
        W - padR, H - 4
      );

      /* title */
      ctx.fillStyle    = "#0d0d0d";
      ctx.font         = "700 12px Inter, system-ui, sans-serif";
      ctx.textAlign    = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("Raw float32 storage · 1536-dim", padL, 16);

      ctx.fillStyle = "#9a9a9a";
      ctx.font      = "9px JetBrains Mono, monospace, sans-serif";
      ctx.fillText("same raw footprint for every vector database", padL, 30);
    }

    draw();
    window.addEventListener("resize", draw, { passive: true });
  }

  initBenchMatrix();
  initBenchScale();
  initBenchStorage();

  /* ================================================================
     SCROLL REVEAL — IntersectionObserver
     ================================================================ */
  function initReveal() {
    if (!("IntersectionObserver" in window)) return;

    /* Auto-apply data-reveal + staggered delays to card grids */
    var grids = [
      { selector: "#why .grid--3 .card",        delay: 0.10 },
      { selector: "#quickstart .grid--3 .card",  delay: 0.10 },
      { selector: "#contribute .grid--3 .card",  delay: 0.10 },
      { selector: ".grid--2 .card",              delay: 0.12 },
      { selector: ".builton .svc",               delay: 0.15 },
      { selector: ".features__grid li",          delay: 0.04, type: "left" },
      { selector: ".hero__stats li",             delay: 0.10, type: "up"  }
    ];

    grids.forEach(function (g) {
      document.querySelectorAll(g.selector).forEach(function (el, i) {
        el.setAttribute("data-reveal", g.type || "");
        el.style.transitionDelay = (i * g.delay) + "s";
      });
    });

    /* Explicitly-marked reveal elements in HTML */
    var explicit = [
      { sel: ".builton__eyebrow",         type: "fade" },
      { sel: ".builton__note",            type: "fade" },
      { sel: "#why .section__title",      type: "" },
      { sel: "#why .section__lede",       type: "fade" },
      { sel: "#architecture .section__title", type: "" },
      { sel: "#architecture .section__lede",  type: "fade" },
      { sel: ".arch",                     type: "scale" },
      { sel: ".arch__notes > div",        type: "" },
      { sel: "#benchmarks .section__title", type: "" },
      { sel: "#benchmarks .section__lede", type: "fade" },
      { sel: ".fig",                      type: "scale" },
      { sel: ".table__scroll",            type: "" },
      { sel: "#quickstart .section__title", type: "" },
      { sel: "#quickstart .section__lede", type: "fade" },
      { sel: "#docs .section__title",     type: "" },
      { sel: "#docs .section__lede",      type: "fade" },
      { sel: ".tabs",                     type: "scale" },
      { sel: ".features",                 type: "" },
      { sel: "#contribute .section__title", type: "" },
      { sel: "#contribute .section__lede", type: "fade" }
    ];

    /* Stagger .fig and .arch__notes children */
    document.querySelectorAll(".fig").forEach(function (el, i) {
      el.style.transitionDelay = (i * 0.12) + "s";
    });
    document.querySelectorAll(".arch__notes > div").forEach(function (el, i) {
      el.style.transitionDelay = (i * 0.15) + "s";
    });

    explicit.forEach(function (item) {
      document.querySelectorAll(item.sel).forEach(function (el) {
        if (!el.hasAttribute("data-reveal")) el.setAttribute("data-reveal", item.type);
      });
    });

    /* Observer */
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -28px 0px" });

    document.querySelectorAll("[data-reveal]").forEach(function (el) {
      observer.observe(el);
    });
  }

  initReveal();

  /* ================================================================
     HERO PARALLAX (subtle depth on scroll)
     ================================================================ */
  var heroCol    = document.querySelector(".hero__col");
  var heroCanvas = document.querySelector(".hero__canvas-wrap");
  window.addEventListener("scroll", function () {
    var sy = window.scrollY;
    if (sy > 600) return;
    if (heroCol)    heroCol.style.transform    = "translateY(" + sy * 0.06 + "px)";
    if (heroCanvas) heroCanvas.style.transform = "translateY(" + sy * 0.03 + "px)";
  }, { passive: true });

})();
