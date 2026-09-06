/* dynavec explainer — auto-advancing scenes with play/pause, dots, keyboard. */
(function () {
  "use strict";
  var scenes = Array.prototype.slice.call(document.querySelectorAll(".scene"));
  var fill = document.getElementById("barfill");
  var playBtn = document.getElementById("play");
  var dotsWrap = document.getElementById("dots");
  if (!scenes.length) return;

  var i = 0, playing = true, start = 0, raf = null;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // build dots
  var dots = scenes.map(function (_, n) {
    var d = document.createElement("button");
    d.className = "dot" + (n === 0 ? " is-active" : "");
    d.setAttribute("role", "tab");
    d.setAttribute("aria-label", "Scene " + (n + 1));
    d.addEventListener("click", function () { go(n); });
    dotsWrap.appendChild(d);
    return d;
  });

  function dur() { return parseInt(scenes[i].getAttribute("data-dur"), 10) || 7000; }

  function render() {
    scenes.forEach(function (s, n) { s.classList.toggle("is-active", n === i); });
    dots.forEach(function (d, n) { d.classList.toggle("is-active", n === i); });
  }

  function go(n) {
    i = (n + scenes.length) % scenes.length;
    start = performance.now();
    render();
  }
  function next() { go(i + 1); }
  function prev() { go(i - 1); }

  function tick(now) {
    if (!playing) return;
    var elapsed = now - start;
    var pct = Math.min(1, elapsed / dur());
    fill.style.width = (pct * 100) + "%";
    if (pct >= 1) { next(); }
    raf = requestAnimationFrame(tick);
  }

  function play() {
    playing = true; playBtn.textContent = "Pause";
    start = performance.now() - 0;
    cancelAnimationFrame(raf); raf = requestAnimationFrame(tick);
  }
  function pause() {
    playing = false; playBtn.textContent = "Play";
    cancelAnimationFrame(raf);
  }

  playBtn.addEventListener("click", function () { playing ? pause() : play(); });
  document.getElementById("next").addEventListener("click", function () { go(i + 1); if (playing) start = performance.now(); });
  document.getElementById("prev").addEventListener("click", function () { go(i - 1); if (playing) start = performance.now(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowRight") { next(); start = performance.now(); }
    else if (e.key === "ArrowLeft") { prev(); start = performance.now(); }
    else if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
  });

  render();
  if (reduce) { pause(); } else { play(); }
})();
