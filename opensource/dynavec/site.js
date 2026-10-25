/* ================================================
   dynavec landing — site.js
   Mega menu, mobile nav, lifecycle orbit.
   (Charts, calculator, tabs, copy, stars live in script.js.)
   ================================================ */

/* ---- Mega menu + mobile nav ---- */
(function () {
  const nav = document.getElementById("nav");
  if (!nav) return;
  const triggers = nav.querySelectorAll("[data-mega]");
  const burger = nav.querySelector(".snav__burger");

  function closeAll(except) {
    triggers.forEach((t) => {
      if (t === except) return;
      t.setAttribute("aria-expanded", "false");
      const panel = document.getElementById(t.getAttribute("aria-controls"));
      if (panel) panel.classList.remove("is-open");
    });
  }

  triggers.forEach((t) => {
    const panel = document.getElementById(t.getAttribute("aria-controls"));
    let hoverOpenedAt = 0;
    t.addEventListener("click", (e) => {
      e.stopPropagation();
      // A click that follows a hover-open should keep the menu open, not toggle it shut.
      const justHovered = Date.now() - hoverOpenedAt < 600;
      const open = justHovered || t.getAttribute("aria-expanded") !== "true";
      closeAll(t);
      t.setAttribute("aria-expanded", String(open));
      if (panel) panel.classList.toggle("is-open", open);
    });
    if (panel) {
      // hover-intent on desktop pointers only
      const li = t.closest("li");
      let timer;
      const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
      li.addEventListener("mouseenter", () => {
        if (!fine.matches || window.innerWidth <= 860) return;
        clearTimeout(timer);
        closeAll(t);
        if (t.getAttribute("aria-expanded") !== "true") hoverOpenedAt = Date.now();
        t.setAttribute("aria-expanded", "true");
        panel.classList.add("is-open");
      });
      li.addEventListener("mouseleave", () => {
        if (!fine.matches || window.innerWidth <= 860) return;
        timer = setTimeout(() => {
          t.setAttribute("aria-expanded", "false");
          panel.classList.remove("is-open");
        }, 160);
      });
      panel.addEventListener("click", (e) => {
        if (e.target.closest("a")) {
          closeAll();
          nav.classList.remove("is-open");
          if (burger) burger.setAttribute("aria-expanded", "false");
        }
      });
    }
  });

  document.addEventListener("click", (e) => {
    if (!nav.contains(e.target)) {
      closeAll();
      nav.classList.remove("is-open");
      if (burger) burger.setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeAll();
      nav.classList.remove("is-open");
      if (burger) burger.setAttribute("aria-expanded", "false");
    }
  });

  if (burger) {
    burger.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = !nav.classList.contains("is-open");
      nav.classList.toggle("is-open", open);
      burger.setAttribute("aria-expanded", String(open));
      if (!open) closeAll();
    });
    nav.querySelectorAll(".snav__links > li > a").forEach((a) =>
      a.addEventListener("click", () => {
        nav.classList.remove("is-open");
        burger.setAttribute("aria-expanded", "false");
      })
    );
  }
})();

/* ---- Lifecycle orbit (auto-advance, pause on hover/focus) ---- */
(function () {
  const root = document.querySelector("[data-cycle]");
  if (!root) return;
  const section = root.closest(".cycle");
  const nodes = root.querySelectorAll(".orbit__node");
  const panes = root.querySelectorAll(".cycle__pane");
  const shots = root.querySelectorAll(".cycle__shot");
  const bars = section.querySelectorAll(".cycle__progress button");
  const DURATION = 7000;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  section.style.setProperty("--cycle-ms", DURATION + "ms");

  let current = 0;
  let timer = null;
  let paused = false;
  let visible = false;

  function show(i) {
    current = (i + nodes.length) % nodes.length;
    nodes.forEach((n, k) => {
      n.classList.toggle("is-active", k === current);
      n.setAttribute("aria-selected", String(k === current));
    });
    panes.forEach((p, k) => p.classList.toggle("is-active", k === current));
    shots.forEach((s, k) => s.classList.toggle("is-active", k === current));
    bars.forEach((b, k) => {
      b.classList.remove("is-active");
      if (k === current) {
        void b.offsetWidth; // restart the progress animation
        b.classList.add("is-active");
      }
    });
    schedule();
  }

  function schedule() {
    clearTimeout(timer);
    if (reduced || paused || !visible) return;
    timer = setTimeout(() => show(current + 1), DURATION);
  }

  function setPaused(p) {
    paused = p;
    section.classList.toggle("is-paused", p);
    if (p) clearTimeout(timer);
    else show(current); // restart the timer and progress bar from zero
  }

  nodes.forEach((n, k) => n.addEventListener("click", () => show(k)));
  bars.forEach((b, k) => b.addEventListener("click", () => show(k)));

  root.addEventListener("keydown", (e) => {
    if (!e.target.classList.contains("orbit__node")) return;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") { e.preventDefault(); show(current + 1); nodes[current].focus(); }
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") { e.preventDefault(); show(current - 1); nodes[current].focus(); }
  });

  root.addEventListener("mouseenter", () => setPaused(true));
  root.addEventListener("mouseleave", () => setPaused(false));
  root.addEventListener("focusin", () => setPaused(true));
  root.addEventListener("focusout", (e) => { if (!root.contains(e.relatedTarget)) setPaused(false); });

  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      if (visible) show(current); else clearTimeout(timer);
    }, { threshold: 0.35 }).observe(section);
  } else {
    visible = true;
    show(0);
  }
})();
