/* dynavec docs — code highlighting, copy buttons, sidebar toggle,
   heading anchors, "On this page" rail and ⌘K search. */
(function () {
  "use strict";

  // mobile sidebar toggle
  var toggle = document.querySelector(".side__toggle");
  var side = document.querySelector(".side");
  if (toggle && side) {
    toggle.addEventListener("click", function () { side.classList.toggle("is-open"); });
  }

  // grayscale-friendly Python highlighter (trusted, authored content)
  var KW = /\b(from|import|for|in|as|def|return|if|else|elif|not|and|or|None|True|False|with|class|lambda|print|is|await|async|yield)\b/g;
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
  document.querySelectorAll(".code code").forEach(function (el) {
    if (el.textContent.indexOf("<") === -1) el.innerHTML = highlight(el.textContent);
  });

  // copy buttons on code blocks — styled via .code__copy in docs.css
  document.querySelectorAll(".code").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.className = "code__copy";
    btn.textContent = "copy";
    btn.setAttribute("aria-label", "Copy code");
    btn.addEventListener("click", function () {
      var text = pre.innerText.replace(/\ncopy$/i, "").replace(/\ncopied$/i, "").trim();
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "copied";
        setTimeout(function () { btn.textContent = "copy"; }, 1300);
      });
    });
    pre.appendChild(btn);
  });

  // ---- heading anchors (same slug rule as build_docs.heading_id) ----
  function slugify(s) { return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, ""); }
  var heads = Array.prototype.slice.call(document.querySelectorAll(".doc h2"));
  heads.forEach(function (h) {
    if (!h.id) h.id = slugify(h.textContent);
    var a = document.createElement("a");
    a.className = "doc__anchor";
    a.href = "#" + h.id;
    a.setAttribute("aria-label", "Link to this section");
    a.textContent = "#";
    h.appendChild(a);
  });

  // ---- "On this page" rail with scroll-spy ----
  var toc = document.querySelector(".toc");
  if (toc && heads.length >= 2) {
    var html = '<p class="toc__title">On this page</p><ul>';
    heads.forEach(function (h) {
      var label = h.textContent.replace(/#$/, "").trim();
      html += '<li><a href="#' + h.id + '">' + label.replace(/</g, "&lt;") + "</a></li>";
    });
    toc.innerHTML = html + "</ul>";
    var links = toc.querySelectorAll("a");
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          links.forEach(function (l) { l.classList.toggle("is-active", l.getAttribute("href") === "#" + e.target.id); });
        });
      }, { rootMargin: "-90px 0px -70% 0px" });
      heads.forEach(function (h) { io.observe(h); });
    }
  } else if (toc) {
    toc.classList.add("is-empty");
  }

  // ---- ⌘K search ----
  var trigger = document.querySelector(".dsearch");
  var dialog = document.querySelector(".dsearch__dialog");
  var data = window.DV_SEARCH || [];
  if (!trigger || !dialog || !data.length) return;
  var input = dialog.querySelector("input");
  var list = dialog.querySelector(".dsearch__results");
  var results = [];
  var cursor = 0;

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function mark(s, terms) {
    var out = esc(s);
    terms.forEach(function (t) {
      if (!t) return;
      out = out.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
    });
    return out;
  }
  function score(e, terms) {
    var t = e.t.toLowerCase(), d = (e.d + " " + e.g).toLowerCase(), x = e.x.toLowerCase(), s = 0;
    for (var i = 0; i < terms.length; i++) {
      var q = terms[i];
      if (t.indexOf(q) === 0) s += 12; else if (t.indexOf(q) > -1) s += 8;
      else if (d.indexOf(q) > -1) s += 4; else if (x.indexOf(q) > -1) s += 1;
      else return 0;
    }
    return s + (e.u.indexOf("#") === -1 ? 1 : 0);
  }
  function run() {
    var terms = input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      results = data.filter(function (e) { return e.u.indexOf("#") === -1; }).slice(0, 8);
    } else {
      results = data.map(function (e) { return { e: e, s: score(e, terms) }; })
        .filter(function (r) { return r.s > 0; })
        .sort(function (a, b) { return b.s - a.s; })
        .slice(0, 10).map(function (r) { return r.e; });
    }
    cursor = 0;
    if (!results.length) { list.innerHTML = '<li class="dsearch__empty">No results for &ldquo;' + esc(input.value) + "&rdquo;</li>"; return; }
    list.innerHTML = results.map(function (e, i) {
      var sub = e.d || e.g;
      return '<li role="option"' + (i === 0 ? ' class="is-active" aria-selected="true"' : "") + '><a href="' + e.u + '">' +
        '<span class="dsearch__kind">' + (e.u.indexOf("#") > -1 ? "#" : "&#9636;") + "</span>" +
        "<span><strong>" + mark(e.t, terms) + "</strong>" + (sub ? "<small>" + mark(sub, terms) + "</small>" : "") + "</span></a></li>";
    }).join("");
  }
  function move(d) {
    var items = list.querySelectorAll("li[role=option]");
    if (!items.length) return;
    items[cursor].classList.remove("is-active");
    cursor = (cursor + d + items.length) % items.length;
    items[cursor].classList.add("is-active");
    items[cursor].scrollIntoView({ block: "nearest" });
  }
  function open() { dialog.hidden = false; document.body.classList.add("is-searching"); input.value = ""; run(); setTimeout(function () { input.focus(); }, 0); }
  function close() { dialog.hidden = true; document.body.classList.remove("is-searching"); trigger.focus(); }

  trigger.addEventListener("click", open);
  dialog.addEventListener("click", function (e) { if (e.target === dialog) close(); });
  input.addEventListener("input", run);
  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") {
      var a = list.querySelector("li.is-active a");
      if (a) { e.preventDefault(); window.location.href = a.getAttribute("href"); }
    }
  });
  document.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); if (dialog.hidden) open(); else close(); }
    else if (e.key === "/" && dialog.hidden && !/input|textarea/i.test(document.activeElement.tagName)) { e.preventDefault(); open(); }
    else if (e.key === "Escape" && !dialog.hidden) close();
  });
})();
