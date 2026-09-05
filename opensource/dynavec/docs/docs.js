/* dynavec docs — code highlighting, sidebar toggle, copy buttons. */
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

  // copy buttons on code blocks
  document.querySelectorAll(".code").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.className = "code__copy";
    btn.textContent = "copy";
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(pre.innerText).then(function () {
        btn.textContent = "copied"; setTimeout(function () { btn.textContent = "copy"; }, 1300);
      });
    });
    pre.style.position = "relative";
    btn.style.cssText = "position:absolute;top:8px;right:8px;font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.05em;border:1px solid var(--line);background:var(--surface);color:var(--muted);padding:4px 9px;border-radius:6px;cursor:pointer;";
    pre.appendChild(btn);
  });
})();
