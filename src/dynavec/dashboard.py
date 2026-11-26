"""Observability dashboard server for dynavec.

Serves a comprehensive, on-brand (dynavec landing-page theme) telemetry UI backed
by a :class:`~dynavec.telemetry.TelemetryRecorder`. All numbers are **real** —
they come from actual operations recorded on the client, not a simulator.

    from dynavec.telemetry import TelemetryRecorder
    from dynavec import Dynavec, DynavecConfig
    rec = TelemetryRecorder()
    db = Dynavec(cfg, embedder=emb, telemetry=rec)
    ...  # run your searches
    from dynavec.dashboard import serve
    serve(rec, port=8778)

Vanilla JS + inline SVG charts (no CDN, no build step). Endpoints:
    GET /                      the dashboard
    GET /api/metrics?window=   aggregated stats + histogram
    GET /api/traces?limit=&op=&namespace=&status=   recent events
    GET /api/trace/{id}        one event
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .logging import redact_secrets
from .telemetry import TelemetryRecorder, aggregate, aggregate_eval

_TOKEN_ENV_VAR = "DYNAVEC_DASHBOARD_TOKEN"
_MAX_WINDOW_SECONDS = 31_536_000  # 1 year
_MAX_LIMIT = 10_000


def _is_loopback_host(host: str) -> bool:
    """Return True if *host* resolves syntactically to a loopback address."""
    cleaned = host.strip().lower()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if cleaned == "localhost":
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def _is_allowed_origin(origin: str, allowed_origins: Sequence[str] | None = None) -> bool:
    """Return True if *origin* is a loopback origin or explicitly allowlisted."""
    if allowed_origins and origin in allowed_origins:
        return True
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return _is_loopback_host(parsed.hostname)


def resolve_api_token(api_token: str | None = None) -> str:
    """Resolve the dashboard API token from argument, env var, or random generation."""
    if api_token is not None:
        cleaned = api_token.strip()
        if not cleaned:
            raise ValueError("api_token must be a non-empty string")
        return cleaned
    env_token = os.environ.get(_TOKEN_ENV_VAR, "").strip()
    if env_token:
        return env_token
    return secrets.token_urlsafe(32)


_INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>dynavec · Observability</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#fbfaf8;--surface:#fff;--fg:#14110f;--muted:#6f6862;--faint:#a99f97;--line:#ece6df;
--accent:#e8623b;--accent-ink:#b8472a;--accent-soft:#fdeee8;--ok:#2f7d5b;--err:#b8472a;
--sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
--mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;--shadow:0 6px 30px rgba(20,17,15,.07)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;border-top:3px solid var(--accent);-webkit-font-smoothing:antialiased}
a{color:inherit}
.top{display:flex;align-items:center;gap:16px;padding:12px 20px;background:var(--surface);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10}
.brand{display:flex;align-items:center;gap:9px;font-family:var(--mono);font-weight:700;font-size:16px}
.brand svg{color:var(--accent)}
.top .sub{color:var(--muted);font-size:13px;margin-right:auto}
.rangebtns{display:flex;border:1.5px solid var(--line-strong,#14110f);border-radius:8px;overflow:hidden}
.rangebtns button{font-family:var(--mono);font-size:12px;border:0;background:var(--surface);padding:7px 12px;cursor:pointer;color:var(--muted);border-right:1px solid var(--line)}
.rangebtns button:last-child{border-right:0}
.rangebtns button.on{background:var(--accent);color:#fff}
.toggle{font-family:var(--mono);font-size:12px;border:1.5px solid #14110f;border-radius:8px;background:var(--surface);padding:7px 12px;cursor:pointer}
.toggle.on{background:#14110f;color:#fff}
.layout{display:grid;grid-template-columns:210px 1fr;min-height:calc(100vh - 52px)}
.side{border-right:1px solid var(--line);padding:18px 12px;background:var(--surface)}
.side h4{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin:16px 8px 8px}
.side a{display:block;padding:7px 10px;border-radius:7px;text-decoration:none;color:var(--muted);font-size:13.5px;border-left:2px solid transparent}
.side a.on{background:var(--accent-soft);color:var(--accent-ink);border-left-color:var(--accent);font-weight:600}
.side a.soon{opacity:.55;cursor:default}
.side .tag{font-size:10px;font-family:var(--mono);color:var(--faint);float:right}
.main{padding:22px 24px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:20px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.kpi .l{font-size:12px;color:var(--muted);margin-bottom:8px}
.kpi .v{font-family:var(--mono);font-size:26px;font-weight:700;letter-spacing:-.02em}
.kpi .v small{font-size:13px;color:var(--muted);font-weight:400}
.kpi .v.win{color:var(--accent-ink)}
.panels{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;margin-bottom:20px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card h3{margin:0 0 14px;font-size:14px}
.card h3 .hint{font-family:var(--mono);font-size:11px;color:var(--faint);font-weight:400}
.tablecard{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.tablehead{display:flex;align-items:center;gap:12px;padding:14px 18px;border-bottom:1px solid var(--line)}
.tablehead h3{margin:0;font-size:14px}
.tablehead .filters{margin-left:auto;display:flex;gap:8px}
.tablehead select,.tablehead input{font-family:var(--mono);font-size:12px;border:1px solid var(--line);border-radius:7px;padding:6px 9px;background:var(--bg)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);padding:10px 18px;background:#faf6f1;border-bottom:1px solid var(--line)}
td{padding:10px 18px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
tr.row{cursor:pointer}
tr.row:hover{background:var(--accent-soft)}
.pill{font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line)}
.pill.search{background:#eef3ff;color:#3b5bdb;border-color:#dbe3ff}
.pill.graph_search{background:#f3eeff;color:#7048e8;border-color:#e5dbff}
.pill.upsert{background:#eafaf1;color:#2f7d5b;border-color:#d3f0e0}
.st-ok{color:var(--ok)}
.st-error{color:var(--err);font-weight:600}
.hit{color:var(--ok)}.miss{color:var(--muted)}.na{color:var(--faint)}
.mono{font-family:var(--mono)}
.drawer{position:fixed;top:0;right:0;height:100vh;width:min(460px,92vw);background:var(--surface);border-left:1px solid var(--line);box-shadow:var(--shadow);transform:translateX(100%);transition:transform .2s ease;z-index:20;overflow-y:auto;padding:22px}
.drawer.open{transform:none}
.drawer h3{margin:0 0 4px;font-size:16px}
.drawer .close{position:absolute;top:16px;right:18px;border:0;background:none;font-size:20px;cursor:pointer;color:var(--muted)}
.kv{display:grid;grid-template-columns:130px 1fr;gap:8px 14px;margin-top:16px;font-size:13px}
.kv .k{color:var(--muted);font-family:var(--mono);font-size:12px}
.kv .val{font-variant-numeric:tabular-nums;word-break:break-word}
.wf{margin-top:18px}
.wf .bar{height:10px;border-radius:5px;background:var(--accent);margin:4px 0}
.empty{padding:40px;text-align:center;color:var(--muted)}
.note{color:var(--faint);font-size:11.5px;font-family:var(--mono);margin-top:8px}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.panels{grid-template-columns:1fr}.layout{grid-template-columns:1fr}.side{display:none}}
</style></head>
<body>
<div class="top">
  <span class="brand"><svg width="18" height="18" viewBox="0 0 22 22"><rect x="1" y="1" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5"/><line x1="1" y1="11" x2="21" y2="11" stroke="currentColor" stroke-width="1.5"/><line x1="11" y1="1" x2="11" y2="21" stroke="currentColor" stroke-width="1.5"/><circle cx="6" cy="6" r="2" fill="currentColor"/><circle cx="16" cy="16" r="2" fill="currentColor"/></svg>dynavec</span>
  <span class="sub">Observability</span>
  <div class="rangebtns" id="ranges">
    <button data-w="1800">30m</button><button data-w="3600" class="on">1h</button>
    <button data-w="21600">6h</button><button data-w="86400">24h</button>
  </div>
  <button class="toggle on" id="auto">Auto-refresh</button>
</div>
<div class="layout">
  <nav class="side">
    <h4>Observability</h4>
    <a class="on" href="#">Tracing</a>
    <a href="#" onclick="return false">Latency</a>
    <a href="#" onclick="return false">Cost</a>
    <h4>Evaluation</h4>
    <a class="soon">Scores <span class="tag">soon</span></a>
    <a class="soon">Faithfulness <span class="tag">soon</span></a>
    <h4>Resources</h4>
    <a class="soon">Buckets &amp; Indexes <span class="tag">soon</span></a>
    <a class="soon">Namespaces <span class="tag">soon</span></a>
  </nav>
  <main class="main">
    <div class="kpis" id="kpis"></div>
    <div class="panels">
      <div class="card"><h3>Query volume <span class="hint" id="histhint"></span></h3><div id="hist"></div></div>
      <div class="card"><h3>Latency percentiles <span class="hint">ms</span></h3><div id="lat"></div></div>
    </div>
    <div class="tablecard">
      <div class="tablehead">
        <h3>Traces</h3>
        <div class="filters">
          <select id="fop"><option value="">all ops</option><option>search</option><option>graph_search</option><option>upsert</option></select>
          <select id="fstatus"><option value="">any status</option><option>ok</option><option>error</option></select>
          <input id="fns" placeholder="namespace…" size="12">
        </div>
      </div>
      <div id="table"></div>
    </div>
    <p class="note" id="note"></p>
  </main>
</div>
<div class="drawer" id="drawer"><button class="close" onclick="closeDrawer()">&times;</button><div id="drawerbody"></div></div>
<script>
const TOKEN=__DYNAVEC_API_TOKEN__;
let W=3600, AUTO=true, TIMER=null;
const $=s=>document.querySelector(s);
function apiFetch(url){return fetch(url,{cache:'no-store',headers:TOKEN?{'X-Dynavec-Token':TOKEN}:{}})}
function fmt(n,d=0){return n==null?'—':Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}
function bars(vals,color){const n=vals.length,max=Math.max(1,...vals),w=100/n;
  let r=`<svg viewBox="0 0 100 40" preserveAspectRatio="none" style="width:100%;height:120px">`;
  vals.forEach((v,i)=>{const h=v/max*38;r+=`<rect x="${i*w+w*0.12}" y="${40-h}" width="${w*0.76}" height="${h||0.4}" fill="${color}" rx="0.6"/>`});
  return r+`</svg>`}
function latChart(m){const items=[['p50',m.p50,'#2f7d5b'],['p95',m.p95,'#e8623b'],['p99',m.p99,'#b8472a']];
  const max=Math.max(1,m.p99,m.p95,m.p50);let r='<div style="display:flex;flex-direction:column;gap:14px;padding:6px 0 2px">';
  for(const[l,v,c]of items){const pct=v/max*100;
    r+=`<div><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px"><span class="mono" style="color:${c}">${l}</span><span class="mono">${fmt(v,1)} ms</span></div>
    <div style="height:10px;background:#f0ece6;border-radius:5px"><div style="height:100%;width:${pct}%;background:${c};border-radius:5px"></div></div></div>`}
  return r+'</div>'}
function kpi(l,v,win){return `<div class="kpi"><div class="l">${l}</div><div class="v ${win?'win':''}">${v}</div></div>`}
async function load(){
  const m=await (await apiFetch('/api/metrics?window='+W)).json();
  $('#kpis').innerHTML=[
    kpi('Queries / min',fmt(m.qpm,1),true),
    kpi('p95 latency',fmt(m.p95,0)+'<small> ms</small>'),
    kpi('Cache hit rate',m.cache_hit_rate==null?'—':fmt(m.cache_hit_rate,1)+'<small>%</small>',true),
    kpi('Avg results',fmt(m.avg_results,1)),
    kpi('Error rate',fmt(m.error_rate,1)+'<small>%</small>')
  ].join('');
  $('#hist').innerHTML=bars(m.histogram,'#e8623b');
  $('#histhint').textContent=(m.total||0)+' traces · '+Math.round(m.bucket_width_s)+'s buckets';
  $('#lat').innerHTML=latChart(m);
  await loadTable();
}
async function loadTable(){
  const q=new URLSearchParams({limit:100});
  if($('#fop').value)q.set('op',$('#fop').value);
  if($('#fstatus').value)q.set('status',$('#fstatus').value);
  if($('#fns').value)q.set('namespace',$('#fns').value);
  const rows=await (await apiFetch('/api/traces?'+q)).json();
  if(!rows.length){$('#table').innerHTML='<div class="empty">No traces yet. Run some searches with <span class="mono">Dynavec(..., telemetry=recorder)</span>.</div>';return}
  let h='<table><thead><tr><th>Start</th><th>Op</th><th>Namespace</th><th>Latency</th><th>Results</th><th>Cache</th><th>Rank</th><th>Status</th></tr></thead><tbody>';
  for(const e of rows){const t=new Date(e.ts*1000).toLocaleTimeString();
    const cache=e.cache_hit==null?'<span class="na">—</span>':(e.cache_hit?'<span class="hit">hit</span>':'<span class="miss">miss</span>');
    const rank=[e.rescore,e.rerank].filter(Boolean).join('+')||'<span class="na">—</span>';
    h+=`<tr class="row" onclick="openTrace('${e.id}')"><td class="mono">${t}</td><td><span class="pill ${e.op}">${e.op}</span></td><td>${e.namespace}</td><td class="mono">${fmt(e.latency_ms,1)} ms</td><td class="mono">${e.n_results}</td><td>${cache}</td><td class="mono">${rank}</td><td class="st-${e.status}">${e.status}</td></tr>`}
  $('#table').innerHTML=h+'</tbody></table>';
}
async function openTrace(id){const e=await (await apiFetch('/api/trace/'+id)).json();
  const row=(k,v)=>`<div class="k">${k}</div><div class="val">${v==null?'—':v}</div>`;
  $('#drawerbody').innerHTML=`<h3><span class="pill ${e.op}">${e.op}</span> trace</h3>
  <div class="mono" style="color:var(--muted);font-size:12px">${e.id} · ${new Date(e.ts*1000).toLocaleString()}</div>
  <div class="kv">${row('Namespace',e.namespace)}${row('Latency',fmt(e.latency_ms,2)+' ms')}${row('Results',e.n_results)}${row('top_k',e.top_k)}
  ${row('Cache',e.cache_hit==null?'no cache':(e.cache_hit?'hit':'miss'))}${row('Filtered',e.filtered)}${row('Rescore',e.rescore)}${row('Rerank',e.rerank)}
  ${row('Top score',e.score_top)}${row('Mean score',e.score_mean)}${row('Status',e.status)}${e.error?row('Error',e.error):''}${e.query_preview?row('Query',e.query_preview):''}</div>
  <div class="wf"><div class="k mono" style="color:var(--muted);font-size:12px;margin-bottom:6px">End-to-end latency</div><div class="bar" style="width:100%"></div></div>`;
  $('#drawer').classList.add('open');
}
function closeDrawer(){$('#drawer').classList.remove('open')}
$('#ranges').addEventListener('click',e=>{if(e.target.dataset.w){W=+e.target.dataset.w;[...$('#ranges').children].forEach(b=>b.classList.toggle('on',b===e.target));load()}});
$('#auto').addEventListener('click',()=>{AUTO=!AUTO;$('#auto').classList.toggle('on',AUTO);schedule()});
['#fop','#fstatus','#fns'].forEach(s=>$(s).addEventListener('input',loadTable));
function schedule(){if(TIMER)clearInterval(TIMER);if(AUTO)TIMER=setInterval(load,4000)}
load();schedule();
</script>
</body></html>"""


def _parse_int_param(
    qs: dict[str, list[str]],
    name: str,
    default: int,
    *,
    min_value: int = 1,
    max_value: int,
) -> int:
    raw = qs.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {name} parameter") from exc
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}")
    return value


def _extract_request_token(headers: Any) -> str | None:
    if headers is None:
        return None
    auth = headers.get("Authorization")
    if isinstance(auth, str):
        parts = auth.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
            if token:
                return token
    header_token = headers.get("X-Dynavec-Token")
    if isinstance(header_token, str):
        token = header_token.strip()
        if token:
            return token
    return None


def _make_handler(
    recorder: TelemetryRecorder,
    *,
    api_token: str | None = None,
    allowed_origins: Sequence[str] | None = None,
) -> type[BaseHTTPRequestHandler]:
    rendered_html = _INDEX_HTML.replace("__DYNAVEC_API_TOKEN__", json.dumps(api_token or ""))
    origin_allowlist = tuple(allowed_origins) if allowed_origins else ()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            pass

        def _send(
            self,
            code: int,
            body: str | bytes,
            ctype: str = "application/json",
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            if data:
                self.wfile.write(data)

        def _cors_headers(self) -> tuple[bool, dict[str, str]]:
            headers = getattr(self, "headers", None)
            origin = headers.get("Origin") if headers is not None else None
            if not origin:
                return True, {}
            if not _is_allowed_origin(origin, origin_allowlist):
                return False, {}
            return True, {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Headers": "Authorization, X-Dynavec-Token, Content-Type",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Vary": "Origin",
            }

        def _send_json(
            self,
            code: int,
            payload: Any,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            safe_payload = redact_secrets(payload)
            body = json.dumps(safe_payload)
            if extra_headers:
                self._send(code, body, "application/json", extra_headers=extra_headers)
            else:
                self._send(code, body)

        def do_OPTIONS(self) -> None:
            origin_ok, cors_headers = self._cors_headers()
            if not origin_ok:
                self._send_json(403, {"error": "forbidden origin"})
                return
            self._send(204, b"", "text/plain", extra_headers=cors_headers)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path, qs = parsed.path, parse_qs(parsed.query)
            if path == "/" or path == "/index.html":
                self._send(200, rendered_html, "text/html; charset=utf-8")
                return

            if path.startswith("/api/"):
                origin_ok, cors_headers = self._cors_headers()
                if not origin_ok:
                    self._send_json(403, {"error": "forbidden origin"})
                    return
                if api_token is not None:
                    provided = _extract_request_token(getattr(self, "headers", None))
                    if provided is None or not hmac.compare_digest(provided, api_token):
                        auth_headers = {**cors_headers, "WWW-Authenticate": "Bearer"}
                        self._send_json(401, {"error": "unauthorized"}, extra_headers=auth_headers)
                        return
            else:
                cors_headers = {}

            try:
                if path == "/api/metrics":
                    window = _parse_int_param(qs, "window", 3600, max_value=_MAX_WINDOW_SECONDS)
                    self._send_json(
                        200,
                        aggregate(recorder.snapshot(), window),
                        extra_headers=cors_headers,
                    )
                    return
                if path == "/api/traces":
                    limit = _parse_int_param(qs, "limit", 100, max_value=_MAX_LIMIT)
                    evs = recorder.events(
                        limit=limit,
                        op=(qs.get("op", [None])[0] or None),
                        namespace=(qs.get("namespace", [None])[0] or None),
                        status=(qs.get("status", [None])[0] or None),
                    )
                    self._send_json(
                        200,
                        [e.to_dict() for e in evs],
                        extra_headers=cors_headers,
                    )
                    return
                if path.startswith("/api/trace/"):
                    ev = recorder.get(path.rsplit("/", 1)[-1])
                    if ev is None:
                        self._send_json(404, {"error": "not found"}, extra_headers=cors_headers)
                        return
                    self._send_json(200, ev.to_dict(), extra_headers=cors_headers)
                    return
                if path == "/api/eval/summary":
                    window = _parse_int_param(qs, "window", 86400, max_value=_MAX_WINDOW_SECONDS)
                    self._send_json(
                        200,
                        aggregate_eval(recorder.snapshot(), window),
                        extra_headers=cors_headers,
                    )
                    return
                if path == "/api/eval/runs":
                    limit = _parse_int_param(qs, "limit", 50, max_value=_MAX_LIMIT)
                    all_events = recorder.events(limit=recorder._events.maxlen or 10000)
                    eval_runs = [
                        e.to_dict()
                        for e in all_events
                        if e.eval_faithfulness is not None
                        or e.eval_relevance is not None
                        or e.eval_recall is not None
                        or e.eval_mrr is not None
                        or e.eval_ndcg is not None
                    ][:limit]
                    self._send_json(200, eval_runs, extra_headers=cors_headers)
                    return
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)}, extra_headers=cors_headers)
                return

            self._send_json(404, {"error": "not found"}, extra_headers=cors_headers)

    return Handler


def create_server(
    recorder: TelemetryRecorder,
    port: int = 8778,
    host: str = "127.0.0.1",
    *,
    api_token: str | None = None,
    allow_remote: bool = False,
    allowed_origins: Sequence[str] | None = None,
) -> tuple[HTTPServer, str]:
    """Create a hardened dashboard ``HTTPServer`` and return ``(server, token)``.

    By default, binding is restricted to loopback addresses (``127.0.0.1``,
    ``localhost``, ``::1``) and all ``/api/*`` endpoints require a server-side
    API token passed via ``Authorization: Bearer <token>`` or
    ``X-Dynavec-Token: <token>``. Pass ``allow_remote=True`` only when
    intentionally exposing the dashboard behind a trusted network boundary.
    """
    if not _is_loopback_host(host) and not allow_remote:
        raise ValueError(
            f"Refusing to bind dashboard to non-loopback host {host!r} without "
            "allow_remote=True. Keep the dashboard bound to 127.0.0.1 or localhost."
        )
    token = resolve_api_token(api_token)
    handler = _make_handler(
        recorder,
        api_token=token,
        allowed_origins=allowed_origins,
    )
    return HTTPServer((host, port), handler), token


def serve(
    recorder: TelemetryRecorder,
    port: int = 8778,
    host: str = "127.0.0.1",
    *,
    api_token: str | None = None,
    allow_remote: bool = False,
    allowed_origins: Sequence[str] | None = None,
) -> None:
    """Start the dashboard server (blocking), bound to localhost with token auth."""
    httpd, token = create_server(
        recorder,
        port=port,
        host=host,
        api_token=api_token,
        allow_remote=allow_remote,
        allowed_origins=allowed_origins,
    )
    bound_host_raw, bound_port = httpd.server_address[:2]
    bound_host = (
        bound_host_raw.decode()
        if isinstance(bound_host_raw, (bytes, bytearray))
        else bound_host_raw
    )
    print(f"dynavec observability dashboard: http://{bound_host}:{bound_port}/")
    print(f"dynavec API token (X-Dynavec-Token / Bearer): {token}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    finally:
        httpd.server_close()
