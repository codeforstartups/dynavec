"""Security tests for dynavec.dashboard (localhost bind, token auth, CORS, redaction)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from dynavec.dashboard import _is_loopback_host, create_server, resolve_api_token
from dynavec.telemetry import TelemetryRecorder


@contextmanager
def _running_dashboard(
    recorder: TelemetryRecorder,
    *,
    api_token: str | None = None,
    allowed_origins: list[str] | None = None,
) -> Iterator[tuple[str, str]]:
    httpd, token = create_server(
        recorder,
        port=0,
        host="127.0.0.1",
        api_token=api_token,
        allowed_origins=allowed_origins,
    )
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", token
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


class TestLoopbackBinding:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]"])
    def test_loopback_hosts_accepted(self, host: str) -> None:
        assert _is_loopback_host(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "10.0.0.1", "example.com"])
    def test_non_loopback_hosts_rejected_by_default(self, host: str) -> None:
        rec = TelemetryRecorder()
        with pytest.raises(ValueError, match="non-loopback"):
            create_server(rec, port=0, host=host)


class TestTokenResolution:
    def test_generates_random_token_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DYNAVEC_DASHBOARD_TOKEN", raising=False)
        t1 = resolve_api_token()
        t2 = resolve_api_token()
        assert len(t1) >= 32
        assert t1 != t2

    def test_reads_env_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DYNAVEC_DASHBOARD_TOKEN", "env-secret-token")
        assert resolve_api_token() == "env-secret-token"

    def test_explicit_token_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DYNAVEC_DASHBOARD_TOKEN", "env-secret-token")
        assert resolve_api_token("explicit-token") == "explicit-token"

    def test_empty_explicit_token_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            resolve_api_token("   ")


class TestDashboardAuthenticationAndOrigins:
    def _sample_recorder(self) -> tuple[TelemetryRecorder, str]:
        rec = TelemetryRecorder()
        ev = rec.new_event(
            "search",
            namespace="kb",
            latency_ms=12.5,
            n_results=3,
            eval_faithfulness=0.9,
        )
        rec.record(ev)
        return rec, ev.id

    def test_api_endpoints_require_valid_token(self) -> None:
        rec, ev_id = self._sample_recorder()
        with _running_dashboard(rec, api_token="test-secret-123") as (base, token):
            endpoints = [
                "/api/metrics?window=3600",
                "/api/traces?limit=10",
                f"/api/trace/{ev_id}",
                "/api/eval/summary?window=3600",
                "/api/eval/runs?limit=10",
            ]
            for ep in endpoints:
                status, headers, body = _request(f"{base}{ep}")
                assert status == 401
                assert headers.get("WWW-Authenticate") == "Bearer"
                assert json.loads(body) == {"error": "unauthorized"}

                # Wrong token is rejected
                status_bad, _, _ = _request(
                    f"{base}{ep}",
                    headers={"Authorization": "Bearer wrong-token"},
                )
                assert status_bad == 401

                # Query-string token is NOT accepted
                sep = "&" if "?" in ep else "?"
                status_qs, _, _ = _request(f"{base}{ep}{sep}token={token}")
                assert status_qs == 401

                # Valid Bearer header succeeds
                status_bearer, resp_headers, _ = _request(
                    f"{base}{ep}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                assert status_bearer == 200
                assert resp_headers.get("Cache-Control") == "no-store"
                assert resp_headers.get("X-Content-Type-Options") == "nosniff"
                assert resp_headers.get("Referrer-Policy") == "no-referrer"

                # Valid X-Dynavec-Token header succeeds
                status_hdr, _, _ = _request(
                    f"{base}{ep}",
                    headers={"X-Dynavec-Token": token},
                )
                assert status_hdr == 200

    def test_origin_enforcement_and_cors_preflight(self) -> None:
        rec, _ = self._sample_recorder()
        with _running_dashboard(rec, api_token="tok-cors") as (base, token):
            # Non-loopback Origin is blocked with 403 even with a valid token
            status, _, body = _request(
                f"{base}/api/metrics",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Origin": "https://evil.example",
                },
            )
            assert status == 403
            assert json.loads(body) == {"error": "forbidden origin"}

            # Loopback Origin (e.g. Next.js dev server) is allowed and gets CORS headers
            status_ok, hdrs_ok, _ = _request(
                f"{base}/api/metrics",
                headers={
                    "X-Dynavec-Token": token,
                    "Origin": "http://localhost:3000",
                },
            )
            assert status_ok == 200
            assert hdrs_ok.get("Access-Control-Allow-Origin") == "http://localhost:3000"

            # Preflight OPTIONS succeeds for localhost and fails for external origins
            status_opt, hdrs_opt, _ = _request(
                f"{base}/api/metrics",
                method="OPTIONS",
                headers={"Origin": "http://localhost:3000"},
            )
            assert status_opt == 204
            assert hdrs_opt.get("Access-Control-Allow-Origin") == "http://localhost:3000"

            status_opt_bad, _, _ = _request(
                f"{base}/api/metrics",
                method="OPTIONS",
                headers={"Origin": "https://attacker.test"},
            )
            assert status_opt_bad == 403

    def test_invalid_query_params_return_400(self) -> None:
        rec, _ = self._sample_recorder()
        with _running_dashboard(rec, api_token="tok-params") as (base, token):
            hdr = {"X-Dynavec-Token": token}
            for bad_ep in [
                "/api/metrics?window=notanint",
                "/api/metrics?window=0",
                "/api/traces?limit=-5",
                "/api/eval/summary?window=abc",
                "/api/eval/runs?limit=999999",
            ]:
                status, _, body = _request(f"{base}{bad_ep}", headers=hdr)
                assert status == 400
                assert "error" in json.loads(body)


class TestCredentialIsolationAndRedaction:
    def test_aws_creds_never_exposed_and_event_secrets_redacted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_access_key = "AKIAIOSFODNN7EXAMPLE"
        fake_secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        fake_bearer = "Bearer secret_token_1234567890abcdef"
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", fake_access_key)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", fake_secret_key)

        rec = TelemetryRecorder()
        ev = rec.new_event(
            "search",
            namespace="kb",
            latency_ms=10.0,
            status="error",
            error=(
                f"ClientError for {fake_access_key} "
                f"secret_access_key='{fake_secret_key}' {fake_bearer}"
            ),
            query_preview=f"key={fake_access_key}",
            eval_faithfulness=0.5,
        )
        rec.record(ev)

        with _running_dashboard(rec, api_token="tok-redact") as (base, token):
            # HTML index never leaks AWS env credentials
            status_html, _, html_body = _request(f"{base}/")
            assert status_html == 200
            html_text = html_body.decode()
            assert fake_access_key not in html_text
            assert fake_secret_key not in html_text

            # JSON API responses redact any credentials embedded in telemetry events
            hdr = {"X-Dynavec-Token": token}
            for ep in ["/api/traces", f"/api/trace/{ev.id}", "/api/eval/runs"]:
                status, _, raw = _request(f"{base}{ep}", headers=hdr)
                assert status == 200
                text = raw.decode()
                assert fake_access_key not in text
                assert fake_secret_key not in text
                assert "secret_token_1234567890abcdef" not in text
                data: Any = json.loads(text)
                assert data  # payload still returned
