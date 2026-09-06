from dynavec import cli


class _FakeClient:
    def __init__(self, identity=None, error=None):
        self.identity = identity or {"Account": "123456789012"}
        self.error = error

    def get_caller_identity(self):
        if self.error:
            raise self.error
        return self.identity

    def get_index(self, **_kwargs):
        if self.error:
            raise self.error

    def describe_table(self, **_kwargs):
        if self.error:
            raise self.error


class _FakeSession:
    def __init__(self, error=None):
        self.error = error

    def client(self, service_name, **_kwargs):
        return _FakeClient(error=self.error)


def test_doctor_passes_with_configured_resources(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())

    result = cli.main(
        [
            "doctor",
            "--bucket",
            "vectors",
            "--index",
            "docs",
            "--table",
            "documents",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "[PASS] AWS credentials / STS identity" in output
    assert "[PASS] S3 Vectors index: docs" in output
    assert "[PASS] DynamoDB table: documents" in output
    assert "Doctor checks passed." in output


def test_doctor_fails_when_sts_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_session",
        lambda _profile, _region: _FakeSession(error=RuntimeError("access denied")),
    )

    result = cli.main(["doctor"])

    assert result == 1
    output = capsys.readouterr().out
    assert "[FAIL] AWS credentials / STS identity" in output
    assert "access denied" in output
    assert "Doctor checks failed." in output


def test_doctor_requires_bucket_and_index_together(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())

    result = cli.main(["doctor", "--bucket", "vectors"])

    assert result == 1
    assert "--bucket and --index must be provided together" in capsys.readouterr().out


def test_main_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "doctor" in capsys.readouterr().out