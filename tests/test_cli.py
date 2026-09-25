from dynavec import __version__, cli


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
    output = capsys.readouterr().out
    assert "doctor" in output
    assert "version" in output


def test_version_prints_installed_package_version(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out == f"{__version__}\n"


def test_export_missing_resources_fails(capsys):
    result = cli.main(["export"])
    assert result == 1
    err = capsys.readouterr().err
    assert "[FAIL] Missing required resource configuration" in err


def test_import_missing_resources_fails(capsys):
    result = cli.main(["import"])
    assert result == 1
    err = capsys.readouterr().err
    assert "[FAIL] Missing required resource configuration" in err


def test_export_to_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())

    out_file = tmp_path / "export.jsonl"
    calls = []

    class DummyDynavec:
        def __init__(self, config, boto_session=None):
            self.config = config

        def export_namespace(self, output, namespace="default"):
            calls.append((output, namespace))
            with open(output, "w", encoding="utf-8") as f:
                f.write('{"id": "doc1", "vector": [0.1, 0.2], "text": "hi", "metadata": {}}\n')
            return 1

    monkeypatch.setattr(cli, "Dynavec", DummyDynavec)

    result = cli.main(
        [
            "export",
            "--bucket",
            "b",
            "--index",
            "i",
            "--table",
            "t",
            "--namespace",
            "tenant-a",
            "--output",
            str(out_file),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0][1] == "tenant-a"
    assert out_file.exists()
    assert "Exported 1 documents from namespace 'tenant-a'" in capsys.readouterr().out


def test_export_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())

    class DummyDynavec:
        def __init__(self, config, boto_session=None):
            self.config = config

        def export_namespace(self, output, namespace="default"):
            output.write('{"id": "doc1", "vector": [0.1, 0.2], "text": "hi", "metadata": {}}\n')
            return 1

    monkeypatch.setattr(cli, "Dynavec", DummyDynavec)

    result = cli.main(
        [
            "export",
            "--bucket",
            "b",
            "--index",
            "i",
            "--table",
            "t",
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert '{"id": "doc1"' in captured.out
    assert "Exported 1 documents from namespace 'default'" in captured.err


def test_import_from_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())

    in_file = tmp_path / "import.jsonl"
    in_file.write_text(
        '{"id": "doc1", "vector": [0.1, 0.2, 0.3], "text": "sample", "metadata": {"k": "v"}}\n',
        encoding="utf-8",
    )

    captured_config = []
    calls = []

    class DummyDynavec:
        def __init__(self, config, boto_session=None):
            self.config = config
            captured_config.append(config)

        def import_namespace(self, input_source, namespace="default", batch_size=100):
            calls.append((input_source, namespace, batch_size))
            return 1

    monkeypatch.setattr(cli, "Dynavec", DummyDynavec)

    result = cli.main(
        [
            "import",
            "--bucket",
            "b",
            "--index",
            "i",
            "--table",
            "t",
            "--namespace",
            "tenant-b",
            "--input",
            str(in_file),
            "--batch-size",
            "50",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0][1] == "tenant-b"
    assert calls[0][2] == 50
    # Inferred dimension from vector [0.1, 0.2, 0.3] -> 3
    assert captured_config[0].dimension == 3
    assert "Imported 1 documents into namespace 'tenant-b'" in capsys.readouterr().out


def test_import_from_stdin(monkeypatch, capsys):
    import io

    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.StringIO('{"id": "doc1", "vector": [0.1], "text": "sample", "metadata": {}}\n'),
    )

    class DummyDynavec:
        def __init__(self, config, boto_session=None):
            self.config = config

        def import_namespace(self, input_source, namespace="default", batch_size=100):
            return 1

    monkeypatch.setattr(cli, "Dynavec", DummyDynavec)

    result = cli.main(
        [
            "import",
            "--bucket",
            "b",
            "--index",
            "i",
            "--table",
            "t",
        ]
    )

    assert result == 0
    assert "Imported 1 documents into namespace 'default'" in capsys.readouterr().out


def test_export_failure_handles_exception(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())

    class DummyDynavec:
        def __init__(self, config, boto_session=None):
            self.config = config

        def export_namespace(self, output, namespace="default"):
            raise RuntimeError("S3 permission denied")

    monkeypatch.setattr(cli, "Dynavec", DummyDynavec)

    result = cli.main(
        [
            "export",
            "--bucket",
            "b",
            "--index",
            "i",
            "--table",
            "t",
        ]
    )

    assert result == 1
    assert "[FAIL] Export failed: S3 permission denied" in capsys.readouterr().err


def test_import_failure_handles_exception(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_session", lambda _profile, _region: _FakeSession())

    class DummyDynavec:
        def __init__(self, config, boto_session=None):
            self.config = config

        def import_namespace(self, input_source, namespace="default", batch_size=100):
            raise RuntimeError("DDB batch write failure")

    monkeypatch.setattr(cli, "Dynavec", DummyDynavec)

    result = cli.main(
        [
            "import",
            "--bucket",
            "b",
            "--index",
            "i",
            "--table",
            "t",
        ]
    )

    assert result == 1
    assert "[FAIL] Import failed: DDB batch write failure" in capsys.readouterr().err
