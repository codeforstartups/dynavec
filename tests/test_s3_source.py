import io
from unittest.mock import MagicMock

from dynavec.ingest import S3Source, ingest


def test_s3_source_lists_and_reads_text_and_markdown():
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "documents/"},  # folder marker -> skipped
            {"Key": "documents/guide.md"},
            {"Key": "documents/notes.txt"},
        ],
        "IsTruncated": False,
    }

    md_content = (
        b"---\ntitle: Architecture Guide\nauthor: Shivam\n---\n\n# Header\nThis is the content."
    )
    txt_content = b"Simple text document."

    def mock_get_object(Bucket, Key):
        if Key == "documents/guide.md":
            return {
                "ContentType": "text/markdown",
                "Body": io.BytesIO(md_content),
            }
        elif Key == "documents/notes.txt":
            return {
                "ContentType": "text/plain",
                "Body": io.BytesIO(txt_content),
            }
        raise ValueError(f"Unknown key {Key}")

    client.get_object.side_effect = mock_get_object

    source = S3Source("my-bucket", prefix="documents/", s3_client=client)
    records = list(source)

    assert len(records) == 2

    # Verify Markdown record with parsed YAML front-matter
    md_rec = records[0]
    assert md_rec.id == "s3://my-bucket/documents/guide.md"
    assert "# Header\nThis is the content." in md_rec.text
    assert md_rec.metadata["source"] == "s3"
    assert md_rec.metadata["bucket"] == "my-bucket"
    assert md_rec.metadata["key"] == "documents/guide.md"
    assert md_rec.metadata["title"] == "Architecture Guide"
    assert md_rec.metadata["author"] == "Shivam"

    # Verify Plain text record
    txt_rec = records[1]
    assert txt_rec.id == "s3://my-bucket/documents/notes.txt"
    assert txt_rec.text == "Simple text document."
    assert txt_rec.metadata["source"] == "s3"
    assert txt_rec.metadata["key"] == "documents/notes.txt"


def test_s3_source_suffix_filtering():
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "data/readme.md"},
            {"Key": "data/report.pdf"},
            {"Key": "data/log.txt"},
        ],
        "IsTruncated": False,
    }

    client.get_object.return_value = {
        "ContentType": "text/markdown",
        "Body": io.BytesIO(b"# Only Markdown"),
    }

    source = S3Source("my-bucket", prefix="data/", suffix=".md", s3_client=client)
    records = list(source)

    assert len(records) == 1
    assert records[0].id == "s3://my-bucket/data/readme.md"


def test_s3_source_csv_routing():
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": [{"Key": "data/users.csv"}],
        "IsTruncated": False,
    }

    csv_data = b"name,role,team\nAlice,Engineer,Search\nBob,PM,Platform"
    client.get_object.return_value = {
        "ContentType": "text/csv",
        "Body": io.BytesIO(csv_data),
    }

    source = S3Source("my-bucket", s3_client=client)
    records = list(source)

    assert len(records) == 2
    assert records[0].id == "s3://my-bucket/data/users.csv#row1"
    assert records[0].text == "name: Alice, role: Engineer, team: Search"
    assert records[0].metadata["row"] == 1

    assert records[1].id == "s3://my-bucket/data/users.csv#row2"
    assert records[1].text == "name: Bob, role: PM, team: Platform"
    assert records[1].metadata["row"] == 2


def test_s3_source_pagination():
    client = MagicMock()
    client.list_objects_v2.side_effect = [
        {
            "Contents": [{"Key": "part1.txt"}],
            "IsTruncated": True,
            "NextContinuationToken": "token-123",
        },
        {
            "Contents": [{"Key": "part2.txt"}],
            "IsTruncated": False,
        },
    ]

    client.get_object.side_effect = [
        {"ContentType": "text/plain", "Body": io.BytesIO(b"Part 1 content")},
        {"ContentType": "text/plain", "Body": io.BytesIO(b"Part 2 content")},
    ]

    source = S3Source("paginated-bucket", s3_client=client)
    records = list(source)

    assert len(records) == 2
    assert records[0].id == "s3://paginated-bucket/part1.txt"
    assert records[1].id == "s3://paginated-bucket/part2.txt"

    # Verify ContinuationToken was passed in second call
    assert client.list_objects_v2.call_count == 2
    first_call_args = client.list_objects_v2.call_args_list[0][1]
    second_call_args = client.list_objects_v2.call_args_list[1][1]
    assert "ContinuationToken" not in first_call_args
    assert second_call_args["ContinuationToken"] == "token-123"


def test_s3_source_ingest_integration():
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": [{"Key": "article.txt"}],
        "IsTruncated": False,
    }
    client.get_object.return_value = {
        "ContentType": "text/plain",
        "Body": io.BytesIO(b"This is an article that will be ingested into Dynavec."),
    }

    db = MagicMock()
    upsert_result = MagicMock()
    upsert_result.count = 1
    db.upsert.return_value = upsert_result

    source = S3Source("test-bucket", s3_client=client)
    count = ingest(db, source, namespace="knowledge")

    assert count == 1
    db.upsert.assert_called_once()
    call_args = db.upsert.call_args[0][0]
    assert len(call_args) == 1
    doc = call_args[0]
    assert doc.id == "s3://test-bucket/article.txt#chunk0"
    assert doc.metadata["source_id"] == "s3://test-bucket/article.txt"
    assert doc.metadata["source"] == "s3"
