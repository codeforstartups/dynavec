from dynavec.models import NamespaceStats


class FakeDocsStore:
    def __init__(self):
        self.data = {
            "knowledge-base": {
                "document_count": 3,
                "text_bytes": 120,
            },
            "research": {
                "document_count": 2,
                "text_bytes": 80,
            },
            "empty": {
                "document_count": 0,
                "text_bytes": 0,
            },
        }

    def list_namespaces(self, search=None):
        namespaces = sorted(self.data.keys())

        if search:
            search = search.lower()
            namespaces = [
                namespace
                for namespace in namespaces
                if search in namespace.lower()
            ]

        return namespaces

    def namespace_stats(self, namespace):
        return self.data.get(
            namespace,
            {
                "document_count": 0,
                "text_bytes": 0,
            },
        )


def test_namespace_stats_model():
    stats = NamespaceStats(
        namespace="knowledge-base",
        document_count=3,
        text_bytes=120,
    )

    assert stats.namespace == "knowledge-base"
    assert stats.document_count == 3
    assert stats.approximate_size_bytes == 120


def test_namespace_stats_to_dict():
    stats = NamespaceStats(
        namespace="research",
        document_count=2,
        text_bytes=80,
    )

    assert stats.to_dict() == {
        "namespace": "research",
        "document_count": 2,
        "approximate_size_bytes": 80,
    }


def test_fake_store_lists_namespaces():
    store = FakeDocsStore()

    assert store.list_namespaces() == [
        "empty",
        "knowledge-base",
        "research",
    ]


def test_fake_store_filters_namespaces():
    store = FakeDocsStore()

    assert store.list_namespaces("KNOWLEDGE") == [
        "knowledge-base"
    ]


def test_fake_store_returns_namespace_stats():
    store = FakeDocsStore()

    stats = store.namespace_stats("research")

    assert stats["document_count"] == 2
    assert stats["text_bytes"] == 80