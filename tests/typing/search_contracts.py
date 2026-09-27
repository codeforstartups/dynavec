"""Consumer return types must follow the explain flag across search entry points."""

from typing import assert_type

from dynavec import Dynavec, ExplainedSearchResult, SearchResult


def check_search_contracts(db: Dynavec, explain: bool) -> None:
    assert_type(db.search("q"), list[SearchResult])
    assert_type(db.search("q", explain=False), list[SearchResult])
    assert_type(db.search("q", explain=True), ExplainedSearchResult)
    assert_type(db.search("q", explain=explain), list[SearchResult] | ExplainedSearchResult)

    ns = db.namespace("kb")
    assert_type(ns.search("q"), list[SearchResult])
    assert_type(ns.search("q", explain=False), list[SearchResult])
    assert_type(ns.search("q", explain=True), ExplainedSearchResult)
    assert_type(ns.search("q", explain=explain), list[SearchResult] | ExplainedSearchResult)

    assert_type(db.search_many(["q"]), list[list[SearchResult]])
    assert_type(db.search_many(["q"], explain=False), list[list[SearchResult]])
    assert_type(db.search_many(["q"], explain=True), list[ExplainedSearchResult])
    assert_type(
        db.search_many(["q"], explain=explain),
        list[list[SearchResult]] | list[ExplainedSearchResult],
    )

    assert_type(ns.search_many(["q"]), list[list[SearchResult]])
    assert_type(ns.search_many(["q"], explain=False), list[list[SearchResult]])
    assert_type(ns.search_many(["q"], explain=True), list[ExplainedSearchResult])
    assert_type(
        ns.search_many(["q"], explain=explain),
        list[list[SearchResult]] | list[ExplainedSearchResult],
    )
