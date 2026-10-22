"""Tests for the TF-IDF vectoriser."""

from conversaflow.nlu.vectorizer import TfidfVectorizer, cosine


def test_fit_transform_roundtrip():
    docs = ["hello world", "hello there", "goodbye world"]
    v = TfidfVectorizer().fit(docs)
    vec = v.transform("hello world")
    assert vec
    assert v.size > 0


def test_empty_corpus_raises():
    import pytest

    with pytest.raises(ValueError):
        TfidfVectorizer().fit([])


def test_no_features_raises():
    import pytest

    with pytest.raises(ValueError):
        TfidfVectorizer(min_df=100).fit(["a", "b"])


def test_transform_empty():
    v = TfidfVectorizer().fit(["hello world"])
    assert v.transform("zzzzz") == {}


def test_sublinear_toggle():
    v = TfidfVectorizer(sublinear_tf=False).fit(["hello hello", "world"])
    assert v.transform("hello") != {}


def test_transform_batch():
    v = TfidfVectorizer().fit(["a b", "c d"])
    assert len(v.transform_batch(["a", "c"])) == 2


def test_to_dict_from_dict():
    v = TfidfVectorizer().fit(["hello world", "foo bar"])
    restored = TfidfVectorizer.from_dict(v.to_dict())
    assert restored.vocabulary == v.vocabulary
    assert restored.idf == v.idf
    assert restored.transform("hello") == v.transform("hello")


def test_from_dict_malformed():
    import pytest

    with pytest.raises(ValueError):
        TfidfVectorizer.from_dict({"vocabulary": "x", "idf": []})


def test_cosine():
    assert cosine({1: 0.5}, {1: 0.5}) == 0.25
    assert cosine({}, {1: 1.0}) == 0.0


def test_ngram_max_one():
    v = TfidfVectorizer(ngram_max=1).fit(["hello world"])
    assert "b:hello world" not in v.vocabulary
