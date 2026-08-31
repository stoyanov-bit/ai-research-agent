import pytest

from research_agent.retrieval import (
    DOCUMENT_INDEX,
    chunk_text,
)


def test_chunk_text():

    text = " ".join(
        [
            f"word{i}"
            for i in range(100)
        ]
    )

    chunks = chunk_text(
        text,
        chunk_size=30,
        overlap=5,
    )

    assert len(chunks) > 1


def test_chunk_overlap():

    text = " ".join(
        [
            f"word{i}"
            for i in range(50)
        ]
    )

    chunks = chunk_text(
        text,
        chunk_size=20,
        overlap=5,
    )

    first_words = (
        chunks[0].split()
    )

    second_words = (
        chunks[1].split()
    )

    assert (
        first_words[-5:]
        == second_words[:5]
    )


def test_invalid_chunk_size():

    with pytest.raises(
        ValueError
    ):

        chunk_text(
            "some text",
            chunk_size=0,
        )


def test_invalid_overlap():

    with pytest.raises(
        ValueError
    ):

        chunk_text(
            "some text",
            chunk_size=10,
            overlap=10,
        )


def test_resnet_retrieval():

    results = (
        DOCUMENT_INDEX.retrieve(
            query=(
                "How can ResNet be used "
                "as a feature extractor?"
            ),
            top_k=3,
        )
    )

    sources = [
        result["source"]
        for result in results
    ]

    assert (
        "resnet.txt"
        in sources
    )


def test_irrelevant_retrieval():

    results = (
        DOCUMENT_INDEX.retrieve(
            query=(
                "Who won the Formula 1 "
                "world championship?"
            ),
            top_k=3,
        )
    )

    assert results == []