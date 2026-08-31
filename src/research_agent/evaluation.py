import re

from research_agent.agent import (
    run_agent,
)
from research_agent.retrieval import (
    DOCUMENT_INDEX,
)


# =========================================================
# Configuration
# =========================================================

TOP_K = 3


# =========================================================
# Retrieval test cases
# =========================================================

RETRIEVAL_TESTS = [
    {
        "query": (
            "How can ResNet be used "
            "as a feature extractor?"
        ),
        "expected_source": "resnet.txt",
    },
    {
        "query": (
            "What dimensionality does the "
            "ResNet18 feature representation have?"
        ),
        "expected_source": "resnet.txt",
    },
    {
        "query": (
            "What are residual connections?"
        ),
        "expected_source": "resnet.txt",
    },
    {
        "query": (
            "How does nearest-neighbour "
            "anomaly detection work?"
        ),
        "expected_source": "anomaly_detection.txt",
    },
    {
        "query": (
            "Why can anomaly detection models "
            "be trained only on normal images?"
        ),
        "expected_source": "anomaly_detection.txt",
    },
    {
        "query": (
            "How can feature representations "
            "be used for anomaly detection?"
        ),
        "expected_source": "anomaly_detection.txt",
    },
    {
        "query": (
            "What is the capital of France?"
        ),
        "expected_source": None,
    },
    {
        "query": (
            "Who won the latest Formula 1 race?"
        ),
        "expected_source": None,
    },
    {
        "query": (
            "How do I bake sourdough bread?"
        ),
        "expected_source": None,
    },
    {
        "query": (
            "What is the population of Japan?"
        ),
        "expected_source": None,
    },
]


# =========================================================
# Tool-selection test cases
# =========================================================

TOOL_SELECTION_TESTS = [
    {
        "query": (
            "What is 17.4 multiplied by 8.2?"
        ),
        "expected_tools": [
            "calculator",
        ],
    },
    {
        "query": (
            "What is 144 divided by 12?"
        ),
        "expected_tools": [
            "calculator",
        ],
    },
    {
        "query": (
            "What is the mean SNR in "
            "data/experiment.csv?"
        ),
        "expected_tools": [
            "analyze_csv",
        ],
    },
    {
        "query": (
            "What is the maximum accuracy in "
            "data/experiment.csv?"
        ),
        "expected_tools": [
            "analyze_csv",
        ],
    },
    {
        "query": (
            "Inspect data/experiment.csv "
            "and tell me its columns."
        ),
        "expected_tools": [
            "inspect_csv",
        ],
    },
    {
        "query": (
            "According to the available documents, "
            "how can ResNet be used as a feature extractor?"
        ),
        "expected_tools": [
            "search_documents",
        ],
    },
    {
        "query": (
            "According to the available documents, "
            "how does nearest-neighbour anomaly detection work?"
        ),
        "expected_tools": [
            "search_documents",
        ],
    },
    {
        "query": (
            "What is the mean SNR in "
            "data/experiment.csv multiplied by 2?"
        ),
        "expected_tools": [
            "analyze_csv",
            "calculator",
        ],
    },
    {
        "query": (
            "Take the maximum accuracy from "
            "data/experiment.csv and subtract 0.1."
        ),
        "expected_tools": [
            "analyze_csv",
            "calculator",
        ],
    },
    {
        "query": (
            "What are the latest developments "
            "in artificial intelligence?"
        ),
        "expected_tools": [
            "web_search",
        ],
    },
    {
        "query": (
            "What are some recent developments "
            "in large language models?"
        ),
        "expected_tools": [
            "web_search",
        ],
    },
]


# =========================================================
# Citation test cases
# =========================================================

CITATION_TESTS = [
    {
        "query": (
            "According to the available documents, "
            "how can ResNet be used as a feature extractor?"
        ),
        "expected_source": "resnet.txt",
    },
    {
        "query": (
            "According to the available documents, "
            "what dimensionality does the ResNet18 "
            "feature representation have?"
        ),
        "expected_source": "resnet.txt",
    },
    {
        "query": (
            "According to the available documents, "
            "how does nearest-neighbour anomaly detection work?"
        ),
        "expected_source": "anomaly_detection.txt",
    },
    {
        "query": (
            "According to the available documents, "
            "why can anomaly detection models "
            "use normal training images?"
        ),
        "expected_source": "anomaly_detection.txt",
    },
]


# =========================================================
# Helper: Citation extraction
# =========================================================

def extract_citations(
    answer: str,
) -> list[tuple[str, int | None, int]]:
    """
    Extract both supported citation formats.

    TXT:
        [Source: resnet.txt, Chunk: 0]

    PDF:
        [Source: paper.pdf, Page: 2, Chunk: 7]

    Returns:
        [
            (
                source,
                page_or_none,
                chunk_id
            )
        ]
    """

    citations = []

    pdf_pattern = (
        r"\[Source:\s*"
        r"([^,\]]+),\s*"
        r"Page:\s*(\d+),\s*"
        r"Chunk:\s*(\d+)\]"
    )

    pdf_matches = re.findall(
        pdf_pattern,
        answer,
    )

    for source, page, chunk_id in pdf_matches:

        citations.append(
            (
                source.strip(),
                int(page),
                int(chunk_id),
            )
        )

    txt_pattern = (
        r"\[Source:\s*"
        r"([^,\]]+),\s*"
        r"Chunk:\s*(\d+)\]"
    )

    txt_matches = re.findall(
        txt_pattern,
        answer,
    )

    for source, chunk_id in txt_matches:

        citations.append(
            (
                source.strip(),
                None,
                int(chunk_id),
            )
        )

    return citations


# =========================================================
# Helper: Extract chunks actually returned by RAG
# =========================================================

def extract_retrieved_chunks(
    tool_results,
) -> set[tuple[str, int | None, int]]:
    """
    Extract Source + optional Page + Chunk from
    actual search_documents tool results.
    """

    retrieved_chunks = set()

    for tool_result in tool_results:

        if tool_result.tool != "search_documents":
            continue

        text = tool_result.result

        result_blocks = re.split(
            r"\n\s*\n",
            text,
        )

        for block in result_blocks:

            source_match = re.search(
                r"Source:\s*([^\n]+)",
                block,
            )

            chunk_match = re.search(
                r"Chunk:\s*(\d+)",
                block,
            )

            page_match = re.search(
                r"Page:\s*(\d+)",
                block,
            )

            if (
                source_match is None
                or chunk_match is None
            ):
                continue

            source = (
                source_match
                .group(1)
                .strip()
            )

            chunk_id = int(
                chunk_match.group(1)
            )

            if page_match is not None:

                page = int(
                    page_match.group(1)
                )

            else:

                page = None

            retrieved_chunks.add(
                (
                    source,
                    page,
                    chunk_id,
                )
            )

    return retrieved_chunks


# =========================================================
# Helper: check citation source
# =========================================================

def citation_contains_source(
    citations,
    expected_source: str,
) -> bool:

    return any(
        source == expected_source
        for source, _, _ in citations
    )


# =========================================================
# Retrieval evaluation
# =========================================================

def evaluate_retrieval():

    top1_correct = 0
    topk_correct = 0

    relevant_tests = 0

    rejection_correct = 0
    rejection_tests = 0

    print(
        "\nIndividual Retrieval Tests"
    )
    print(
        "-" * 60
    )

    for test_case in RETRIEVAL_TESTS:

        query = (
            test_case["query"]
        )

        expected_source = (
            test_case[
                "expected_source"
            ]
        )

        results = (
            DOCUMENT_INDEX.retrieve(
                query=query,
                top_k=TOP_K,
            )
        )

        retrieved_sources = [
            result["source"]
            for result in results
        ]

        print(
            f"\nQuery: {query}"
        )

        print(
            "Expected source: "
            f"{expected_source}"
        )

        print(
            "Retrieved: "
            f"{retrieved_sources}"
        )

        # -------------------------------------------------
        # Irrelevant query
        # -------------------------------------------------

        if expected_source is None:

            rejection_tests += 1

            success = (
                len(results) == 0
            )

            if success:
                rejection_correct += 1

            print(
                "Rejection: "
                + (
                    "PASS"
                    if success
                    else "FAIL"
                )
            )

            continue

        # -------------------------------------------------
        # Relevant query
        # -------------------------------------------------

        relevant_tests += 1

        top1_success = (
            len(retrieved_sources) > 0
            and retrieved_sources[0]
            == expected_source
        )

        topk_success = (
            expected_source
            in retrieved_sources
        )

        if top1_success:
            top1_correct += 1

        if topk_success:
            topk_correct += 1

        print(
            "Top-1: "
            + (
                "PASS"
                if top1_success
                else "FAIL"
            )
        )

        print(
            f"Top-{TOP_K}: "
            + (
                "PASS"
                if topk_success
                else "FAIL"
            )
        )

    top1_accuracy = (
        top1_correct
        / relevant_tests
        if relevant_tests
        else 0
    )

    topk_accuracy = (
        topk_correct
        / relevant_tests
        if relevant_tests
        else 0
    )

    rejection_accuracy = (
        rejection_correct
        / rejection_tests
        if rejection_tests
        else 0
    )

    print(
        "\nRetrieval Metrics"
    )
    print(
        "-" * 60
    )

    print(
        "Top-1 Accuracy: "
        f"{top1_accuracy:.2%}"
    )

    print(
        f"Top-{TOP_K} Accuracy: "
        f"{topk_accuracy:.2%}"
    )

    print(
        "Irrelevant Query Rejection: "
        f"{rejection_accuracy:.2%}"
    )

    return {
        "top1_accuracy": (
            top1_accuracy
        ),
        "topk_accuracy": (
            topk_accuracy
        ),
        "rejection_accuracy": (
            rejection_accuracy
        ),
    }


# =========================================================
# Tool selection evaluation
# =========================================================
def normalize_tool_sequence(
    tools: list[str],
) -> list[str]:
    """
    Collapse consecutive calls to the same tool.

    Example:

    [
        "web_search",
        "web_search"
    ]

    becomes:

    [
        "web_search"
    ]

    This evaluates tool-routing decisions rather than
    the number of searches performed by a tool.
    """

    normalized = []

    for tool in tools:

        if (
            not normalized
            or normalized[-1] != tool
        ):

            normalized.append(
                tool
            )

    return normalized

def evaluate_tool_selection():

    loose_correct = 0
    exact_correct = 0

    unnecessary_tool_free = 0

    total = len(
        TOOL_SELECTION_TESTS
    )

    print(
        "\nIndividual Tool Tests"
    )
    print(
        "-" * 60
    )

    for test_case in TOOL_SELECTION_TESTS:

        query = (
            test_case["query"]
        )

        expected_tools = (
            test_case[
                "expected_tools"
            ]
        )

        result = run_agent(
            query,
            return_trace=True,
        )

        raw_used_tools = (
            result.trace.tools
        )

        used_tools = (
            normalize_tool_sequence(
                raw_used_tools
            )
        )

        # -------------------------------------------------
        # Loose accuracy
        #
        # Agent used all necessary tools,
        # but may have used additional tools.
        # -------------------------------------------------

        loose_success = all(
            tool in used_tools
            for tool in expected_tools
        )

        # -------------------------------------------------
        # Exact accuracy
        #
        # Required tools and tool order must match.
        # -------------------------------------------------

        exact_success = (
            used_tools
            == expected_tools
        )

        # -------------------------------------------------
        # No unnecessary tools
        # -------------------------------------------------

        unexpected_tools = [
            tool
            for tool in used_tools
            if tool not in expected_tools
        ]

        no_unnecessary_tools = (
            len(unexpected_tools) == 0
        )

        if loose_success:
            loose_correct += 1

        if exact_success:
            exact_correct += 1

        if no_unnecessary_tools:
            unnecessary_tool_free += 1

        print(
            f"\nQuery: {query}"
        )

        print(
            "Expected: "
            f"{expected_tools}"
        )

        print(
            "Used: "
            f"{used_tools}"
        )

        print(
            "Required Tools Present: "
            + (
                "PASS"
                if loose_success
                else "FAIL"
            )
        )

        print(
            "Exact Tool Sequence: "
            + (
                "PASS"
                if exact_success
                else "FAIL"
            )
        )

        print(
            "No Unnecessary Tools: "
            + (
                "PASS"
                if no_unnecessary_tools
                else "FAIL"
            )
        )

        print(
            "Raw calls: "
            f"{raw_used_tools}"
        )

        if unexpected_tools:

            print(
                "Unexpected tools: "
                f"{unexpected_tools}"
            )

    loose_accuracy = (
        loose_correct / total
        if total
        else 0
    )

    exact_accuracy = (
        exact_correct / total
        if total
        else 0
    )

    tool_efficiency = (
        unnecessary_tool_free
        / total
        if total
        else 0
    )

    print(
        "\nTool Selection Metrics"
    )
    print(
        "-" * 60
    )

    print(
        "Required Tool Accuracy: "
        f"{loose_accuracy:.2%}"
    )

    print(
        "Exact Tool Sequence Accuracy: "
        f"{exact_accuracy:.2%}"
    )

    print(
        "No-Unnecessary-Tool Rate: "
        f"{tool_efficiency:.2%}"
    )

    return {
        "required_tool_accuracy": (
            loose_accuracy
        ),
        "exact_tool_accuracy": (
            exact_accuracy
        ),
        "tool_efficiency": (
            tool_efficiency
        ),
    }


# =========================================================
# Citation evaluation
# =========================================================

def evaluate_citations():

    answers_with_citations = 0

    answers_with_expected_source = 0

    answers_with_all_valid_citations = 0

    total_citations = 0
    valid_citations = 0

    total = len(
        CITATION_TESTS
    )

    print(
        "\nIndividual Citation Tests"
    )
    print(
        "-" * 60
    )

    for test_case in CITATION_TESTS:

        query = (
            test_case["query"]
        )

        expected_source = (
            test_case[
                "expected_source"
            ]
        )

        result = run_agent(
            query,
            return_trace=True,
        )

        answer = (
            result.answer
        )

        citations = extract_citations(
            answer
        )

        retrieved_chunks = (
            extract_retrieved_chunks(
                result.trace.tool_results
            )
        )

        citation_present = (
            len(citations) > 0
        )

        expected_source_present = (
            citation_contains_source(
                citations,
                expected_source,
            )
        )

        citation_validity = []

        for citation in citations:

            total_citations += 1

            valid = (
                citation
                in retrieved_chunks
            )

            citation_validity.append(
                valid
            )

            if valid:
                valid_citations += 1

        all_valid = (
            len(citations) > 0
            and all(
                citation_validity
            )
        )

        if citation_present:
            answers_with_citations += 1

        if expected_source_present:
            answers_with_expected_source += 1

        if all_valid:
            answers_with_all_valid_citations += 1

        print(
            f"\nQuery: {query}"
        )

        print(
            "Expected source: "
            f"{expected_source}"
        )

        print(
            "Citations:"
        )

        if not citations:

            print(
                "- No citations"
            )

        for citation in citations:

            source, page, chunk = (
                citation
            )

            valid = (
                citation
                in retrieved_chunks
            )

            if page is None:

                citation_text = (
                    f"{source}, "
                    f"Chunk {chunk}"
                )

            else:

                citation_text = (
                    f"{source}, "
                    f"Page {page}, "
                    f"Chunk {chunk}"
                )

            print(
                f"- {citation_text}: "
                + (
                    "VALID"
                    if valid
                    else "INVALID"
                )
            )

        print(
            "Citation Present: "
            + (
                "PASS"
                if citation_present
                else "FAIL"
            )
        )

        print(
            "Expected Source Cited: "
            + (
                "PASS"
                if expected_source_present
                else "FAIL"
            )
        )

        print(
            "All Citations Valid: "
            + (
                "PASS"
                if all_valid
                else "FAIL"
            )
        )

    citation_presence = (
        answers_with_citations
        / total
        if total
        else 0
    )

    expected_source_accuracy = (
        answers_with_expected_source
        / total
        if total
        else 0
    )

    answer_validity = (
        answers_with_all_valid_citations
        / total
        if total
        else 0
    )

    citation_precision = (
        valid_citations
        / total_citations
        if total_citations
        else 0
    )

    print(
        "\nCitation Metrics"
    )
    print(
        "-" * 60
    )

    print(
        "Citation Presence: "
        f"{citation_presence:.2%}"
    )

    print(
        "Expected Source Accuracy: "
        f"{expected_source_accuracy:.2%}"
    )

    print(
        "Answer Citation Validity: "
        f"{answer_validity:.2%}"
    )

    print(
        "Citation Precision: "
        f"{citation_precision:.2%}"
    )

    return {
        "citation_presence": (
            citation_presence
        ),
        "expected_source_accuracy": (
            expected_source_accuracy
        ),
        "answer_citation_validity": (
            answer_validity
        ),
        "citation_precision": (
            citation_precision
        ),
    }


# =========================================================
# Dynamic PDF citation test
# =========================================================

def find_pdf_source():
    """
    Return the first PDF source found
    in the current document index.
    """

    for chunk in DOCUMENT_INDEX.chunks:

        source = (
            chunk["source"]
        )

        if source.lower().endswith(
            ".pdf"
        ):
            return source

    return None


def evaluate_pdf_citation():
    """
    Run a general PDF citation test if a PDF
    exists in the document index.

    This test mainly checks whether the agent
    returns source, page and chunk metadata
    correctly.
    """

    pdf_source = (
        find_pdf_source()
    )

    if pdf_source is None:

        print(
            "\nPDF Citation Test: SKIPPED"
        )

        print(
            "No PDF found in document index."
        )

        return None

    query = (
        "According to the available documents, "
        f"summarize one important point from "
        f"{pdf_source}. "
        "Use only information retrieved from "
        "that document."
    )

    result = run_agent(
        query,
        return_trace=True,
    )

    citations = extract_citations(
        result.answer
    )

    retrieved_chunks = (
        extract_retrieved_chunks(
            result.trace.tool_results
        )
    )

    pdf_citations = [
        citation
        for citation in citations
        if citation[0] == pdf_source
    ]

    valid_pdf_citations = [
        citation
        for citation in pdf_citations
        if (
            citation
            in retrieved_chunks
            and citation[1] is not None
        )
    ]

    success = (
        len(valid_pdf_citations) > 0
    )

    print(
        "\nPDF Citation Test"
    )
    print(
        "-" * 60
    )

    print(
        f"PDF: {pdf_source}"
    )

    print(
        f"Answer: {result.answer}"
    )

    print(
        "PDF citations: "
        f"{pdf_citations}"
    )

    print(
        "Result: "
        + (
            "PASS"
            if success
            else "FAIL"
        )
    )

    return success


# =========================================================
# Run all evaluations
# =========================================================

def run_all_evaluations():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESEARCH AGENT EVALUATION"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "1. RETRIEVAL"
    )

    print(
        "=" * 70
    )

    retrieval_metrics = (
        evaluate_retrieval()
    )

    # -----------------------------------------------------
    # Tool selection
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "2. TOOL SELECTION"
    )

    print(
        "=" * 70
    )

    tool_metrics = (
        evaluate_tool_selection()
    )

    # -----------------------------------------------------
    # Citations
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "3. CITATION VALIDATION"
    )

    print(
        "=" * 70
    )

    citation_metrics = (
        evaluate_citations()
    )

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "4. PDF CITATION"
    )

    print(
        "=" * 70
    )

    pdf_result = (
        evaluate_pdf_citation()
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nRetrieval"
    )

    print(
        "  Top-1 Accuracy: "
        f"{retrieval_metrics['top1_accuracy']:.2%}"
    )

    print(
        f"  Top-{TOP_K} Accuracy: "
        f"{retrieval_metrics['topk_accuracy']:.2%}"
    )

    print(
        "  Irrelevant Query Rejection: "
        f"{retrieval_metrics['rejection_accuracy']:.2%}"
    )

    print(
        "\nTool Selection"
    )

    print(
        "  Required Tool Accuracy: "
        f"{tool_metrics['required_tool_accuracy']:.2%}"
    )

    print(
        "  Exact Sequence Accuracy: "
        f"{tool_metrics['exact_tool_accuracy']:.2%}"
    )

    print(
        "  No-Unnecessary-Tool Rate: "
        f"{tool_metrics['tool_efficiency']:.2%}"
    )

    print(
        "\nCitations"
    )

    print(
        "  Citation Presence: "
        f"{citation_metrics['citation_presence']:.2%}"
    )

    print(
        "  Expected Source Accuracy: "
        f"{citation_metrics['expected_source_accuracy']:.2%}"
    )

    print(
        "  Answer Citation Validity: "
        f"{citation_metrics['answer_citation_validity']:.2%}"
    )

    print(
        "  Citation Precision: "
        f"{citation_metrics['citation_precision']:.2%}"
    )

    print(
        "\nPDF Citation"
    )

    if pdf_result is None:

        print(
            "  SKIPPED"
        )

    else:

        print(
            "  "
            + (
                "PASS"
                if pdf_result
                else "FAIL"
            )
        )


if __name__ == "__main__":

    run_all_evaluations()