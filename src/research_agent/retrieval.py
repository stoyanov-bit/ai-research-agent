from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Embedding model
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# Text loading
# =========================================================

def load_text_file(
    file_path: Path,
) -> list[dict]:
    """
    Load a plain text file.

    Returns one document unit.
    """

    text = file_path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        return []

    return [
        {
            "source": file_path.name,
            "page": None,
            "text": text,
        }
    ]


# =========================================================
# PDF loading
# =========================================================

def load_pdf_file(
    file_path: Path,
) -> list[dict]:
    """
    Extract text page by page from a PDF file.

    Each page becomes its own document unit so
    page information can later be preserved in
    citations and retrieval metadata.
    """

    reader = PdfReader(
        str(file_path)
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = (
            page.extract_text()
            or ""
        )

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "page": page_number,
                "text": text,
            }
        )

    return documents


# =========================================================
# General document loading
# =========================================================

def load_documents(
    directory: str,
) -> list[dict]:
    """
    Load all supported documents in a directory.

    Supported formats:
    - .txt
    - .pdf
    """

    directory_path = Path(
        directory
    )

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: "
            f"{directory}"
        )

    documents = []

    supported_extensions = {
        ".txt",
        ".pdf",
    }

    for file_path in sorted(
        directory_path.iterdir()
    ):

        if not file_path.is_file():
            continue

        extension = (
            file_path.suffix.lower()
        )

        if extension not in supported_extensions:
            continue

        try:

            if extension == ".txt":

                loaded_documents = (
                    load_text_file(
                        file_path
                    )
                )

            elif extension == ".pdf":

                loaded_documents = (
                    load_pdf_file(
                        file_path
                    )
                )

            else:

                loaded_documents = []

            documents.extend(
                loaded_documents
            )

        except Exception as error:

            print(
                f"Warning: Could not load "
                f"{file_path.name}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    return documents


# =========================================================
# Chunking
# =========================================================

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller "
            "than chunk_size."
        )

    words = text.split()

    if not words:
        return []

    chunks = []

    step_size = (
        chunk_size - overlap
    )

    for start in range(
        0,
        len(words),
        step_size,
    ):

        chunk_words = words[
            start:start + chunk_size
        ]

        if not chunk_words:
            continue

        chunks.append(
            " ".join(
                chunk_words
            )
        )

    return chunks


# =========================================================
# Document index
# =========================================================

class DocumentIndex:

    def __init__(
        self,
        directory: str,
    ):

        self.directory = directory

        self.chunks = []

        self.embeddings = None

        self.build()

    # -----------------------------------------------------
    # Build index
    # -----------------------------------------------------

    def build(
        self,
    ):

        documents = load_documents(
            self.directory
        )

        chunks = []

        global_chunk_id = 0

        for document in documents:

            document_chunks = chunk_text(
                document["text"]
            )

            for local_chunk_id, chunk in enumerate(
                document_chunks
            ):

                chunks.append(
                    {
                        "source": (
                            document[
                                "source"
                            ]
                        ),
                        "page": (
                            document[
                                "page"
                            ]
                        ),
                        "chunk_id": (
                            global_chunk_id
                        ),
                        "local_chunk_id": (
                            local_chunk_id
                        ),
                        "text": chunk,
                    }
                )

                global_chunk_id += 1

        if not chunks:

            raise ValueError(
                f"No supported documents "
                f"with readable text were found "
                f"in {self.directory}"
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )

        self.chunks = chunks

        self.embeddings = (
            embeddings
        )

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.25,
        source: str | None = None,
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = (
            embedding_model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )

        similarities = (
            cosine_similarity(
                query_embedding,
                self.embeddings,
            )[0]
        )

        best_indices = (
            np.argsort(
                similarities
            )[::-1]
        )

        results = []

        for index in best_indices:

            chunk = self.chunks[index]

            # -------------------------------------------------
            # Optional source filter
            # -------------------------------------------------

            if source is not None:

                if (
                    chunk["source"].lower()
                    != source.lower()
                ):
                    continue

            score = float(
                similarities[index]
            )

            if score < min_score:
                continue

            results.append(
                {
                    "source": (
                        chunk["source"]
                    ),
                    "page": (
                        chunk["page"]
                    ),
                    "chunk_id": (
                        chunk["chunk_id"]
                    ),
                    "text": (
                        chunk["text"]
                    ),
                    "score": score,
                }
            )

            if len(results) >= top_k:
                break

        return results

    # -----------------------------------------------------
    # Human / LLM readable search result
    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.25,
        source: str | None = None,
    ) -> str:

        # When the user explicitly selects a document,
        # retrieve the most relevant chunks from that
        # document even if similarity is relatively low.
        effective_min_score = (
            0.0
            if source is not None
            else min_score
        )

        results = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=effective_min_score,
            source=source,
        )

        if not results:

            if source is not None:

                return (
                    "No relevant information was found "
                    f"in the requested document: {source}"
                )

            return (
                "No sufficiently relevant "
                "information was found in "
                "the available documents."
            )

        formatted_results = []

        for result in results:

            lines = [
                (
                    f"Source: "
                    f"{result['source']}"
                ),
            ]

            if result["page"] is not None:

                lines.append(
                    f"Page: "
                    f"{result['page']}"
                )

            lines.extend(
                [
                    (
                        f"Chunk: "
                        f"{result['chunk_id']}"
                    ),
                    (
                        f"Similarity: "
                        f"{result['score']:.3f}"
                    ),
                    (
                        f"Text: "
                        f"{result['text']}"
                    ),
                ]
            )

            formatted_results.append(
                "\n".join(lines)
            )

        return "\n\n".join(
            formatted_results
        )


# =========================================================
# Global document index
# =========================================================

DOCUMENT_DIRECTORY = (
    "data/documents"
)

DOCUMENT_INDEX = DocumentIndex(
    DOCUMENT_DIRECTORY
)


# =========================================================
# Agent tool
# =========================================================

def search_documents(
    query: str,
    source: str | None = None,
    top_k: int = 3,
) -> str:
    """
    Search the local document collection.

    If source is provided, retrieval is restricted
    to that specific document.
    """

    return DOCUMENT_INDEX.search(
        query=query,
        source=source,
        top_k=top_k,
    )