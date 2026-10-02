from typing import List, Tuple

from app.data.devices import DEVICE_DOCUMENTS
from app.utils.bm25_index import BM25Index
from app.utils.embeddings import EmbeddingModel


bm25_index = BM25Index(DEVICE_DOCUMENTS)
embedding_model = EmbeddingModel()

document_embeddings = (
    embedding_model.encode_documents(DEVICE_DOCUMENTS)
)


def dense_search(query: str) -> list[int]:
    """
    Return document indices ordered by dense similarity.
    """

    query_embedding = (
        embedding_model.encode_query(query)
    )

    similarities = (
        document_embeddings @ query_embedding
    )

    ranked_indices = sorted(
        range(len(similarities)),
        key=lambda i: similarities[i],
        reverse=True
    )

    return ranked_indices


def reciprocal_rank_fusion( sparse_ranking: list[int], dense_ranking: list[int], rrf_k: int = 60) -> dict[int, float]:

    scores: dict[int, float] = {}

    for rank, document_index in enumerate(
        sparse_ranking,
        start=1
    ):
        scores[document_index] = (
            scores.get(document_index, 0.0)
            + 1 / (rrf_k + rank)
        )

    for rank, document_index in enumerate(
        dense_ranking,
        start=1
    ):
        scores[document_index] = (
            scores.get(document_index, 0.0)
            + 1 / (rrf_k + rank)
        )

    return scores


def hybrid_search( query: str, k: int = 3 ) -> List[Tuple[str, float]]:

    sparse_ranking = bm25_index.search(
        query
    )

    dense_ranking = dense_search(
        query
    )

    fused_scores = reciprocal_rank_fusion(
        sparse_ranking,
        dense_ranking,
        rrf_k=60
    )

    ranked_documents = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for document_index, score in ranked_documents[:k]:

        results.append(
            (
                DEVICE_DOCUMENTS[document_index],
                score
            )
        )

    return results