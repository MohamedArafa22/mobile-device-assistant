from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:

    def __init__(self, documents: list[str]):
        self.documents = documents

        tokenized_documents = [
            tokenize(document)
            for document in documents
        ]

        self.index = BM25Okapi(
            tokenized_documents
        )

    def search(self, query: str) -> list[int]:
        """
        Return document indices ordered by BM25 relevance.
        """

        query_tokens = tokenize(query)

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return ranked_indices